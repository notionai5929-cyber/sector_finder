"""기간별 ETF 상승률 순위 웹 화면 (미장/국장 전환).

실행: python app.py  →  http://localhost:5000
"""
import threading
from datetime import date
from functools import lru_cache
from urllib.parse import parse_qsl, urlencode

import pandas as pd
from flask import Flask, redirect, render_template, request

from analysis import (conditional_after_top, excess_returns, lagged_spearman,
                      load_etf_list, load_prices, period_rankings, period_returns)
from fetch_macro_data import GUIDES, INDICATORS, MACRO_DIR, load_macro, update_macro
from fetch_sector_data import MARKETS, update_data
from macro import (META, REGIME4_GUIDE, REGIME4_STATES, TRANSITION_GUIDE, current_state, regime4, release_periods,
                   state_stats, transition_stats)

DEFAULT_START = "2000-01-01"
DEFAULT_N = 5
DEFAULT_MARKET = "us"      # "us" = 미장, "kr" = 국장
DEFAULT_BASIS = "excess"   # "raw" = 원수익률, "excess" = 벤치마크(SPY / KODEX200) 초과수익
DEFAULT_TOP = 20           # 조건부 표: 선행 ETF 상승률 상위 %
T_MIN = 1.96               # 추천 기준 |t| 고정값 (1.96 = 양측 5% 유의수준, README 검증 참고)
MIN_N = 10                 # 조건부 표: 표본이 이보다 적으면 회색 처리, 유의성 표시 안 함
LAGS = (1, 2, 3)           # 시간 딜레이 = 단위일수 * lag
TABS = ("time", "macro")   # 순위표 아래 탭: 단순 시간 영향성 / 매크로 분석
MIN_N_TRANSITION = 5       # 국면 전환표 추천에 필요한 최소 전환 횟수

app = Flask(__name__)
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True
REFRESH_LOCK = threading.Lock()


PRICES: dict[str, pd.DataFrame] = {}
ETF_INFO: dict[str, dict] = {}   # market -> {표시이름: {"category", "description", ...}}
MACRO: dict[str, pd.DataFrame] = {}  # "data" -> load_macro() (파일이 없으면 비어 있음)


def reload_data(market: str | None = None) -> None:
    """data/{market}/ 파일을 다시 읽고 계산 캐시를 비운다. market=None이면 전체."""
    for m in [market] if market else MARKETS:
        PRICES[m] = load_prices(m)
        ETF_INFO[m] = load_etf_list(m).set_index("ticker").to_dict("index")
    if (MACRO_DIR / "macro.csv").exists():
        MACRO["data"] = load_macro()
    for f in (analysis_returns, lag_tables, cond_tables, macro_tables):
        f.cache_clear()


def expected_last_day(market: str) -> date:
    """지금 받을 수 있는 최신 거래일(주말만 고려, 공휴일은 무시)."""
    cfg = MARKETS[market]
    now = pd.Timestamp.now(tz=cfg["tz"])
    d = now.normalize().tz_localize(None)
    if now.hour < cfg["close_hour"]:
        d -= pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d.date()


@lru_cache(maxsize=32)
def analysis_returns(market: str, start: str, n: int, basis: str):
    returns = period_returns(PRICES[market], start, n)
    return excess_returns(returns, MARKETS[market]["benchmark"]) if basis == "excess" else returns


@lru_cache(maxsize=32)
def lag_tables(market: str, start: str, n: int, basis: str) -> list[dict]:
    returns = analysis_returns(market, start, n, basis)
    tables = []
    for k in LAGS:
        res = lagged_spearman(returns, k)
        tables.append({
            "lag": k,
            "days": n * k,
            "tickers": list(returns.columns),
            # 모든 값은 [선행][후행] 순서로 접근
            **{key: df.to_dict() for key, df in res.items()},
        })
    return tables


@lru_cache(maxsize=32)
def cond_tables(market: str, start: str, n: int, basis: str, top: int) -> list[dict]:
    returns = analysis_returns(market, start, n, basis)
    rank_returns = period_returns(PRICES[market], start, n)  # 순위는 순위표와 같은 원수익률 기준
    tables = []
    for k in LAGS:
        res = conditional_after_top(returns, rank_returns, k, top / 100)
        tables.append({
            "lag": k,
            "days": n * k,
            "tickers": list(returns.columns),
            **{key: df.to_dict() for key, df in res.items()},
        })
    return tables


def pick_info(periods: list[dict], lag: int, top: int) -> dict | None:
    """순위표의 (마지막 - lag)번째 구간에서 상위 top%에 든 ETF.

    마지막 구간을 현재 진행 중인 구간으로 보고, lag구간 전의 상위 ETF가
    지금 구간에 미칠 영향을 조건부 표에서 찾기 위해 쓴다.
    """
    idx = len(periods) - 1 - lag
    if idx < 0:
        return None
    ranking = periods[idx]["ranking"]
    cutoff = max(1, int(len(ranking) * top / 100 + 0.5))   # top_rank_flags와 같은 규칙
    return {"tickers": [t for t, _ in ranking[:cutoff]],
            "start": periods[idx]["start"], "end": periods[idx]["end"],
            "current": periods[-1]}


def recommend(C: dict, leaders: list[str], side: str, t_min: float) -> list[dict]:
    """노란 테두리(선행 상위) 열에서 유의한 신호가 있는 후행 ETF를 고른다.

    side="buy":  N >= MIN_N, t >  t_min, 조건부 평균 > 0, 상승확률 > 50%
    side="sell": N >= MIN_N, t < -t_min, 조건부 평균 < 0, 상승확률 < 50%
    """
    buy = side == "buy"
    recs = {}
    for c in leaders:
        if c not in C["tickers"]:
            continue
        for r in C["tickers"]:
            m, t, h, n = C["mean"][c][r], C["t"][c][r], C["hit"][c][r], C["n"][c][r]
            if n < MIN_N or t != t:
                continue
            ok = (t > t_min and m > 0 and h > 0.5) if buy else (t < -t_min and m < 0 and h < 0.5)
            if ok:
                recs.setdefault(r, []).append({
                    "leader": c, "mean": m, "hit": h, "n": int(n), "t": t,
                    "base_mean": C["base_mean"][c][r], "base_hit": C["base_hit"][c][r],
                })
    sign = -1 if buy else 1   # 매수: 평균 높은 순, 매도: 평균 낮은 순
    out = [{"ticker": r, "reasons": sorted(v, key=lambda x: sign * x["mean"])} for r, v in recs.items()]
    return sorted(out, key=lambda x: sign * x["reasons"][0]["mean"])


def column_recs(stats: dict, state: str, side: str, min_n: int) -> list[dict]:
    """국면 열 하나에서 유의한 신호가 있는 ETF (기존 추천과 같은 기준)."""
    col = stats["by_state"].get(state)
    if not col:
        return []
    buy = side == "buy"
    out = []
    for tk, m in col["mean"].items():
        t, h, n = col["t"][tk], col["hit"][tk], col["n"][tk]
        if n < min_n or t != t:
            continue
        if (t > T_MIN and m > 0 and h > 0.5) if buy else (t < -T_MIN and m < 0 and h < 0.5):
            out.append({"ticker": tk, "mean": m, "hit": h, "n": int(n), "t": t,
                        "base_mean": stats["base_mean"][tk], "base_hit": stats["base_hit"][tk]})
    return sorted(out, key=lambda x: (-1 if buy else 1) * x["mean"])


def _to_basis(periods: pd.DataFrame, bench: str, basis: str) -> pd.DataFrame:
    """초과수익 기준이면 각 ETF 구간 수익률에서 벤치마크를 빼고 벤치마크 열은 뺀다."""
    if basis != "excess" or bench not in periods.columns:
        return periods
    tick = [c for c in periods.columns if c not in META and c != bench]
    out = periods.copy()
    out[tick] = out[tick].astype(float).sub(out[bench].astype(float), axis=0)
    return out.drop(columns=bench)


def _macro_table(key, title, desc, states, periods, releases, min_n=MIN_N, stats=None, guide=None) -> dict:
    stats = stats or state_stats(periods, states)
    cur = current_state(releases) if releases["state"].notna().any() else None
    state = cur["state"] if cur else None
    return {
        "key": key, "title": title, "desc": desc, "states": states, "stats": stats,
        "tickers": [c for c in periods.columns if c not in META],
        "n_periods": len(periods), "current": cur, "min_n": min_n, "guide": guide or {},
        "buy": column_recs(stats, state, "buy", min_n) if state else [],
        "sell": column_recs(stats, state, "sell", min_n) if state else [],
    }


@lru_cache(maxsize=32)
def macro_tables(market: str, start: str, basis: str) -> dict | None:
    """매크로 탭의 표들: 지표 5개, 4국면, 국면 전환. 매크로 파일이 없으면 None."""
    if "data" not in MACRO:
        return None
    m, prices = MACRO["data"], PRICES[market]
    bench = MARKETS[market]["benchmark"]
    tables = []
    for key, cfg in INDICATORS.items():
        rel = m[m["indicator"] == key]
        periods = _to_basis(release_periods(prices, rel, market, start), bench, basis)
        tables.append(_macro_table(key, f"{cfg['name']} 국면", cfg["desc"], cfg["states"], periods, rel,
                                   guide=GUIDES[key]))
    r4 = regime4(m)
    p4 = _to_basis(release_periods(prices, r4, market, start), bench, basis)
    regime = _macro_table("regime4", "경기 × 물가 4국면",
                          "경기 = 제조업 경기 방향, 물가 = CPI 방향. CPI 발표일 기준 구간",
                          REGIME4_STATES, p4, r4, guide=REGIME4_GUIDE)
    trans = None
    if regime["current"]:
        cur4 = regime["current"]["state"]
        keys = [f"{cur4}→{s}" for s in REGIME4_STATES if s != cur4]
        tr = transition_stats(p4, cur4)   # 평소 평균 = 전체 구간 평균
        tr["by_state"] = {k: tr["by_state"].get(k, state_stats(p4.iloc[0:0], ["-"])["by_state"]["-"]) for k in keys}
        trans = _macro_table("transition", f"국면 전환: {cur4} → 다른 국면",
                             "직전 구간이 현재 국면이었다가 다른 국면으로 바뀐 직후 1개월(다음 CPI 발표까지) 성과",
                             keys, p4, r4, MIN_N_TRANSITION, stats=tr, guide={"읽는 법": TRANSITION_GUIDE})
        trans["n_periods"] = int(sum(max(v["n"].values(), default=0) for v in tr["by_state"].values()))
        trans["current"] = None
        trans["moves"] = {k: {"buy": column_recs(trans["stats"], k, "buy", MIN_N_TRANSITION),
                              "sell": column_recs(trans["stats"], k, "sell", MIN_N_TRANSITION)}
                          for k in keys}
    return {"tables": tables, "regime": regime, "transition": trans}


@app.route("/refresh", methods=["POST"])
def refresh():
    """데이터를 새로 받아 저장하고 서버 메모리에 반영한 뒤 보던 화면으로 돌아간다."""
    back = request.form.get("back", "")
    params = dict(parse_qsl(back))
    market = request.form.get("market") if request.form.get("market") in MARKETS else DEFAULT_MARKET
    params["market"] = market
    if not REFRESH_LOCK.acquire(blocking=False):
        params["msg"] = "이미 갱신 중입니다. 잠시 후 다시 확인하세요."
    else:
        try:
            before = PRICES[market].index[-1].date()
            last = update_data(market, log=lambda m: app.logger.info(m))
            msg = f"{MARKETS[market]['name']} 데이터 갱신 완료: 최종일 {before} → {last.date()}"
            try:
                rel = update_macro(log=lambda m: app.logger.info(m))
                msg += f" · 매크로 갱신 완료(최근 발표 {rel.date()})"
            except Exception as e:  # FRED 오류: 기존 매크로 유지, ETF 갱신은 그대로
                msg += f" · 매크로 갱신 실패(기존 매크로 유지): {e}"
            reload_data(market)
            params["msg"] = msg
        except Exception as e:  # 네트워크 오류 등: 기존 데이터 유지
            params["msg"] = f"데이터 갱신 실패(기존 데이터 유지): {e}"
        finally:
            REFRESH_LOCK.release()
    return redirect("/?" + urlencode(params))


@app.route("/")
def index():
    market = request.args.get("market") if request.args.get("market") in MARKETS else DEFAULT_MARKET
    prices = PRICES[market]
    start = request.args.get("start") or DEFAULT_START
    basis = request.args.get("basis") if request.args.get("basis") in ("raw", "excess") else DEFAULT_BASIS
    tab = request.args.get("tab") if request.args.get("tab") in TABS else TABS[0]
    error, lags, conds, mac = None, [], [], None
    try:
        date.fromisoformat(start)
        n = int(request.args.get("n") or DEFAULT_N)
        top = int(request.args.get("top") or DEFAULT_TOP)
        if not 1 <= top <= 50:
            raise ValueError("상위 %는 1~50 사이여야 합니다")
        periods = period_rankings(prices, start, n)
        if tab == "time":
            lags = lag_tables(market, start, n, basis)
            conds = cond_tables(market, start, n, basis, top)
        else:
            mac = macro_tables(market, start, basis)
    except ValueError as e:
        error, n, top, periods, lags, conds, mac = (
            f"입력값 오류: {e}", DEFAULT_N, DEFAULT_TOP, [], [], [], None)

    picks = [pick_info(periods, k, top) for k in LAGS] if conds else []
    recs = [{"buy": recommend(C, p["tickers"], "buy", T_MIN) if p else [],
             "sell": recommend(C, p["tickers"], "sell", T_MIN) if p else []}
            for C, p in zip(conds, picks)]
    max_rank = max((len(p["ranking"]) for p in periods), default=0)
    return render_template(
        "index.html",
        market=market, markets=MARKETS, bench=MARKETS[market]["benchmark"],
        etf_info=ETF_INFO[market],
        start=start, n=n, basis=basis, top=top, t_min=T_MIN,
        periods=periods, max_rank=max_rank, lags=lags, conds=conds,
        picks=picks, recs=recs, tab=tab, mac=mac, indicators=INDICATORS,
        tab_query={t: urlencode({"tab": t, **{k: v for k, v in request.args.items() if k not in ("msg", "tab")}})
                   for t in TABS},
        error=error, min_n=MIN_N,
        msg=request.args.get("msg"), expected_end=expected_last_day(market),
        query=urlencode({k: v for k, v in request.args.items() if k != "msg"}),
        switch_query={m: urlencode({**{k: v for k, v in request.args.items() if k != "msg"}, "market": m})
                      for m in MARKETS},
        data_end=prices.index[-1].date(),
    )


reload_data()  # 서버 시작 시 데이터 로드

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
