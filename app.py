"""기간별 ETF 상승률 순위 웹 화면.

실행: python app.py  →  http://localhost:5000
"""
from datetime import date
from functools import lru_cache

from flask import Flask, render_template, request

from analysis import (conditional_after_top, excess_returns, lagged_spearman,
                      load_etf_list, load_prices, period_rankings, period_returns)

DEFAULT_START = "2000-01-01"
DEFAULT_N = 5
DEFAULT_BASIS = "excess"   # "raw" = 원수익률, "excess" = SPY 초과수익
DEFAULT_TOP = 20           # 조건부 표: 선행 ETF 상승률 상위 %
DEFAULT_TMIN = 1.96        # 추천 기준 |t| (1.96 = 양측 5% 유의수준)
MIN_N = 10                 # 조건부 표: 표본이 이보다 적으면 회색 처리, 유의성 표시 안 함
LAGS = (1, 2, 3)           # 시간 딜레이 = 단위일수 * lag

app = Flask(__name__)
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True
PRICES = load_prices()
CATEGORY = dict(zip(load_etf_list()["ticker"], load_etf_list()["category"]))


@lru_cache(maxsize=32)
def analysis_returns(start: str, n: int, basis: str):
    returns = period_returns(PRICES, start, n)
    return excess_returns(returns) if basis == "excess" else returns


@lru_cache(maxsize=32)
def lag_tables(start: str, n: int, basis: str) -> list[dict]:
    returns = analysis_returns(start, n, basis)
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
def cond_tables(start: str, n: int, basis: str, top: int) -> list[dict]:
    returns = analysis_returns(start, n, basis)
    rank_returns = period_returns(PRICES, start, n)  # 순위는 순위표와 같은 원수익률 기준
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


@app.route("/")
def index():
    start = request.args.get("start") or DEFAULT_START
    basis = request.args.get("basis") if request.args.get("basis") in ("raw", "excess") else DEFAULT_BASIS
    error = None
    try:
        date.fromisoformat(start)
        n = int(request.args.get("n") or DEFAULT_N)
        top = int(request.args.get("top") or DEFAULT_TOP)
        t_min = float(request.args.get("tmin") or DEFAULT_TMIN)
        if t_min < 0:
            raise ValueError("t 기준은 0 이상이어야 합니다")
        if not 1 <= top <= 50:
            raise ValueError("상위 %는 1~50 사이여야 합니다")
        periods = period_rankings(PRICES, start, n)
        lags = lag_tables(start, n, basis)
        conds = cond_tables(start, n, basis, top)
    except ValueError as e:
        error, n, top, t_min, periods, lags, conds = (
            f"입력값 오류: {e}", DEFAULT_N, DEFAULT_TOP, DEFAULT_TMIN, [], [], [])

    picks = [pick_info(periods, k, top) for k in LAGS] if conds else []
    recs = [{"buy": recommend(C, p["tickers"], "buy", t_min) if p else [],
             "sell": recommend(C, p["tickers"], "sell", t_min) if p else []}
            for C, p in zip(conds, picks)]
    max_rank = max((len(p["ranking"]) for p in periods), default=0)
    return render_template(
        "index.html",
        start=start, n=n, basis=basis, top=top, t_min=t_min,
        periods=periods, max_rank=max_rank, lags=lags, conds=conds,
        picks=picks, recs=recs,
        category=CATEGORY, error=error, min_n=MIN_N,
        data_end=PRICES.index[-1].date(),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
