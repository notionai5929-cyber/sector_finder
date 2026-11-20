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


@app.route("/")
def index():
    start = request.args.get("start") or DEFAULT_START
    basis = request.args.get("basis") if request.args.get("basis") in ("raw", "excess") else DEFAULT_BASIS
    error = None
    try:
        date.fromisoformat(start)
        n = int(request.args.get("n") or DEFAULT_N)
        top = int(request.args.get("top") or DEFAULT_TOP)
        if not 1 <= top <= 50:
            raise ValueError("상위 %는 1~50 사이여야 합니다")
        periods = period_rankings(PRICES, start, n)
        lags = lag_tables(start, n, basis)
        conds = cond_tables(start, n, basis, top)
    except ValueError as e:
        error, n, top, periods, lags, conds = f"입력값 오류: {e}", DEFAULT_N, DEFAULT_TOP, [], [], []

    max_rank = max((len(p["ranking"]) for p in periods), default=0)
    return render_template(
        "index.html",
        start=start, n=n, basis=basis, top=top,
        periods=periods, max_rank=max_rank, lags=lags, conds=conds,
        category=CATEGORY, error=error, min_n=MIN_N,
        data_end=PRICES.index[-1].date(),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
