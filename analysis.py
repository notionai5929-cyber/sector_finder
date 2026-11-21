"""기간별 ETF 상승률 순위 및 시차 교차상관 계산."""
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def load_prices(market: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / market / "adj_close.csv", index_col="Date",
                       parse_dates=True, encoding="utf-8-sig")


def load_etf_list(market: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / market / "etf_list.csv", encoding="utf-8-sig")


def period_rankings(prices: pd.DataFrame, start: str, n: int) -> list[dict]:
    """start 이후 거래일을 n일씩 묶고, 각 구간의 상승률을 높은 순서로 정렬해 반환한다.

    구간 상승률 = 구간 마지막 날 종가 / 구간 직전 거래일 종가 - 1
    (구간 사이에 빠지는 날이 없도록 직전 구간 종가를 기준으로 삼는다.)
    기준가가 없는 종목(상장 전)은 해당 구간에서 제외된다.
    마지막 구간은 n일보다 짧을 수 있다.
    """
    if n < 1:
        raise ValueError("n은 1 이상이어야 합니다")
    dates = prices.index
    first = dates.searchsorted(pd.Timestamp(start))
    periods = []
    for i in range(first, len(dates), n):
        j = min(i + n, len(dates)) - 1          # 구간 마지막 날 위치
        base = prices.iloc[i - 1] if i > 0 else prices.iloc[i]
        ret = (prices.iloc[j] / base - 1).dropna().sort_values(ascending=False)
        periods.append({
            "start": dates[i].date(),
            "end": dates[j].date(),
            "days": j - i + 1,
            "ranking": list(ret.items()),
        })
    return periods


def period_returns(prices: pd.DataFrame, start: str, n: int) -> pd.DataFrame:
    """period_rankings와 같은 구간 정의로 구간 상승률 표(행=구간 종료일, 열=티커)를 만든다.

    n일보다 짧은 마지막 구간은 제외한다(상관분석에서 구간 길이를 통일하기 위해).
    """
    dates = prices.index
    first = dates.searchsorted(pd.Timestamp(start))
    # 구간 경계: 첫 구간 직전 거래일 + 각 구간 마지막 날
    ends = list(range(first + n - 1, len(dates), n))
    if not ends:
        return pd.DataFrame(columns=prices.columns, dtype=float)
    bounds = [max(first - 1, 0)] + ends
    closes = prices.iloc[bounds]
    return (closes / closes.shift(1) - 1).iloc[1:]


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """두 배열의 스피어만 상관계수, 양측 p-value(정규근사), 표본 수."""
    mask = ~(np.isnan(x) | np.isnan(y))
    m = int(mask.sum())
    if m < 10:
        return float("nan"), float("nan"), m
    rx = pd.Series(x[mask]).rank().to_numpy()
    ry = pd.Series(y[mask]).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if abs(rho) >= 1:
        return rho, 0.0, m
    t = rho * math.sqrt((m - 2) / (1 - rho * rho))
    return rho, math.erfc(abs(t) / math.sqrt(2)), m


def lagged_spearman(returns: pd.DataFrame, lag: int) -> dict:
    """시차 교차상관: 열 티커의 t구간 상승률 vs 행 티커의 t+lag구간 상승률.

    반환: {"rho", "p", "n"} 각각 DataFrame(index=행=후행 티커, columns=열=선행 티커)
    """
    tickers = list(returns.columns)
    lead = returns.shift(lag).to_numpy()   # t+lag 행에 t 구간 값이 오도록
    follow = returns.to_numpy()
    rho = pd.DataFrame(index=tickers, columns=tickers, dtype=float)
    p, cnt = rho.copy(), rho.copy()
    for j, col in enumerate(tickers):
        for i, row in enumerate(tickers):
            rho.iat[i, j], p.iat[i, j], cnt.iat[i, j] = _spearman(lead[:, j], follow[:, i])
    return {"rho": rho, "p": p, "n": cnt}


def excess_returns(returns: pd.DataFrame, bench: str = "SPY") -> pd.DataFrame:
    """각 ETF 구간 상승률에서 벤치마크(SPY) 구간 상승률을 뺀 초과수익. 벤치마크 열은 제외."""
    return returns.drop(columns=bench).sub(returns[bench], axis=0)


def top_rank_flags(rank_returns: pd.DataFrame, top: float) -> pd.DataFrame:
    """구간마다 ETF들을 상승률 순으로 줄 세웠을 때 상위 top 비율 안에 든 칸을 True로 표시.

    순위표(period_rankings)와 같은 방식으로, 그 구간에 데이터가 있는 ETF끼리만 비교한다.
    상위 개수 = 반올림(해당 구간 ETF 수 × top), 최소 1개 (예: 24개×20% → 5위까지, 11개×20% → 2위까지).
    """
    ranks = rank_returns.rank(axis=1, ascending=False, method="first")
    cutoff = np.floor(rank_returns.notna().sum(axis=1) * top + 0.5).clip(lower=1)
    return ranks.le(cutoff, axis=0)


def conditional_after_top(returns: pd.DataFrame, rank_returns: pd.DataFrame,
                          lag: int, top: float = 0.2) -> dict:
    """조건부 분석: t구간 순위표에서 열 티커가 상위 top 비율에 들었을 때, 행 티커의 t+lag구간 성과.

    rank_returns: 순위를 매길 구간 상승률(원수익률, 순위표와 동일)
    returns:      성과를 측정할 구간 상승률(원수익률 또는 SPY 초과수익)
    순위는 t 시점에 알 수 있는 정보만 쓰므로 사후적 기준이 아니다.
    반환(각 DataFrame, index=행=후행, columns=열=선행):
      mean      조건부 평균 상승률
      hit       조건부 상승확률(>0)
      n         조건부 표본 수
      base_mean 같은 기간(선행 데이터가 있는 구간) 전체의 평균 상승률
      base_hit  같은 기간 전체의 상승확률
      t         (mean - base_mean) / (조건부 표준편차 / sqrt(n))
      entries   (Series) 열 티커가 상위에 든 횟수(그 뒤 t+lag구간이 있는 경우만)
    """
    flags = top_rank_flags(rank_returns, top).shift(lag, fill_value=False)
    avail_lead = rank_returns.shift(lag).notna()
    out = {k: pd.DataFrame(index=returns.columns, columns=returns.columns, dtype=float)
           for k in ("mean", "hit", "n", "base_mean", "base_hit", "t")}
    for col in returns.columns:
        avail = returns[avail_lead[col]]
        cond = returns[flags[col]]
        n = cond.notna().sum()
        mean, std = cond.mean(), cond.std()
        base_mean = avail.mean()
        out["mean"][col] = mean
        out["hit"][col] = (cond > 0).sum() / n.where(n > 0)
        out["n"][col] = n
        out["base_mean"][col] = base_mean
        out["base_hit"][col] = (avail > 0).sum() / avail.notna().sum().where(lambda s: s > 0)
        out["t"][col] = (mean - base_mean) / (std / np.sqrt(n)).where(n > 1)
    out["entries"] = flags[returns.columns].sum()
    return out
