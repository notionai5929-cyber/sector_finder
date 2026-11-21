"""매크로 발표일 기준 구간 수익률과 국면별 통계."""
import numpy as np
import pandas as pd

META = ["state", "release_date", "end"]
REGIME4 = {  # (제조업 국면, CPI 국면) -> 4국면
    ("제조업 개선", "CPI 하락"): "회복",
    ("제조업 개선", "CPI 상승"): "과열",
    ("제조업 악화", "CPI 상승"): "스태그플레이션",
    ("제조업 악화", "CPI 하락"): "침체",
}
REGIME4_STATES = ["회복", "과열", "스태그플레이션", "침체"]
REGIME4_GUIDE = {
    "회복": "경기는 좋아지는데 물가는 안정된 시기. 주식에 가장 우호적인 국면으로, 경기민감주·성장주·소형주가 강하다는 것이 일반론.",
    "과열": "경기도 좋고 물가도 오르는 시기. 원자재·에너지·산업재가 강하고, 금리 상승으로 채권과 고평가 성장주는 불리하다는 것이 일반론.",
    "스태그플레이션": "경기는 식는데 물가는 오르는 가장 어려운 국면. 주식·채권이 함께 약하기 쉬워 원자재·현금성 자산·필수소비재 같은 방어주가 상대적으로 낫다는 것이 일반론.",
    "침체": "경기도 식고 물가도 내려가는 시기. 금리 하락으로 국채가 강하고 방어주가 버티며, 후반에는 금리 인하 기대로 성장주가 먼저 반등한다는 것이 일반론.",
}
TRANSITION_GUIDE = ("국면이 바뀐 것은 새 CPI(와 그때까지의 제조업 지수)가 발표된 날 확인된다. "
                    "이 표는 그 발표일 종가에 갈아탔다면 다음 CPI 발표까지 어땠는지를 보여준다. "
                    "전환 횟수가 적어 우연의 영향이 크므로, 지표별·4국면 표와 같은 방향인지 함께 확인하는 것이 좋다.")


def release_periods(prices: pd.DataFrame, releases: pd.DataFrame, market: str, start: str) -> pd.DataFrame:
    """발표일 k의 기준 종가 → 발표일 k+1의 기준 종가 구간 수익률.

    기준 거래일: 미장 = 발표일 당일 또는 이후 첫 거래일(발표가 장 시작 전),
                 국장 = 발표일 다음 날 이후 첫 거래일(미국 발표가 한국 장 마감 뒤).
    아직 다음 발표가 없는 마지막 구간은 제외한다.
    """
    r = releases.dropna(subset=["state"]).sort_values("release_date")
    side = "right" if market == "kr" else "left"
    pos = prices.index.searchsorted(r["release_date"].values, side=side)
    r = r.assign(pos=pos)
    r = r[r["pos"] < len(prices)].drop_duplicates("pos", keep="first")
    r = r[prices.index[r["pos"]] >= pd.Timestamp(start)]
    if len(r) < 2:
        return pd.DataFrame(columns=list(prices.columns) + META)
    base = r["pos"].values[:-1]
    end = r["pos"].values[1:]
    rets = prices.iloc[end].values / prices.iloc[base].values - 1
    out = pd.DataFrame(rets, index=prices.index[base], columns=prices.columns)
    out["state"] = r["state"].values[:-1]
    out["release_date"] = r["release_date"].values[:-1]
    out["end"] = prices.index[end]
    return out


def regime4(macro: pd.DataFrame) -> pd.DataFrame:
    """CPI 발표일마다 경기(그때까지 발표된 최신 제조업 국면) × 물가(CPI 국면) 4국면."""
    cpi = macro[(macro["indicator"] == "cpi") & macro["state"].notna()].sort_values("release_date")
    mfg = macro[(macro["indicator"] == "mfg") & macro["state"].notna()].sort_values("release_date")
    merged = pd.merge_asof(cpi[["release_date", "state"]],
                           mfg[["release_date", "state"]].rename(columns={"state": "mfg"}),
                           on="release_date", direction="backward")
    merged["state"] = [REGIME4.get((m, c)) for m, c in zip(merged["mfg"], merged["state"])]
    return merged.dropna(subset=["state"])[["release_date", "state"]].reset_index(drop=True)


def _tickers(periods: pd.DataFrame) -> list[str]:
    return [c for c in periods.columns if c not in META]


def _stats(sub: pd.DataFrame, base_mean: pd.Series) -> dict:
    n = sub.notna().sum()
    mean, std = sub.mean(), sub.std()
    hit = (sub > 0).sum() / n.where(n > 0)
    t = (mean - base_mean) / (std / np.sqrt(n)).where(n > 1)
    return {k: v.to_dict() for k, v in {"mean": mean, "hit": hit, "n": n, "t": t}.items()}


def state_stats(periods: pd.DataFrame, states: list[str]) -> dict:
    """국면별 다음 구간 평균 상승률·상승확률·표본 수·t(전체 평균 대비)."""
    tick = _tickers(periods)
    data = periods[tick].astype(float)
    base_mean = data.mean()
    base_hit = (data > 0).sum() / data.notna().sum().where(lambda s: s > 0)
    return {
        "by_state": {s: _stats(data[periods["state"] == s], base_mean) for s in states},
        "base_mean": base_mean.to_dict(),
        "base_hit": base_hit.to_dict(),
    }


def transition_stats(periods: pd.DataFrame, current: str) -> dict:
    """직전 구간 국면이 current였다가 이번 구간에 다른 국면으로 바뀐 직후 구간의 성과."""
    tick = _tickers(periods)
    data = periods[tick].astype(float)
    base_mean = data.mean()
    prev = periods["state"].shift(1)
    moved = (prev == current) & periods["state"].notna() & (periods["state"] != current)
    by = {}
    for nxt in pd.unique(periods.loc[moved, "state"]):
        by[f"{current}→{nxt}"] = _stats(data[moved & (periods["state"] == nxt)], base_mean)
    base_hit = (data > 0).sum() / data.notna().sum().where(lambda s: s > 0)
    return {"by_state": by, "base_mean": base_mean.to_dict(), "base_hit": base_hit.to_dict()}


def current_state(releases: pd.DataFrame) -> dict:
    """가장 최근 발표의 국면과 최근 12개 국면 기록."""
    r = releases.dropna(subset=["state"]).sort_values("release_date")
    last = r.iloc[-1]
    return {
        "release_date": last["release_date"],
        "state": last["state"],
        "value": last.get("value"),
        "signal": last.get("signal"),
        "history": list(zip(r["release_date"].iloc[-12:], r["state"].iloc[-12:])),
    }
