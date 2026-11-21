"""미국 매크로 지표 5종을 FRED에서 받아 국면과 근사 발표일을 계산해 data/macro/에 저장한다.

실행: python fetch_macro_data.py
"""
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

MACRO_DIR = Path(__file__).parent / "data" / "macro"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

# 지표 키 -> 한글명, FRED 시리즈, 국면 라벨(좋은/올라가는 쪽 먼저)
INDICATORS = {
    "rate": {"name": "기준금리", "series": "FEDFUNDS", "states": ["인상", "동결", "인하"],
             "unit": "%", "desc": "연방기금금리 월평균, 전월 대비 ±0.1%p"},
    "cpi": {"name": "CPI", "series": "CPIAUCSL", "states": ["CPI 상승", "CPI 하락"],
            "unit": "% YoY", "desc": "소비자물가 전년 대비 상승률, 3개월 전 대비 방향"},
    "pce": {"name": "PCE", "series": "PCEPI", "states": ["PCE 상승", "PCE 하락"],
            "unit": "% YoY", "desc": "개인소비지출 물가 전년 대비 상승률, 3개월 전 대비 방향"},
    "mfg": {"name": "제조업 경기", "series": "GACDFSA066MSFRBPHI", "states": ["제조업 개선", "제조업 악화"],
            "unit": "", "desc": "필라델피아 연은 제조업지수(ISM PMI 대체), 3개월 전 대비 방향"},
    "curve": {"name": "장단기 금리차", "series": "T10Y2Y", "states": ["금리차 확대", "금리차 축소"],
              "unit": "%p", "desc": "10년-2년 국채 금리차 월말 값, 3개월 전 대비 방향"},
}
# 국면별 해석 메모(교과서적 일반론). 이 데이터에서 실제로 그랬는지는 화면의 표로 확인한다.
GUIDES = {
    "rate": {
        "인상": "연준이 물가를 잡으려고 돈줄을 죄는 시기. 미래 이익을 당겨 평가받는 성장주·부동산·소형주에 부담이 커진다. 금리가 오르면 이자 마진이 늘어나는 금융(은행)과, 경기가 아직 강해서 버티는 에너지·소재가 상대적으로 낫다는 것이 일반론.",
        "동결": "다음 움직임을 기다리는 시기. 이 국면 자체보다 직전이 인상이었는지(긴축 끝물) 인하였는지(완화 중)에 따라 의미가 달라진다. 시장은 다음 인상·인하 기대에 따라 움직인다.",
        "인하": "경기 둔화나 위기에 대응해 돈을 푸는 시기. 보통 경기가 이미 나빠진 뒤라 초반에는 채권·방어주가 강하고 주식이 더 빠지기도 한다(2001, 2008). 완화가 효과를 내면 성장주·소형주·부동산이 먼저 회복한다.",
    },
    "cpi": {
        "CPI 상승": "물가 상승세가 빨라지는 시기. 원자재·에너지 같은 실물자산과 가격 전가력이 있는 기업이 유리하고, 금리 인상 우려로 장기채·성장주(기술)가 불리하다는 것이 일반론.",
        "CPI 하락": "물가 상승세가 꺾이는 시기(디스인플레이션). 금리 하락 기대로 장기채와 성장주(기술)가 유리하고, 원자재·에너지는 힘이 빠지기 쉽다.",
    },
    "pce": {
        "PCE 상승": "연준이 공식 목표(2%)로 삼는 물가 지표가 오르는 시기. 해석은 CPI 상승과 같지만, 연준 정책 판단에 더 직접적으로 쓰여 금리 인상 압력으로 읽힌다.",
        "PCE 하락": "연준 목표 물가가 내려오는 시기. 금리 인하 여지가 생겨 채권·성장주에 우호적이라는 것이 일반론.",
    },
    "mfg": {
        "제조업 개선": "공장 주문·생산이 늘어나는 시기로, 경기를 앞서 보여주는 지표다. 산업재·소재·에너지·소형주·경기소비재 같은 경기민감주가 유리하다는 것이 일반론.",
        "제조업 악화": "공장 경기가 식는 시기. 경기와 덜 엮인 필수소비재·헬스케어·유틸리티 같은 방어주와 국채가 상대적으로 낫다는 것이 일반론.",
    },
    "curve": {
        "금리차 확대": "장기금리와 단기금리의 차이가 벌어지는 시기(가팔라짐). 경기 회복 기대로 장기금리가 오르는 '좋은 확대'와, 경기 악화로 연준이 단기금리를 급히 내리는 '나쁜 확대'(역전이 풀리는 시점, 과거 침체 직전에 자주 나타남)가 있어 구분해서 봐야 한다. 은행의 예대마진에는 유리하다.",
        "금리차 축소": "차이가 좁아지는 시기(평탄화). 연준 긴축으로 단기금리가 오르거나 성장 기대가 꺾일 때 나타난다. 0 아래로 내려가면 '역전'으로, 과거 대부분의 침체에 앞서 나타났지만 시점은 6개월~3년으로 들쭉날쭉했다.",
    },
}
# CPI 근사 발표일: 실제 발표(대개 10~15일, 과거에는 더 늦은 달도 있음)보다 빠르면
# 아직 나오지 않은 CPI로 구간을 나누게 되므로, 늦은 쪽인 20일로 잡는다.
# TODO: FRED API 키로 실제 발표일을 받아 교체 (README "나중에 할 일")
CPI_DAY = 20
RATE_BAND = 0.1   # 기준금리 동결로 볼 전월 대비 변화 범위(%p)
LOOKBACK = 3      # 방향 판단: 3개월 전 대비


def _next_weekday(d: pd.Timestamp) -> pd.Timestamp:
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def _last_weekday(d: pd.Timestamp) -> pd.Timestamp:
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def release_date(indicator: str, ref_month: pd.Timestamp) -> pd.Timestamp:
    """기준월(월초 날짜) 값의 근사 발표일. 공휴일은 고려하지 않는다."""
    m = pd.Timestamp(ref_month).normalize().replace(day=1)
    month_end = m + pd.offsets.MonthEnd(0)
    nxt = m + pd.offsets.MonthBegin(1)
    if indicator in ("rate", "curve"):
        return month_end
    if indicator == "cpi":
        return _next_weekday(nxt.replace(day=CPI_DAY))
    if indicator == "pce":
        return _last_weekday(nxt + pd.offsets.MonthEnd(0))
    if indicator == "mfg":
        first_thu = m + pd.Timedelta(days=(3 - m.weekday()) % 7)
        return first_thu + pd.Timedelta(weeks=2)
    raise ValueError(f"알 수 없는 지표: {indicator}")


def classify(indicator: str, values: pd.Series) -> pd.DataFrame:
    """월별 값으로 국면 판단 값(signal)과 국면(state)을 계산한다."""
    v = values.astype(float)
    if indicator == "rate":
        signal = v
        change = v.diff()
        up, down = "인상", "인하"
        state = pd.Series(pd.NA, index=v.index, dtype=object)
        state[change > RATE_BAND] = up
        state[change < -RATE_BAND] = down
        state[change.abs() <= RATE_BAND] = "동결"
    else:
        signal = v.pct_change(12, fill_method=None) * 100 if indicator in ("cpi", "pce") else v
        change = signal.diff(LOOKBACK)
        up, down = INDICATORS[indicator]["states"]
        state = pd.Series(pd.NA, index=v.index, dtype=object)
        state[change > 0] = up
        state[change <= 0] = down
    return pd.DataFrame({"value": v, "signal": signal, "state": state})


def _fetch_series(series: str) -> pd.Series:
    r = requests.get(FRED_CSV.format(series), timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    if df.shape[1] != 2:
        raise RuntimeError(f"{series}: 예상과 다른 응답")
    df.columns = ["date", "value"]
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values,
                  index=pd.to_datetime(df["date"]), name=series).dropna()
    if s.empty:
        raise RuntimeError(f"{series}: 데이터가 비어 있음")
    return s


def drop_unreleased(macro: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """발표일이 오늘 이후인 행(진행 중인 달의 값 등)은 아직 확정되지 않았으므로 뺀다."""
    today = pd.Timestamp.now().normalize() if today is None else today
    return macro[macro["release_date"] <= today].reset_index(drop=True)


def update_macro(log=print) -> pd.Timestamp:
    """5개 지표를 모두 받은 뒤 저장한다(중간에 실패하면 기존 파일 유지). 가장 최근 발표일을 반환."""
    raw = {key: _fetch_series(cfg["series"]) for key, cfg in INDICATORS.items()}
    rows = []
    for key, s in raw.items():
        monthly = s.resample("MS").last() if key == "curve" else s
        out = classify(key, monthly)
        out["indicator"] = key
        out["ref_month"] = out.index
        out["release_date"] = [release_date(key, m) for m in out.index]
        rows.append(out)
        log(f"[macro] {INDICATORS[key]['name']:8s} {s.index[0].date()} ~ {s.index[-1].date()}")
    macro = pd.concat(rows, ignore_index=True)[["indicator", "ref_month", "value", "signal", "state", "release_date"]]
    macro = drop_unreleased(macro)

    raw_dir = MACRO_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for key, s in raw.items():
        s.to_csv(raw_dir / f"{INDICATORS[key]['series']}.csv", header=True)
    macro.to_csv(MACRO_DIR / "macro.csv", index=False, encoding="utf-8-sig")
    log(f"[macro] 저장 완료: {MACRO_DIR}")
    return macro["release_date"].max()


def load_macro() -> pd.DataFrame:
    return pd.read_csv(MACRO_DIR / "macro.csv", encoding="utf-8-sig",
                       parse_dates=["ref_month", "release_date"])


if __name__ == "__main__":
    update_macro()
