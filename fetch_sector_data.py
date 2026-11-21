"""미국/한국 섹터 대표 ETF와 특징 ETF의 일별 주가 데이터를 data/{market}/ 폴더에 저장한다.

실행: python fetch_sector_data.py          # 미국 + 한국
      python fetch_sector_data.py us       # 미국만
      python fetch_sector_data.py kr       # 한국만
"""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_ROOT = Path(__file__).parent / "data"

# 각 ETF: (그룹, 분류, 표시이름, Yahoo 심볼, 설명)
# 표시이름은 표와 data/{market}/adj_close.csv의 열 이름으로 쓰인다.
MARKETS = {
    "us": {
        "name": "미장",
        "benchmark": "SPY",
        "min_years": 20,
        "tz": "America/New_York",
        "close_hour": 16,          # 장 마감 16:00 ET
        "etfs": [
            ("Sector", "Information Technology", "XLK", "XLK", "Information Technology"),
            ("Sector", "Financials", "XLF", "XLF", "Financials"),
            ("Sector", "Energy", "XLE", "XLE", "Energy"),
            ("Sector", "Health Care", "XLV", "XLV", "Health Care"),
            ("Sector", "Industrials", "XLI", "XLI", "Industrials"),
            ("Sector", "Consumer Discretionary", "XLY", "XLY", "Consumer Discretionary"),
            ("Sector", "Consumer Staples", "XLP", "XLP", "Consumer Staples"),
            ("Sector", "Utilities", "XLU", "XLU", "Utilities"),
            ("Sector", "Materials", "XLB", "XLB", "Materials"),
            ("Sector", "Real Estate", "VNQ", "VNQ", "Real Estate (XLRE는 2015년 상장이라 대체)"),
            ("Sector", "Communication Services", "VOX", "VOX", "Communication Services (XLC는 2018년 상장이라 대체)"),
            ("Other", "Benchmark", "SPY", "SPY", "S&P 500"),
            ("Other", "Dividend", "DVY", "DVY", "미국 고배당 (SCHD는 2011년 상장이라 대체)"),
            ("Other", "Style", "QQQ", "QQQ", "나스닥 100 (대형 성장/기술)"),
            ("Other", "Style", "IWM", "IWM", "러셀 2000 (소형주)"),
            ("Other", "Style", "IVW", "IVW", "S&P 500 성장주"),
            ("Other", "Style", "IVE", "IVE", "S&P 500 가치주"),
            ("Other", "Bond", "SHY", "SHY", "미국 국채 1-3년"),
            ("Other", "Bond", "IEF", "IEF", "미국 국채 7-10년"),
            ("Other", "Bond", "TLT", "TLT", "미국 국채 20년+"),
            ("Other", "Bond", "TIP", "TIP", "미국 물가연동국채"),
            ("Other", "Bond", "LQD", "LQD", "미국 투자등급 회사채"),
            ("Other", "Commodity", "GLD", "GLD", "금"),
            ("Other", "Commodity", "DBC", "DBC", "원자재 바스켓"),
        ],
    },
    "kr": {
        "name": "국장",
        "benchmark": "KODEX200",
        # 국내 ETF는 Yahoo 데이터가 2007-01부터라 20년 이상이 없음 → 15년으로 완화
        "min_years": 15,
        "tz": "Asia/Seoul",
        "close_hour": 16,          # 장 마감 15:30 KST (+여유)
        # 기간 기준(min_years) 예외: 표시이름 -> 사유. 상장 전 구간은 분석에서 자동으로 빠진다.
        "exempt": {
            "2차전지": "2018-09 상장. 15년 이상 된 2차전지 ETF가 없어 예외로 포함",
            "신재생": "2021-03 상장. 15년 이상 된 신재생에너지 ETF가 없어 예외로 포함",
            "태양광": "2023-06 상장. 국내 태양광 ETF가 이것뿐이라 예외로 포함",
            "원자력": "2022-06 상장. 15년 이상 된 원자력 ETF가 없어 예외로 포함",
        },
        "etfs": [
            ("Sector", "Information Technology", "200IT", "139260.KS", "TIGER 200 IT"),
            ("Sector", "Information Technology", "반도체", "091160.KS", "KODEX 반도체"),
            ("Sector", "Financials", "200금융", "139270.KS", "TIGER 200 금융"),
            ("Sector", "Financials", "은행", "091170.KS", "KODEX 은행"),
            ("Sector", "Financials", "증권", "102970.KS", "KODEX 증권"),
            ("Sector", "Financials", "보험", "140700.KS", "KODEX 보험"),
            ("Sector", "Energy", "에너지화학", "139250.KS", "TIGER 200 에너지화학"),
            ("Sector", "Materials", "철강소재", "139240.KS", "TIGER 200 철강소재"),
            ("Sector", "Industrials", "중공업", "139230.KS", "TIGER 200 중공업"),
            ("Sector", "Industrials", "건설", "139220.KS", "TIGER 200 건설"),
            ("Sector", "Industrials", "운송", "140710.KS", "KODEX 운송"),
            ("Sector", "Consumer Discretionary", "경기소비재", "139290.KS", "TIGER 200 경기소비재"),
            ("Sector", "Consumer Discretionary", "자동차", "091180.KS", "KODEX 자동차"),
            ("Sector", "Consumer Staples", "경기방어", "139280.KS", "TIGER 200 경기방어(필수소비재)"),
            ("Sector", "Industrials", "2차전지", "305720.KS", "KODEX 2차전지산업"),
            ("Sector", "Energy Theme", "신재생", "377990.KS", "TIGER Fn신재생에너지"),
            ("Sector", "Energy Theme", "태양광", "457990.KS", "PLUS 태양광&ESS"),
            ("Sector", "Energy Theme", "원자력", "434730.KS", "HANARO 원자력iSelect"),
            ("Other", "Benchmark", "KODEX200", "069500.KS", "KODEX 200 (코스피 200)"),
            ("Other", "Dividend", "고배당", "104530.KS", "KIWOOM 고배당 (구 KOSEF 고배당)"),
            ("Other", "Bond", "국채3년", "114260.KS", "KODEX 국고채3년"),
            ("Other", "Bond", "국채10년", "148070.KS", "KIWOOM 국고채10년 (구 KOSEF)"),
            ("Other", "Commodity", "골드", "132030.KS", "KODEX 골드선물(H)"),
            ("Other", "Commodity", "원유", "130680.KS", "TIGER 원유선물Enhanced(H)"),
            ("Other", "FX", "달러", "138230.KS", "KIWOOM 미국달러선물 (구 KOSEF)"),
        ],
    },
}


def fetch(symbol: str, tz: str, close_hour: int) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period="max", auto_adjust=False, actions=False)
    if df.empty:
        raise RuntimeError(f"{symbol}: 데이터가 비어 있음")
    df.index = df.index.tz_localize(None).normalize()
    df.index.name = "Date"
    # 장 마감 전에 받으면 당일 봉은 장중 값이므로 제외
    now = pd.Timestamp.now(tz=tz)
    if now.hour < close_hour:
        df = df[df.index < now.tz_localize(None).normalize()]
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]


GAP_DAYS = 30        # 이보다 긴 데이터 공백이 있으면 공백 이전 데이터는 버림
# 잘못 찍힌 값: 직전 60거래일 일간 변동성(표준편차)의 15배 넘게 튀었다가
# 다음 날 전일 가격 ±1% 안으로 돌아온 경우. 실제 급등락(예: 2020-03 폭락/반등)은
# 변동성이 함께 커져 있어 걸리지 않고, 저변동 상품(채권)의 비정상 체결만 걸린다.
SPIKE_SIGMA = 15
SPIKE_MIN = 0.05     # 최소 5% 이상 움직인 경우만 검사
REVERT = 0.01
VOL_WINDOW = 60
FFILL_LIMIT = 5      # 다른 종목은 거래됐는데 이 종목만 빠진 날은 최대 5일까지 직전 가격으로 채움


def clean_prices(s: pd.Series) -> tuple[pd.Series, list[str]]:
    """Yahoo 데이터 오류 정제. 수정 내역을 함께 반환한다."""
    notes = []
    valid = s.dropna()
    gaps = valid.index.to_series().diff().dt.days
    long_gaps = gaps[gaps > GAP_DAYS]
    if not long_gaps.empty:
        cut = long_gaps.index[-1]
        notes.append(f"{valid.index[0].date()}~{valid[:cut].index[-2].date()} 이후 {int(long_gaps.iloc[-1])}일 공백 → {cut.date()} 이전 제거")
        s = s[s.index >= cut]
    v = s.dropna()
    vol = v.pct_change().rolling(VOL_WINDOW, min_periods=20).std().shift(1)  # t일 이전 변동성
    for i in range(1, len(v) - 1):
        prev, cur, nxt = v.iloc[i - 1], v.iloc[i], v.iloc[i + 1]
        move = abs(cur / prev - 1)
        if (move > SPIKE_MIN and vol.iloc[i] == vol.iloc[i] and move > SPIKE_SIGMA * vol.iloc[i]
                and abs(nxt / prev - 1) < REVERT):
            notes.append(f"{v.index[i].date()} 튀는 값 {cur:,.0f} (전일 {prev:,.0f}, 익일 {nxt:,.0f}) → 전일 가격으로 대체")
            s.loc[v.index[i]] = prev
    first = s.first_valid_index()
    if first is not None:
        s = s.copy()
        s.loc[first:] = s.loc[first:].ffill(limit=FFILL_LIMIT)
    return s, notes


def update_data(market: str, log=print) -> pd.Timestamp:
    """한 시장의 전체 ETF를 상장일부터 다시 받아 data/{market}/에 저장하고 데이터 최종일을 반환한다.

    배당 등으로 과거 수정종가도 바뀌므로 매번 전체 기간을 새로 받는다.
    모든 종목을 먼저 받은 뒤에 저장하므로, 중간에 실패하면 기존 파일은 그대로 남는다.
    """
    cfg = MARKETS[market]
    data_dir = DATA_ROOT / market
    frames, meta = {}, []
    for group, category, label, symbol, desc in cfg["etfs"]:
        df = fetch(symbol, cfg["tz"], cfg["close_hour"])
        years = (df.index[-1] - df.index[0]).days / 365.25
        if years < cfg["min_years"] and label not in cfg.get("exempt", {}):
            raise RuntimeError(f"{label}({symbol}): 이력 {years:.1f}년 < {cfg['min_years']}년")
        frames[label] = df
        meta.append({
            "group": group,
            "category": category,
            "ticker": label,
            "symbol": symbol,
            "description": desc,
            "start_date": df.index[0].date(),
            "end_date": df.index[-1].date(),
            "years": round(years, 1),
            "rows": len(df),
            "note": cfg.get("exempt", {}).get(label, ""),
        })
        log(f"[{market}] {label:8s} {symbol:10s} {df.index[0].date()} ~ {df.index[-1].date()} ({years:.1f}y)")

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for label, df in frames.items():
        df.to_csv(raw_dir / f"{label}.csv", encoding="utf-8-sig")
    # 분석용 수정종가: 공백/튀는 값 정제 (raw/는 받은 그대로 보관)
    adj_close = pd.DataFrame({label: df["Adj Close"] for label, df in frames.items()})
    for m in meta:
        cleaned, notes = clean_prices(adj_close[m["ticker"]])
        adj_close[m["ticker"]] = cleaned
        valid = cleaned.dropna()
        m["start_date"], m["years"] = valid.index[0].date(), round((valid.index[-1] - valid.index[0]).days / 365.25, 1)
        m["cleaning"] = " / ".join(notes)
        for note in notes:
            log(f"[{market}] {m['ticker']} 정제: {note}")
    adj_close = adj_close.dropna(how="all")
    adj_close.to_csv(data_dir / "adj_close.csv", encoding="utf-8-sig")
    pd.DataFrame(meta).to_csv(data_dir / "etf_list.csv", index=False, encoding="utf-8-sig")
    log(f"[{market}] 저장 완료: {data_dir}")
    return adj_close.index[-1]


def main() -> None:
    markets = sys.argv[1:] or list(MARKETS)
    for m in markets:
        update_data(m)


if __name__ == "__main__":
    main()
