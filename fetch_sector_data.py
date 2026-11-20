"""GICS 11개 섹터 대표 ETF와 특징 ETF(모두 상장 20년 이상)의 일별 주가 데이터를 data/ 폴더에 저장한다.

실행: python fetch_sector_data.py
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

# 섹터 -> 대표 ETF (상장 20년 이상인 것 중 섹터 대표성이 가장 높은 것)
SECTOR_ETFS = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "VNQ",             # XLRE는 2015년 상장이라 제외
    "Communication Services": "VOX",  # XLC는 2018년 상장이라 제외
}

# 섹터는 아니지만 특징이 있는 ETF: 티커 -> (분류, 설명)
OTHER_ETFS = {
    "SPY": ("Benchmark", "S&P 500"),
    "DVY": ("Dividend", "미국 고배당 (SCHD는 2011년 상장이라 대체)"),
    "QQQ": ("Style", "나스닥 100 (대형 성장/기술)"),
    "IWM": ("Style", "러셀 2000 (소형주)"),
    "IVW": ("Style", "S&P 500 성장주"),
    "IVE": ("Style", "S&P 500 가치주"),
    "SHY": ("Bond", "미국 국채 1-3년"),
    "IEF": ("Bond", "미국 국채 7-10년"),
    "TLT": ("Bond", "미국 국채 20년+"),
    "TIP": ("Bond", "미국 물가연동국채"),
    "LQD": ("Bond", "미국 투자등급 회사채"),
    "GLD": ("Commodity", "금"),
    "DBC": ("Commodity", "원자재 바스켓"),
}

MIN_YEARS = 20
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"


def fetch(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="max", auto_adjust=False, actions=False)
    if df.empty:
        raise RuntimeError(f"{ticker}: 데이터가 비어 있음")
    df.index = df.index.tz_localize(None).normalize()
    df.index.name = "Date"
    # 미국장 마감(16:00 ET) 전에 받으면 당일 봉은 장중 값이므로 제외
    now_et = pd.Timestamp.now(tz="America/New_York")
    if now_et.hour < 16:
        df = df[df.index < now_et.tz_localize(None).normalize()]
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    meta, adj_close = [], {}

    targets = [("Sector", sector, ticker, sector) for sector, ticker in SECTOR_ETFS.items()]
    targets += [("Other", cat, ticker, desc) for ticker, (cat, desc) in OTHER_ETFS.items()]

    for group, category, ticker, desc in targets:
        df = fetch(ticker)
        years = (df.index[-1] - df.index[0]).days / 365.25
        if years < MIN_YEARS:
            raise RuntimeError(f"{ticker}: 이력 {years:.1f}년 < {MIN_YEARS}년")

        df.to_csv(RAW_DIR / f"{ticker}.csv")
        adj_close[ticker] = df["Adj Close"]
        meta.append({
            "group": group,
            "category": category,
            "ticker": ticker,
            "description": desc,
            "start_date": df.index[0].date(),
            "end_date": df.index[-1].date(),
            "years": round(years, 1),
            "rows": len(df),
        })
        print(f"{ticker:4s} {category:24s} {df.index[0].date()} ~ {df.index[-1].date()} ({years:.1f}y)")

    pd.DataFrame(meta).to_csv(DATA_DIR / "etf_list.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(adj_close).to_csv(DATA_DIR / "adj_close.csv")
    print(f"\n저장 완료: {DATA_DIR}")


if __name__ == "__main__":
    main()
