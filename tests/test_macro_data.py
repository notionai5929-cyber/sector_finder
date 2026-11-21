import pandas as pd

from fetch_macro_data import INDICATORS, classify, release_date

T = pd.Timestamp


def test_indicators_and_state_labels():
    assert set(INDICATORS) == {"rate", "cpi", "pce", "mfg", "curve"}
    assert INDICATORS["rate"]["states"] == ["인상", "동결", "인하"]
    assert INDICATORS["cpi"]["states"] == ["CPI 상승", "CPI 하락"]
    assert INDICATORS["mfg"]["series"] == "GACDFSA066MSFRBPHI"


def test_release_dates():
    assert release_date("cpi", T("2024-01-01")) == T("2024-02-20")   # 화요일
    assert release_date("cpi", T("2023-12-01")) == T("2024-01-22")   # 20일 토 → 월
    assert release_date("pce", T("2024-01-01")) == T("2024-02-29")
    assert release_date("pce", T("2024-05-01")) == T("2024-06-28")   # 30일 일 → 28일 금
    assert release_date("mfg", T("2024-03-01")) == T("2024-03-21")
    assert release_date("rate", T("2024-06-01")) == T("2024-06-30")
    assert release_date("curve", T("2024-02-01")) == T("2024-02-29")


def test_classify_rate():
    v = pd.Series([5.33, 5.33, 5.08, 5.12], index=pd.date_range("2024-08-01", periods=4, freq="MS"))
    s = classify("rate", v)["state"].tolist()
    assert pd.isna(s[0])
    assert s[1:] == ["동결", "인하", "동결"]


def test_classify_cpi_uses_yoy_3m_change():
    idx = pd.date_range("2020-01-01", periods=18, freq="MS")
    v = pd.Series([100 * 1.01 ** i for i in range(18)], index=idx)   # 일정한 상승률 → YoY 일정
    out = classify("cpi", v)
    assert out["signal"].iloc[12:].round(6).nunique() == 1
    assert out["state"].iloc[:15].isna().all()                       # YoY 12개월 + 3개월 변화 필요
    assert out["state"].iloc[15:].notna().all()


def test_classify_mfg_direction():
    idx = pd.date_range("2024-01-01", periods=6, freq="MS")
    v = pd.Series([0, 5, 10, 20, 3, 1], index=idx, dtype=float)
    s = classify("mfg", v)["state"].tolist()
    assert s[3] == "제조업 개선"    # 20 vs 0
    assert s[5] == "제조업 악화"    # 1 vs 10


def test_drop_unreleased_rows():
    from fetch_macro_data import drop_unreleased
    m = pd.DataFrame({"indicator": ["curve", "curve", "cpi"],
                      "release_date": [T("2026-10-31"), T("2026-11-30"), T("2026-11-13")]})
    out = drop_unreleased(m, today=T("2026-11-21"))
    assert out["release_date"].tolist() == [T("2026-10-31"), T("2026-11-13")]
