import numpy as np
import pandas as pd
import pytest

from macro import current_state, regime4, release_periods, state_stats, transition_stats

T = pd.Timestamp
DAYS = pd.bdate_range("2024-01-02", "2024-03-29")


@pytest.fixture
def prices():
    return pd.DataFrame({"A": np.arange(1, len(DAYS) + 1, dtype=float),
                         "B": np.full(len(DAYS), 10.0)}, index=DAYS)


def price(prices, d):
    return prices.loc[d, "A"]


@pytest.fixture
def releases():
    return pd.DataFrame({"release_date": [T("2024-01-13"), T("2024-02-13"), T("2024-03-13")],
                         "state": ["CPI 상승", "CPI 하락", "CPI 상승"]})


def test_period_uses_release_close_and_drops_open_period(prices, releases):
    p = release_periods(prices, releases, "us", "2000-01-01")
    assert list(p.index) == [T("2024-01-15"), T("2024-02-13")]
    assert p.loc[T("2024-01-15"), "A"] == price(prices, T("2024-02-13")) / price(prices, T("2024-01-15")) - 1
    assert p.loc[T("2024-01-15"), "end"] == T("2024-02-13")
    assert p.loc[T("2024-01-15"), "state"] == "CPI 상승"


def test_kr_starts_after_release_day(prices, releases):
    p = release_periods(prices, releases, "kr", "2000-01-01")
    assert list(p.index) == [T("2024-01-15"), T("2024-02-14")]


def test_no_lookahead(prices, releases):
    us = release_periods(prices, releases, "us", "2000-01-01")
    kr = release_periods(prices, releases, "kr", "2000-01-01")
    assert (us["release_date"] <= us.index).all()
    assert (kr["release_date"] < kr.index).all()


def test_start_date_filters_periods(prices, releases):
    p = release_periods(prices, releases, "us", "2024-02-01")
    assert list(p.index) == [T("2024-02-13")]


def test_regime4_uses_latest_known_mfg():
    macro = pd.DataFrame({
        "indicator": ["cpi", "mfg", "mfg"],
        "ref_month": [T("2024-02-01"), T("2024-02-01"), T("2024-03-01")],
        "release_date": [T("2024-03-13"), T("2024-02-15"), T("2024-03-21")],
        "state": ["CPI 상승", "제조업 악화", "제조업 개선"],
        "value": [1.0, -5.0, 5.0], "signal": [3.0, -5.0, 5.0],
    })
    r = regime4(macro)
    assert r["state"].tolist() == ["스태그플레이션"]
    assert r["release_date"].tolist() == [T("2024-03-13")]


def test_state_stats_empty_state():
    p = pd.DataFrame({"A": [0.01, 0.02, -0.01], "state": ["회복", "회복", "침체"]})
    s = state_stats(p, ["회복", "과열", "침체"])
    assert s["by_state"]["과열"]["n"]["A"] == 0
    assert np.isnan(s["by_state"]["과열"]["mean"]["A"])
    assert s["by_state"]["회복"]["mean"]["A"] == pytest.approx(0.015)
    assert s["by_state"]["회복"]["hit"]["A"] == 1.0
    assert s["base_mean"]["A"] == pytest.approx(0.02 / 3)


def test_transition_stats():
    p = pd.DataFrame({"A": [0.1, 0.2, 0.3, 0.4, 0.5],
                      "state": ["과열", "과열", "침체", "과열", "침체"]})
    s = transition_stats(p, "과열")
    assert list(s["by_state"]) == ["과열→침체"]
    assert s["by_state"]["과열→침체"]["n"]["A"] == 2
    assert s["by_state"]["과열→침체"]["mean"]["A"] == pytest.approx(0.4)


def test_current_state():
    r = pd.DataFrame({"release_date": pd.date_range("2024-01-15", periods=14, freq="MS"),
                      "state": ["CPI 상승"] * 13 + ["CPI 하락"],
                      "value": range(14), "signal": range(14)})
    c = current_state(r)
    assert c["state"] == "CPI 하락"
    assert len(c["history"]) == 12
    assert c["history"][-1] == (r["release_date"].iloc[-1], "CPI 하락")
