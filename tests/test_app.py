import pandas as pd
import pytest

import app as A


@pytest.fixture
def client():
    return A.app.test_client()


@pytest.mark.parametrize("q", ["/?tab=macro", "/?tab=macro&market=kr", "/?tab=macro&start=2026-11-01",
                               "/?tab=macro&market=kr&basis=raw", "/?tab=time", "/?tab=zzz",
                               "/?tab=macro&start=abc"])
def test_pages_render(client, q):
    assert client.get(q).status_code == 200


def test_macro_tab_has_tables(client):
    h = client.get("/?tab=macro").get_data(as_text=True)
    assert h.count('class="macro-table"') == 7        # 지표 5 + 4국면 1 + 전환 1
    assert "현재 국면" in h
    assert 'class="tab on"' in h
    assert "<h2>시차 교차상관" not in h               # 다른 탭은 계산·표시하지 않음


def test_time_tab_unchanged(client):
    for q in ("/?tab=time", "/"):
        h = client.get(q).get_data(as_text=True)
        assert h.count("<h2>") == 6
        assert 'class="macro-table"' not in h


def test_tab_links_keep_other_params(client):
    h = client.get("/?tab=time&market=kr&start=2015-01-01").get_data(as_text=True)
    assert 'href="/?tab=macro&amp;market=kr&amp;start=2015-01-01"' in h or \
           ("tab=macro" in h and "market=kr&amp;start=2015-01-01" in h)


def test_refresh_keeps_tab_and_survives_macro_failure(client, monkeypatch):
    last = A.PRICES["us"].index[-1]
    monkeypatch.setattr(A, "update_data", lambda market, log=None: last)
    monkeypatch.setattr(A, "reload_data", lambda market=None: None)

    def boom(log=None):
        raise RuntimeError("FRED 접속 실패")
    monkeypatch.setattr(A, "update_macro", boom)
    r = client.post("/refresh", data={"back": "tab=macro&market=us", "market": "us"})
    assert r.status_code == 302
    loc = r.headers["Location"]
    assert "tab=macro" in loc
    h = client.get(loc).get_data(as_text=True)
    assert "데이터 갱신 완료" in h and "매크로 갱신 실패" in h


def test_transition_compares_against_all_periods():
    mac = A.macro_tables("us", "2000-01-01", "excess")
    assert mac["transition"]["stats"]["base_mean"] == mac["regime"]["stats"]["base_mean"]


def test_macro_tab_has_regime_guides(client):
    h = client.get("/?tab=macro").get_data(as_text=True)
    assert h.count('class="guide"') == 7              # 지표 5 + 4국면 + 전환
    for word in ("인상:", "스태그플레이션:", "금리차 축소:"):
        assert word in h
