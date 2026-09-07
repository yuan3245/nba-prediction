"""betting_math 的單元測試。

執行：.venv/bin/python -m pytest src/ -q
"""
import pytest

from betting_math import (
    american_to_decimal,
    expected_value,
    kelly_fraction,
    haversine_km,
)


# ---------- 賠率轉換 ----------

def test_american_to_decimal_positive():
    """+150：押 100 賺 150，decimal = 1 + 150/100"""
    assert american_to_decimal(150) == 2.5


def test_american_to_decimal_negative():
    """-150：押 150 才賺 100，decimal = 1 + 100/150"""
    assert american_to_decimal(-150) == pytest.approx(1.6667, abs=1e-4)


def test_american_to_decimal_even_money():
    """±100 都是平賠；0 是本實作自訂的預設值"""
    assert american_to_decimal(100) == 2.0
    assert american_to_decimal(-100) == 2.0
    assert american_to_decimal(0) == 2.0


def test_american_to_decimal_accepts_string():
    """實作有 float(ml)，字串型別的賠率也應該吃得下"""
    assert american_to_decimal("-150") == pytest.approx(1.6667, abs=1e-4)


# ---------- 期望值 ----------

def test_expected_value_positive_edge():
    """p=0.6、decimal=2.5 -> 0.6*1.5 - 0.4 = 0.5"""
    assert expected_value(0.6, 2.5) == pytest.approx(0.5)


def test_expected_value_negative_edge():
    """p=0.4、decimal=2.0 -> 0.4*1.0 - 0.6 = -0.2"""
    assert expected_value(0.4, 2.0) == pytest.approx(-0.2)


# ---------- Kelly ----------

def test_kelly_normal():
    """p=0.6、decimal=2.5 -> (0.9-0.4)/1.5 = 1/3"""
    assert kelly_fraction(0.6, 2.5) == pytest.approx(1 / 3)


def test_kelly_no_bet_when_ev_negative():
    """EV 為負時必須回 0（不下注）。

    這是本檔最重要的一條：若 max(f, 0.0) 這道保險被移除，
    函式會安靜地回傳負數，模擬器照樣跑完，ROI 就是錯的。
    """
    assert kelly_fraction(0.3, 2.0) == 0.0


def test_kelly_no_bet_when_odds_invalid():
    """decimal <= 1 代表沒有賠付空間，b <= 0，必須回 0 而非除以零"""
    assert kelly_fraction(0.9, 1.0) == 0.0
    assert kelly_fraction(0.9, 0.5) == 0.0


def test_kelly_equals_ev_over_b():
    """性質檢查：EV 為正時，Kelly 應等於 EV / b"""
    p, o = 0.62, 2.2
    assert kelly_fraction(p, o) == pytest.approx(expected_value(p, o) / (o - 1))


# ---------- 距離 ----------

def test_haversine_same_point_is_zero():
    """同一點的距離為 0"""
    assert haversine_km(42.366, -71.062, 42.366, -71.062) == pytest.approx(0.0)


def test_haversine_is_symmetric():
    """性質檢查：A->B 與 B->A 相等"""
    a = haversine_km(42.366, -71.062, 34.043, -118.267)
    b = haversine_km(34.043, -118.267, 42.366, -71.062)
    assert a == pytest.approx(b)


def test_haversine_known_distance():
    """TD Garden -> Crypto.com Arena，實測 4171.7 km，容許 +-30 km"""
    d = haversine_km(42.366, -71.062, 34.043, -118.267)
    assert d == pytest.approx(4170, abs=30)
