"""投注賠率與地理距離的純函式。

自 NBA_prediction_betting_clean.ipynb 抽出。抽離的動機是這些函式原本
定義在 notebook 內，無法被 import，因此也無法被測試。

對應測試：src/test_betting_math.py
"""
import numpy as np


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def american_to_decimal(ml):
    ml = float(ml)
    if ml > 0:   return 1 + ml / 100
    elif ml < 0: return 1 + 100 / abs(ml)
    return 2.0


def kelly_fraction(model_prob, decimal_odds):
    b = decimal_odds - 1
    if b <= 0: return 0.0
    f = (model_prob * b - (1 - model_prob)) / b
    return max(f, 0.0)


def expected_value(model_prob, decimal_odds):
    return model_prob * (decimal_odds - 1) - (1 - model_prob)
