import numpy as np
import pandas as pd

import pytest


torch = pytest.importorskip("torch")

from alphagen.data.expression import Feature, RSI  # noqa: E402
from alphagen_qlib.stock_data import StockData, FeatureType  # noqa: E402


def _manual_rsi_simple(close: np.ndarray, period: int) -> np.ndarray:
    """
    Manual RSI matching alphagen.data.expression.RSI:
    - Use simple mean of gains/losses over the last `period` diffs (not Wilder smoothing)
    - RSI = 100 * up / (up + down); if (up+down) ~ 0 -> 50
    """
    close = np.asarray(close, dtype=np.float64)
    diffs = np.diff(close)  # length T-1
    out = np.empty(close.shape[0], dtype=np.float64)
    out[:] = np.nan
    if close.shape[0] <= period:
        return out
    for t in range(period, close.shape[0]):
        window = diffs[t - period : t]
        up = np.maximum(window, 0.0).mean()
        down = np.maximum(-window, 0.0).mean()
        total = up + down
        out[t] = 100.0 * up / total if total > 1e-9 else 50.0
    return out


def test_rsi_expression_values_match_manual():
    # --- build a tiny, controlled CLOSE series for the "effective" window ---
    effective_close = np.array([10.0, 11.0, 12.0, 11.0, 11.0], dtype=np.float64)
    period = 2

    # StockData indexing expects a full timeline including max_backtrack_days
    # The effective window is located at indices [mb : mb + n_days).
    mb = 10
    mf = 0
    n_days = effective_close.shape[0]
    total_days = n_days + mb + mf

    # Pad the backtrack part with the first close, so early RSI is stable.
    close_full = np.concatenate([np.full(mb, effective_close[0]), effective_close], axis=0)
    assert close_full.shape[0] == total_days

    # Build (T, F, S) tensor. FeatureType uses enum values as direct indices.
    n_features = len(list(FeatureType))
    n_stocks = 1
    data = torch.full((total_days, n_features, n_stocks), float("nan"), dtype=torch.float32)
    data[:, int(FeatureType.CLOSE), 0] = torch.tensor(close_full, dtype=torch.float32)

    dates = pd.date_range("2020-01-01", periods=total_days, freq="D")
    stock_ids = pd.Index(["TEST"])

    sd = StockData(
        instrument=["TEST"],
        start_time=str(dates[mb].date()),
        end_time=str(dates[mb + n_days - 1].date()),
        max_backtrack_days=mb,
        max_future_days=mf,
        features=list(FeatureType),
        device=torch.device("cpu"),
        preloaded_data=(data, dates, stock_ids),
    )

    expr = RSI(Feature(FeatureType.CLOSE), period)
    got = expr.evaluate(sd, period=slice(0, 1)).squeeze(-1).detach().cpu().numpy()

    # Manual expected values over the effective window, using the same backtrack padding.
    expected_full = _manual_rsi_simple(close_full, period=period)
    expected = expected_full[mb : mb + n_days]

    # Compare with NaN-aware logic
    assert got.shape == expected.shape
    nan_mask = np.isnan(expected)
    assert np.all(np.isnan(got)[nan_mask])
    assert np.allclose(got[~nan_mask], expected[~nan_mask], rtol=1e-5, atol=1e-5)

