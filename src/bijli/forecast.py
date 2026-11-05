from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATE = "date"
TARGET = "energy_met_mu"
FEATURES = ["dow", "month", "dayofyear_sin", "dayofyear_cos", "is_weekend",
            "lag_1", "lag_7", "lag_14", "lag_28", "mean_7", "mean_28", "std_28"]


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {DATE, "state", TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    df[DATE] = pd.to_datetime(df[DATE], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[DATE, "state", TARGET])
    df = df[df[TARGET] > 0].copy()
    return df.sort_values(["state", DATE]).drop_duplicates(["state", DATE], keep="last")


def state_series(df: pd.DataFrame, state: str) -> pd.Series:
    s = df.loc[df.state.eq(state), [DATE, TARGET]].set_index(DATE)[TARGET].sort_index()
    # Gaps remain missing. Lag calculations use calendar reindexing to avoid treating
    # yesterday's observation as yesterday when source reports have skipped dates.
    return s[~s.index.duplicated(keep="last")].asfreq("D")


def make_features(series: pd.Series) -> pd.DataFrame:
    x = pd.DataFrame(index=series.index)
    x["dow"] = x.index.dayofweek
    x["month"] = x.index.month
    doy = x.index.dayofyear
    x["dayofyear_sin"] = np.sin(2 * np.pi * doy / 365.25)
    x["dayofyear_cos"] = np.cos(2 * np.pi * doy / 365.25)
    x["is_weekend"] = (x.index.dayofweek >= 5).astype(int)
    for lag in (1, 7, 14, 28):
        x[f"lag_{lag}"] = series.shift(lag)
    # Shift before rolling: at date t no target value from t is used.
    prior = series.shift(1)
    x["mean_7"] = prior.rolling(7, min_periods=5).mean()
    x["mean_28"] = prior.rolling(28, min_periods=20).mean()
    x["std_28"] = prior.rolling(28, min_periods=20).std()
    return x.replace([np.inf, -np.inf], np.nan)


def metric_pack(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    denom = np.maximum(np.abs(y), 1e-8)
    return {"MAE (MU/day)": float(mean_absolute_error(y, pred)),
            "RMSE (MU/day)": float(np.sqrt(mean_squared_error(y, pred))),
            "MAPE (%)": float(np.mean(np.abs((y - pred) / denom)) * 100)}


def seasonal_naive(series: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    vals = []
    for d in dates:
        v = series.get(d - pd.Timedelta(days=7), np.nan)
        if pd.isna(v):
            v = series.loc[:d - pd.Timedelta(days=1)].dropna().tail(1)
            v = float(v.iloc[0]) if len(v) else np.nan
        vals.append(v)
    return np.asarray(vals, dtype=float)


@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    metrics: dict[str, dict[str, float]]
    anomaly_dates: list[pd.Timestamp]
    calibration_coverage: float
    model_name: str


def fit_forecast(series: pd.Series, horizon: int = 7, alpha: float = .10) -> ForecastResult:
    series = series.sort_index().asfreq("D")
    frame = make_features(series).assign(target=series).dropna()
    if len(frame) < 180:
        raise ValueError("Need at least 180 usable daily observations for training and calibration.")
    # Hold out the latest 20% for an honest chronological evaluation. The tail of
    # that block calibrates the conformal interval, while the preceding portion
    # estimates point-forecast quality and residual anomaly thresholds.
    test_n = max(28, min(90, int(len(frame) * .20)))
    split = len(frame) - test_n
    eval_frame, test = frame.iloc[:split], frame.iloc[split:]
    cal_n = max(14, int(len(test) * .50))
    eval_part, cal = test.iloc[:-cal_n], test.iloc[-cal_n:]
    model = HistGradientBoostingRegressor(max_iter=250, learning_rate=.05,
             max_leaf_nodes=15, l2_regularization=2.0, random_state=42)
    model.fit(eval_frame[FEATURES], eval_frame.target)
    pred_eval = model.predict(eval_part[FEATURES])
    pred_cal = model.predict(cal[FEATURES])
    cal_resid = np.abs(cal.target.to_numpy() - pred_cal)
    # Split conformal finite-sample quantile for a marginal (not conditional) interval.
    rank = min(len(cal_resid), int(np.ceil((len(cal_resid) + 1) * (1 - alpha))))
    radius = float(np.partition(cal_resid, rank - 1)[rank - 1])
    naive = seasonal_naive(series, eval_part.index)
    valid = np.isfinite(naive)
    metrics = {"HistGradientBoosting": metric_pack(eval_part.target.to_numpy(), pred_eval)}
    if valid.any():
        metrics["7-day seasonal naive"] = metric_pack(eval_part.target.to_numpy()[valid], naive[valid])
    # Flag unusually large one-step residuals using only the calibration residual distribution.
    med = float(np.median(cal_resid))
    mad = float(np.median(np.abs(cal_resid - med)))
    threshold = med + 4.5 * max(1.4826 * mad, 1e-8)
    anomaly_dates = list(cal.index[np.abs(cal.target.to_numpy() - pred_cal) > threshold])

    # Refit on all observed history. Recursively forecast each day, so future lags
    # use previous predictions, never the withheld actual values.
    full = make_features(series).assign(target=series).dropna()
    final_model = HistGradientBoostingRegressor(max_iter=250, learning_rate=.05,
                   max_leaf_nodes=15, l2_regularization=2.0, random_state=42)
    final_model.fit(full[FEATURES], full.target)
    hist = series.copy()
    rows = []
    start = hist.index.max() + pd.Timedelta(days=1)
    for d in pd.date_range(start, periods=horizon, freq="D"):
        extended = hist.reindex(hist.index.union(pd.DatetimeIndex([d])))
        one = make_features(extended).reindex([d])[FEATURES]
        if one.isna().any(axis=None):
            raise ValueError("Insufficient contiguous history to form forecast features.")
        point = max(0.0, float(final_model.predict(one)[0]))
        rows.append({"date": d, "forecast_mu": point,
                     "lower_mu": max(0.0, point - radius), "upper_mu": point + radius})
        hist.loc[d] = point
    return ForecastResult(pd.DataFrame(rows), metrics, anomaly_dates,
                          float(np.mean(np.abs(cal.target.to_numpy() - pred_cal) <= radius)),
                          "HistGradientBoosting + split conformal")


def backtest(series: pd.Series) -> dict[str, dict[str, float]]:
    """One chronological holdout evaluation; no randomized split."""
    return fit_forecast(series, horizon=1).metrics

