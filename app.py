from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import plotly.express as px
import streamlit as st
from bijli.forecast import load_data, state_series, fit_forecast, make_features

st.set_page_config(page_title="Bijli | India demand forecast", page_icon="⚡", layout="wide")
st.title("⚡ Bijli")
st.caption("Probabilistic electricity energy-met forecasting and residual anomaly review")
st.info("The target is daily **energy met (MU/day)** from Grid-India/NLDC state PSP reports. This is a historical proxy for served demand; reported energy met is not the same as unconstrained demand when supply is short. Forecasts are a research demo, not grid dispatch advice.")

@st.cache_data
def data():
    return load_data(ROOT / "data/raw/state_daily.csv")

df = data()
st.sidebar.header("Forecast settings")
state = st.sidebar.selectbox("State / reported entity", sorted(df.state.unique()), index=sorted(df.state.unique()).index("Maharashtra") if "Maharashtra" in df.state.unique() else 0)
horizon = st.sidebar.slider("Forecast horizon (days)", 1, 14, 7)
s = state_series(df, state).dropna()
try:
    result = fit_forecast(s, horizon=horizon)
except ValueError as e:
    st.error(str(e)); st.stop()

c1,c2,c3 = st.columns(3)
c1.metric("Latest observed", f"{s.iloc[-1]:,.1f} MU", s.index[-1].strftime("%d %b %Y"))
c2.metric("Next day median", f"{result.forecast.forecast_mu.iloc[0]:,.1f} MU")
c3.metric("Conformal calibration coverage", f"{result.calibration_coverage:.0%}", "target 90% on calibration block")

hist = s.tail(180).rename("Observed MU").reset_index(names="date")
fc = result.forecast.copy()
fig = px.line(hist, x="date", y="Observed MU", title=f"{state}: recent daily energy met and forecast")
fig.add_scatter(x=fc.date, y=fc.forecast_mu, mode="lines+markers", name="Forecast")
fig.add_scatter(x=list(fc.date)+list(fc.date[::-1]), y=list(fc.upper_mu)+list(fc.lower_mu[::-1]),
                fill="toself", fillcolor="rgba(99,110,250,0.18)", line={"color":"rgba(255,255,255,0)"},
                name="90% conformal interval")
st.plotly_chart(fig, use_container_width=True)

left,right=st.columns([1,1])
with left:
    st.subheader("Chronological holdout comparison")
    st.dataframe(pd.DataFrame(result.metrics).T.style.format("{:.2f}"), use_container_width=True)
    st.caption("Evaluation is a single time-ordered holdout, not repeated rolling-origin CV. Treat the scores as a transparent baseline, not a validated production claim.")
with right:
    st.subheader("Forecast table")
    st.dataframe(fc.assign(date=fc.date.dt.strftime("%Y-%m-%d")).style.format({"forecast_mu":"{:,.1f}","lower_mu":"{:,.1f}","upper_mu":"{:,.1f}"}), use_container_width=True)

st.subheader("Residual anomaly review")
if result.anomaly_dates:
    st.warning(f"{len(result.anomaly_dates)} unusually large residual(s) in the recent calibration window. These are review flags, not confirmed grid events.")
    st.dataframe(pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in result.anomaly_dates]}), use_container_width=True)
else:
    st.success("No calibration residual crossed the robust threshold in this window.")
st.subheader("Recent history")
st.line_chart(s.tail(365))
with st.expander("Data and method notes"):
    st.markdown("""
    - Source: daily state power supply position (`energy_met_mu`) parsed from Grid-India/NLDC reports. Dataset snapshot is bundled for repeatability; `python scripts/fetch_data.py` refreshes it.
    - Features use calendar seasonality and lagged target/rolling statistics only. Future target values are recursively predicted.
    - Point model: scikit-learn histogram gradient boosting. Interval: absolute-residual split conformal calibration, marginal coverage target 90% under exchangeability. Real demand series drift and seasonality can break that assumption.
    - Anomalies are large forecast residuals using a robust median/MAD threshold; an alert needs operator/context review.
    - Historical weather is available as optional EDA through Open-Meteo; it is intentionally not used as a forecast feature because archive weather is not known at issue time.
    """)
st.caption("Bijli • research portfolio project • data through " + df.date.max().strftime("%d %b %Y"))
