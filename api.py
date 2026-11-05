from fastapi import FastAPI, HTTPException, Query
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from bijli.forecast import load_data,state_series,fit_forecast
app=FastAPI(title="Bijli demand forecast API",version="1.0.0",description="Daily energy-met demonstration forecasts from Grid-India state PSP history.")
DATA=ROOT/"data/raw/state_daily.csv"
@app.get("/health")
def health(): return {"status":"ok"}
@app.get("/forecast")
def forecast(state:str=Query("Maharashtra"),horizon:int=Query(7,ge=1,le=14)):
    try:
        df=load_data(DATA); s=state_series(df,state).dropna(); r=fit_forecast(s,horizon)
        return {"state":state,"target":"energy_met_mu","unit":"MU/day","last_observed":str(s.index[-1].date()),"method":r.model_name,"metrics":r.metrics,"calibration_coverage":r.calibration_coverage,"forecast":r.forecast.assign(date=r.forecast.date.dt.strftime("%Y-%m-%d")).to_dict("records"),"anomaly_dates":[d.strftime("%Y-%m-%d") for d in r.anomaly_dates]}
    except (ValueError, KeyError) as e: raise HTTPException(status_code=422,detail=str(e))
