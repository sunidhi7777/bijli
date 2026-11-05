from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from bijli.forecast import load_data,state_series,fit_forecast

def main():
    p=argparse.ArgumentParser(description="Train and evaluate Bijli state demand model")
    p.add_argument("--state",default="Maharashtra")
    p.add_argument("--horizon",type=int,default=7)
    p.add_argument("--data",default=str(ROOT/"data/raw/state_daily.csv"))
    args=p.parse_args()
    df=load_data(args.data)
    series=state_series(df,args.state).dropna()
    result=fit_forecast(series,args.horizon)
    print(f"State: {args.state}; last observed: {series.index[-1].date()}; model: {result.model_name}")
    print("Chronological holdout metrics:")
    for name,metrics in result.metrics.items(): print(name, {k:round(v,2) for k,v in metrics.items()})
    print(f"Calibration interval coverage: {result.calibration_coverage:.1%}")
    print(result.forecast.to_string(index=False))
    print("Residual review dates:", ", ".join(d.strftime("%Y-%m-%d") for d in result.anomaly_dates) or "none")
if __name__=="__main__":main()
