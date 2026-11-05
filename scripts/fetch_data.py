from __future__ import annotations
from pathlib import Path
import requests
import pandas as pd

SOURCE = "https://raw.githubusercontent.com/HalcyonVector/Grid-Sentinel/main/Dataset/study3_states.csv"

def main() -> None:
    out = Path("data/raw/state_daily.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(SOURCE, timeout=60)
    response.raise_for_status()
    df = pd.read_csv(pd.io.common.BytesIO(response.content))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "state", "energy_met_mu"])
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows through {df.date.max().date()} to {out}")

if __name__ == "__main__":
    main()
