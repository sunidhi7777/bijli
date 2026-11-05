from __future__ import annotations
import pandas as pd

# Illustrative national observances and major festivals; dates are Gregorian and
# movable religious festivals need annual verification before operational use.
FIXED = [(1, 26), (8, 15), (10, 2), (12, 25)]
FESTIVAL_DATES = {
    2024: [(3, 25), (11, 1)], 2025: [(3, 14), (10, 20)],
    2026: [(3, 4), (11, 8)],
}

def calendar_features(dates: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(dates)
    fixed = {(d.month, d.day) for d in dates if (d.month, d.day) in FIXED}
    festivals = {(m, day) for d in dates for m, day in FESTIVAL_DATES.get(d.year, [])}
    return pd.DataFrame({"is_holiday": [int((d.month,d.day) in fixed or (d.month,d.day) in festivals) for d in dates],
                         "holiday_name_source": ["hand-built illustrative calendar" if ((d.month,d.day) in fixed or (d.month,d.day) in festivals) else "none" for d in dates]}, index=dates.index)
