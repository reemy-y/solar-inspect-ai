"""
generate_sample_data.py
───────────────────────
Run this ONCE locally to create data/solar_data.csv
The script generates a realistic synthetic solar plant dataset
that the app can use for dataset browsing, charts, and predictions.

Usage:
    python generate_sample_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ── Reproducible random seed
np.random.seed(42)

# ── Config
N_DAYS      = 90          # 3 months of data
PANELS      = ["P-001", "P-002", "P-003", "P-004", "P-005"]
DEFECT_TYPES = ["Clean", "Dusty", "Bird-drop", "Electrical-damage",
                "Physical-damage", "Snow-covered"]

rows = []
start_date = datetime(2024, 1, 1, 6, 0)   # start at 6am

for day in range(N_DAYS):
    for hour in range(6, 20):              # daylight hours only
        dt = start_date + timedelta(days=day, hours=hour - 6)

        # Simulate irradiation — peaks at noon
        hour_norm   = (hour - 6) / 14.0   # 0..1 over the day
        irradiation = max(0, np.sin(hour_norm * np.pi) * np.random.uniform(0.6, 1.1))

        ambient_temp = 20 + 15 * np.sin(hour_norm * np.pi) + np.random.normal(0, 2)
        module_temp  = ambient_temp + irradiation * 25 + np.random.normal(0, 3)

        for panel_id in PANELS:
            # Each panel has a baseline efficiency degradation
            panel_idx   = PANELS.index(panel_id)
            efficiency  = 0.95 - panel_idx * 0.02 + np.random.normal(0, 0.01)

            dc_power    = irradiation * 3500 * efficiency + np.random.normal(0, 50)
            dc_power    = max(0, dc_power)
            ac_power    = dc_power * 0.96 * efficiency + np.random.normal(0, 30)
            ac_power    = max(0, ac_power)

            # Assign defect — most panels are clean
            defect_probs = [0.60, 0.15, 0.10, 0.05, 0.05, 0.05]
            defect = np.random.choice(DEFECT_TYPES, p=defect_probs)

            # Defect degrades power
            defect_factor = {
                "Clean": 1.0, "Dusty": 0.85, "Bird-drop": 0.92,
                "Electrical-damage": 0.50, "Physical-damage": 0.60,
                "Snow-covered": 0.10,
            }[defect]
            ac_power *= defect_factor
            dc_power *= defect_factor

            rows.append({
                "timestamp":       dt.strftime("%Y-%m-%d %H:%M"),
                "date":            dt.strftime("%Y-%m-%d"),
                "hour":            hour,
                "panel_id":        panel_id,
                "irradiation":     round(irradiation, 4),
                "ambient_temp_c":  round(ambient_temp, 2),
                "module_temp_c":   round(module_temp, 2),
                "dc_power_kw":     round(dc_power / 1000, 3),
                "ac_power_kw":     round(ac_power / 1000, 3),
                "defect_type":     defect,
                "efficiency_pct":  round(efficiency * 100, 2),
            })

df = pd.DataFrame(rows)

# ── Save
os.makedirs("data", exist_ok=True)
out = "data/solar_data.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df):,} rows to {out}")
print(df.head())
print(df.dtypes)
