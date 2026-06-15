"""Quick Sanity-Check: Verteilungen Train vs. Val."""
import json
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT = Path("c:/Users/georgti/01_Steinbeis/ISE-01_Machine-Learning/02_Vorlesung/02_Mini-Case/ISE-02-Mini-Case")
PROC = PROJECT / "data" / "processed"
DOCS = PROJECT / "docs"

train = pd.read_parquet(PROC / "iad_flights_train.parquet")
val = pd.read_parquet(PROC / "iad_flights_val.parquet")

def stats(df, name):
    return {
        "name": name,
        "n": len(df),
        "n_dest": df["dest_airport"].nunique(),
        "n_dest_only_in_train": None,
        "n_dest_only_in_val": None,
        "mean_delay": float(df["departure_delay_min"].mean()),
        "median_delay": float(df["departure_delay_min"].median()),
        "delay_label_pct": float(df["delay_label"].mean() * 100),
        "mean_sched_dep_hour": float(df["sched_dep_hour"].mean()),
        "is_weekend_pct": float(df["is_weekend"].mean() * 100),
        "is_holiday_pct": float(df["is_holiday"].mean() * 100),
        "mean_sched_elapsed": float(df["sched_elapsed_min"].mean()),
        "mean_origin_daily_arrival_delay": float(df["origin_daily_arrival_delay_mean"].mean()),
    }

s_train = stats(train, "Train")
s_val = stats(val, "Val")

# Neue Destinations in Val (nicht in Train)
train_dests = set(train["dest_airport"].unique())
val_dests = set(val["dest_airport"].unique())
new_in_val = val_dests - train_dests
n_new_in_val_pct = len(new_in_val) / len(val) * 100

# Anteil Val-Flüge zu unseen Destinations
val_to_new = val["dest_airport"].isin(new_in_val).sum()
val_to_new_pct = val_to_new / len(val) * 100

# Update the report
report_path = DOCS / "data_preparation.md"
content = report_path.read_text(encoding="utf-8")

train_md = (
    f"| Kennzahl | Train | Val |\n"
    f"|---|---:|---:|\n"
    f"| Zeilen | {s_train['n']:,} | {s_val['n']:,} |\n"
    f"| Unique Destinations | {s_train['n_dest']} | {s_val['n_dest']} |\n"
    f"| Mittlere Verspätung (Min) | {s_train['mean_delay']:.2f} | {s_val['mean_delay']:.2f} |\n"
    f"| Median Verspätung | {s_train['median_delay']:.2f} | {s_val['median_delay']:.2f} |\n"
    f"| `delay_label=1` (Anteil) | {s_train['delay_label_pct']:.2f} % | {s_val['delay_label_pct']:.2f} % |\n"
    f"| Mittlere Abflug-Stunde | {s_train['mean_sched_dep_hour']:.2f} | {s_val['mean_sched_dep_hour']:.2f} |\n"
    f"| Wochenend-Anteil | {s_train['is_weekend_pct']:.2f} % | {s_val['is_weekend_pct']:.2f} % |\n"
    f"| Feiertags-Anteil | {s_train['is_holiday_pct']:.2f} % | {s_val['is_holiday_pct']:.2f} % |\n"
    f"| Mittlere Flugdauer (Min) | {s_train['mean_sched_elapsed']:.1f} | {s_val['mean_sched_elapsed']:.1f} |\n"
    f"| Mittlere origin_daily_arrival_delay | {s_train['mean_origin_daily_arrival_delay']:.2f} | {s_val['mean_origin_daily_arrival_delay']:.2f} |\n"
)
content = content.replace(
    "## Train vs. Val – Verteilungs-Check\n\nTBD: wird beim Lauf ergänzt (Distinct counts, etc.)",
    "## Train vs. Val – Verteilungs-Check\n\n" + train_md
)

# Add unseen destination info
unseen_section = (
    "\n### Unseen Destinations (Data-Drift-Risiko)\n\n"
    f"- Destinations in Val, die nicht in Train vorkommen: **{len(new_in_val)} von {len(val_dests)}**\n"
    f"- Anteil der Val-Flüge zu unseen Destinations: **{val_to_new_pct:.2f} %** ({val_to_new:,} Flüge)\n\n"
    "Diese Flüge können vom Modell nicht sinnvoll eingeschätzt werden, weil das `dest_freq` und `flight_combo_freq`-Encoding 0/fehlend ist.\n"
)
content = content.replace(
    "## Output-Dateien", unseen_section + "\n## Output-Dateien"
)

report_path.write_text(content, encoding="utf-8")
print(f"Report updated: {report_path}")
print(f"Unseen Destinations in Val: {len(new_in_val)} ({val_to_new_pct:.2f}% of val flights)")
print(f"  {sorted(new_in_val)[:10]}{'...' if len(new_in_val) > 10 else ''}")
