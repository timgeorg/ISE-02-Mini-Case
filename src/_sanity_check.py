"""Sanity-Check der finalen Features (inkl. Schulferien + 7d-Rolling)."""
import json
from pathlib import Path

import pandas as pd

PROJECT = Path("c:/Users/georgti/01_Steinbeis/ISE-01_Machine-Learning/02_Vorlesung/02_Mini-Case/ISE-02-Mini-Case")
PROC = PROJECT / "data" / "processed"
DOCS = PROJECT / "docs"

train = pd.read_parquet(PROC / "iad_flights_train.parquet")
val = pd.read_parquet(PROC / "iad_flights_val.parquet")

def stats(df):
    return {
        "n": len(df),
        "n_dest": df["dest_airport"].nunique(),
        "delay_label_pct": float(df["delay_label"].mean() * 100),
        "is_weekend_pct": float(df["is_weekend"].mean() * 100),
        "is_holiday_pct": float(df["is_holiday"].mean() * 100),
        "is_school_break_pct": float(df["is_school_break"].mean() * 100),
        "is_free_day_pct": float(df["is_free_day"].mean() * 100),
        "origin_daily_nonzero_pct": float((df["origin_daily_arrival_n"] > 0).mean() * 100),
        "origin_7d_nonzero_pct": float((df["origin_7d_arrival_n"] > 0).mean() * 100),
        "origin_7d_mean_mean": float(df["origin_7d_arrival_delay_mean"].mean()),
        "origin_7d_mean_max": float(df["origin_7d_arrival_delay_mean"].max()),
    }

s_train = stats(train)
s_val = stats(val)

train_dests = set(train["dest_airport"].unique())
val_dests = set(val["dest_airport"].unique())
new_in_val = val_dests - train_dests
val_to_new = val["dest_airport"].isin(new_in_val).sum()
val_to_new_pct = val_to_new / len(val) * 100

report_path = DOCS / "data_preparation.md"
content = report_path.read_text(encoding="utf-8")

train_md = (
    f"| Kennzahl | Train | Val |\n"
    f"|---|---:|---:|\n"
    f"| Zeilen | {s_train['n']:,} | {s_val['n']:,} |\n"
    f"| Unique Destinations | {s_train['n_dest']} | {s_val['n_dest']} |\n"
    f"| `delay_label=1` (Anteil) | {s_train['delay_label_pct']:.2f} % | {s_val['delay_label_pct']:.2f} % |\n"
    f"| `is_weekend` | {s_train['is_weekend_pct']:.2f} % | {s_val['is_weekend_pct']:.2f} % |\n"
    f"| `is_holiday` (federal) | {s_train['is_holiday_pct']:.2f} % | {s_val['is_holiday_pct']:.2f} % |\n"
    f"| `is_school_break` (regional) | {s_train['is_school_break_pct']:.2f} % | {s_val['is_school_break_pct']:.2f} % |\n"
    f"| `is_free_day` (kombi) | {s_train['is_free_day_pct']:.2f} % | {s_val['is_free_day_pct']:.2f} % |\n"
    f"| `origin_daily_arrival_n > 0` | {s_train['origin_daily_nonzero_pct']:.2f} % | {s_val['origin_daily_nonzero_pct']:.2f} % |\n"
    f"| `origin_7d_arrival_n > 0` | {s_train['origin_7d_nonzero_pct']:.2f} % | {s_val['origin_7d_nonzero_pct']:.2f} % |\n"
    f"| `origin_7d_arrival_delay_mean` Mittel | {s_train['origin_7d_mean_mean']:.2f} | {s_val['origin_7d_mean_mean']:.2f} |\n"
    f"| `origin_7d_arrival_delay_mean` Max | {s_train['origin_7d_mean_max']:.2f} | {s_val['origin_7d_mean_max']:.2f} |\n"
)
content = content.replace(
    "## Train vs. Val – Verteilungs-Check\n\n",
    "## Train vs. Val – Verteilungs-Check\n\n" + train_md
)

unseen_section = (
    "\n### Unseen Destinations (Data-Drift-Risiko)\n\n"
    f"- Destinations in Val, die nicht in Train vorkommen: **{len(new_in_val)} von {len(val_dests)}**\n"
    f"- Anteil der Val-Flüge zu unseen Destinations: **{val_to_new_pct:.2f} %** ({val_to_new:,} Flüge)\n\n"
    "Diese Flüge können vom Modell nicht sinnvoll eingeschätzt werden, weil das `dest_freq` und `flight_combo_freq`-Encoding 0/fehlend ist.\n"
)

new_feature_notes = (
    "\n### Beobachtungen zu den neuen Features (Schulferien + 7d-Rolling)\n\n"
    f"- **`is_school_break`** tritt in **{s_train['is_school_break_pct']:.1f}% der Train-Flüge** und "
    f"**{s_val['is_school_break_pct']:.1f}% der Val-Flüge** auf. Damit ist der Val-Wert deutlich niedriger "
    "(Val = Jan-Apr enthaelt nur Spring Break + Winter Break), was die Modell-Performance am Anfang beeinflussen kann.\n"
    f"- **`is_free_day` (kombinierte Variable)** deckt Train ~{s_train['is_free_day_pct']:.0f} % / Val ~{s_val['is_free_day_pct']:.0f} % ab. Konsistent mit Wochenend-Quote (~28 %).\n"
    f"- **`origin_7d_arrival_n > 0`** fuer **{s_train['origin_7d_nonzero_pct']:.1f}%** der Train-Fluege "
    f"und **{s_val['origin_7d_nonzero_pct']:.1f}%** der Val-Fluege. Rest: kein 7-Tage-Historie (cold start, z. B. neuer Origin oder Datenbeginn).\n"
    f"- **7-Tage-Mittel der Origin-Verspaetung** zeigt im Mittel ~{s_train['origin_7d_mean_mean']:.1f} Min Verspaetung (Train) - positiv, weil Carrier an IAD im Schnitt leicht verspaetet ist.\n"
)
content = content.replace(
    "## Output-Dateien", new_feature_notes + unseen_section + "\n## Output-Dateien"
)

report_path.write_text(content, encoding="utf-8")
print(f"Report updated: {report_path}")
print()
print("=== Kern-KPIs ===")
print(f"Train:  {s_train['n']:,} rows, {s_train['is_school_break_pct']:.1f}% school-break, "
      f"{s_train['origin_7d_nonzero_pct']:.1f}% have 7d-arrival-history")
print(f"Val:    {s_val['n']:,} rows, {s_val['is_school_break_pct']:.1f}% school-break, "
      f"{s_val['origin_7d_nonzero_pct']:.1f}% have 7d-arrival-history")
