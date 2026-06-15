"""Schema & Business Understanding der BTS-Dateien (CRISP-DM Phase 1+2)."""
from pathlib import Path
import pandas as pd
import json

PROJECT = Path("c:/Users/georgti/01_Steinbeis/ISE-01_Machine-Learning/02_Vorlesung/02_Mini-Case/ISE-02-Mini-Case")
RAW = PROJECT / "data" / "raw"
REPORT = PROJECT / "docs" / "data_exploration.md"

files = [
    "Detailed_Statistics_Departures.csv",
    "Detailed_Statistics_Cancellation.csv",
    "Detailed_Statistics_Diversion.csv",
]
for f in files:
    p = RAW / f
    if not p.exists():
        print(f"FEHLT: {f}")
        continue
    print("=" * 80)
    print(f"FILE: {f}   ({p.stat().st_size/1024:.1f} KB)")
    print("=" * 80)
    # Die BTS-CSVs haben 9 Metadaten-Zeilen oben, dann Spalten, dann Daten
    df = pd.read_csv(p, skiprows=9)
    print(f"Shape: {df.shape[0]} Zeilen x {df.shape[1]} Spalten")
    print(f"\nSpaltennamen ({df.shape[1]}):")
    for i, c in enumerate(df.columns, 1):
        print(f"  {i:2d}. {c!r}  | dtype={df[c].dtype}")
    print(f"\nDtypes:\n{df.dtypes.to_string()}")
    print(f"\nMissing values:\n{df.isna().sum().to_string()}")
    print(f"\nErste 3 Datenzeilen:")
    print(df.head(3).to_string())
    print(f"\nLetzte 3 Datenzeilen (zur Zeitraum-Verifikation):")
    print(df.tail(3).to_string())
    # Datum-Spannweite
    for date_col in [c for c in df.columns if "date" in c.lower() or "(mm" in c.lower()]:
        try:
            d = pd.to_datetime(df[date_col], errors="coerce")
            print(f"\nDatums-Spalte {date_col!r}: min={d.min()}  max={d.max()}  unique={d.nunique()}")
        except Exception:
            pass
    print()
