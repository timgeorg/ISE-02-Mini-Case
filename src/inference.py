"""
Inference-Skript: Nimmt einen Flug (oder eine Liste) und liefert eine Verspätungs-
wahrscheinlichkeit sowie ein Label.

Designprinzipien
----------------
1.  Repliziert exakt das Feature-Engineering der Trainingspipeline
    (`src/02_data_preparation.py`), damit Train- und Serve-Verteilung gleich sind.
2.  Lädt das trainierte Modell + Feature-Metadaten.
3.  Aggregat-Features (Origin-Frequenzen, 1d/7d-Arrival-Aggregate, Cancellation-
    Counts) werden aus dem Trainingsdatensatz vorberechnet und beim Lookup
    verwendet. So vermeiden wir Leakage (wir nutzen keine Labels aus dem
    Inferenzzeitpunkt).
4.  Wetter ist aktuell KEIN Modellfeature, wird aber als Input akzeptiert und
    protokolliert; so ist das Skript bereit, sobald Wetter ins Training kommt.

Nutzung
------
CLI:
    python -m src.inference --flight-json '{"carrier_code":"UA","date":"2026-06-01","flight_number":"123","dest_airport":"LHR","sched_dep_time":"17:30","sched_elapsed_min":420}'
    python -m src.inference --flight-csv pfad/zu/fluege.csv --output-csv predictions.csv
    python -m src.inference --demo

Programmatisch:
    from src.inference import FlightPredictor
    p = FlightPredictor()
    res = p.predict_one({...})
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Pfade & Logging
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "feature_metadata.json"
DEFAULT_TRAIN_PARQUET = PROJECT_ROOT / "data" / "processed" / "iad_flights_train.parquet"

# Schulferien-/Feiertags-Flags kommen aus der Prep-Pipeline. Wir replizieren die
# Logik minimal (gleiche Defaults wie im Training), damit das Inferenz-Skript
# ohne Prep-Import eigenständig läuft.
US_FEDERAL_HOLIDAYS_2025_2026: set[date] = {
    date(2025, 1, 1),   # New Year
    date(2025, 1, 20),  # MLK
    date(2025, 2, 17),  # Presidents Day
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 10, 13), # Columbus Day
    date(2025, 11, 11), # Veterans Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas
    date(2026, 1, 1),
    date(2026, 1, 19),
    date(2026, 2, 16),
    date(2026, 5, 25),
    date(2026, 6, 19),
    date(2026, 7, 3),   # observed
    date(2026, 9, 7),
    date(2026, 10, 12),
    date(2026, 11, 11),
    date(2026, 11, 26),
    date(2026, 12, 25),
}

# Regionale Schulferien – DC + Fairfax + Montgomery (3 Distrikte wie in der
# Trainingspipeline). Vereinfachte, deterministische Liste.
SCHOOL_BREAK_RANGES: list[tuple[date, date]] = [
    # 2025
    (date(2025, 3, 17), date(2025, 3, 21)),  # Spring break
    (date(2025, 4, 14), date(2025, 4, 18)),
    (date(2025, 6, 16), date(2025, 8, 25)),  # Summer break
    (date(2025, 11, 26), date(2025, 11, 28)),  # Thanksgiving
    (date(2025, 12, 22), date(2026, 1, 2)),   # Winter break
    # 2026
    (date(2026, 3, 16), date(2026, 3, 20)),
    (date(2026, 4, 6), date(2026, 4, 10)),
    (date(2026, 6, 22), date(2026, 8, 24)),
    (date(2026, 11, 25), date(2026, 11, 27)),
    (date(2026, 12, 21), date(2027, 1, 1)),
]


def _is_holiday(d: date) -> bool:
    return d in US_FEDERAL_HOLIDAYS_2025_2026


def _is_school_break(d: date) -> bool:
    return any(start <= d <= end for start, end in SCHOOL_BREAK_RANGES)


# ----------------------------------------------------------------------------
# Feature-Engineering (spiegelt `02_data_preparation.py`)
# ----------------------------------------------------------------------------
TIME_OF_DAY_BINS = [
    (0, 6, 0),    # night
    (6, 12, 1),   # morning
    (12, 18, 2),  # afternoon
    (18, 24, 3),  # evening
]


def _time_of_day(hour: int) -> int:
    """Numerischer Code (muss identisch zur Prep-Pipeline sein)."""
    for start, end, code in TIME_OF_DAY_BINS:
        if start <= hour < end:
            return code
    return 0


def build_features(
    flight: dict[str, Any],
    lookup: "LookupTables",
) -> dict[str, float]:
    """Berechnet den Feature-Vektor für EINEN Flug.

    Erwartete Input-Felder (alle optional, sinnvolle Defaults gesetzt):
        carrier_code         (str, default 'UA')
        date                 (str 'YYYY-MM-DD' oder date)
        flight_number        (str|int, optional)
        dest_airport         (str, optional, default 'UNK')
        sched_dep_time       (str 'HH:MM' oder int HHMM, optional, default '12:00')
        sched_elapsed_min    (float, optional, default 120.0)
        weather              (dict, optional; aktuell nur Logging, kein Feature)

    Hinweis: Post-hoc-Informationen (departure_delay_min, delay_label,
    tatsächliche Zeiten) werden ignoriert bzw. nicht akzeptiert.
    """
    # --- Pflicht-/Optionalfelder parsen -------------------------------------
    carrier = str(flight.get("carrier_code", "UA")).upper()
    d = _coerce_date(flight.get("date"))
    if d is None:
        raise ValueError("Feld 'date' ist erforderlich (YYYY-MM-DD).")

    hour, minute = _coerce_time(flight.get("sched_dep_time", "12:00"))
    dest = str(flight.get("dest_airport", "UNK")).upper()
    flight_number = str(flight.get("flight_number", "0"))
    sched_elapsed = float(flight.get("sched_elapsed_min", 120.0))

    # --- Datum / Zeit-Features ---------------------------------------------
    dow = d.weekday()  # 0=Mon
    month = d.month
    day = d.day
    weekofyear = int(d.strftime("%V"))
    quarter = (month - 1) // 3 + 1
    is_weekend = int(dow >= 5)
    is_holiday = int(_is_holiday(d))
    is_school_break = int(_is_school_break(d))
    is_free = int(is_weekend or is_holiday or is_school_break)

    sched_hour = hour
    sched_minute = minute
    min_of_day = sched_hour * 60 + sched_minute
    tod = _time_of_day(sched_hour)

    # --- Zyklische Kodierungen ---------------------------------------------
    dow_sin = math.sin(2 * math.pi * dow / 7)
    dow_cos = math.cos(2 * math.pi * dow / 7)
    month_sin = math.sin(2 * math.pi * (month - 1) / 12)
    month_cos = math.cos(2 * math.pi * (month - 1) / 12)
    tod_sin = math.sin(2 * math.pi * min_of_day / 1440)
    tod_cos = math.cos(2 * math.pi * min_of_day / 1440)

    # --- Aggregat-Lookups (aus Trainingsdaten) ------------------------------
    flight_combo = f"{carrier}{flight_number}"
    flight_combo_freq = float(lookup.flight_combo_freq.get(flight_combo, 0))
    dest_freq = float(lookup.dest_freq.get(dest, 0))

    # 1d-Origin-Arrival-Aggregate: Lookup für Vortag (sonst 0)
    prior = d - timedelta(days=1)
    arrival_stats_1d = lookup.origin_arrival_1d.get(prior, {"mean": 0.0, "n": 0.0})
    arrival_stats_7d = lookup.origin_arrival_7d.get(prior, {"mean": 0.0, "n": 0.0})

    cancellations_on_day = float(lookup.cancellations_per_day.get(d, 0.0))

    return {
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "dayofweek": dow,
        "day": day,
        "month": month,
        "weekofyear": weekofyear,
        "quarter": quarter,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "is_school_break": is_school_break,
        "is_free_day": is_free,
        "sched_dep_hour": sched_hour,
        "sched_dep_minute": sched_minute,
        "sched_dep_min_of_day": min_of_day,
        "time_of_day": tod,  # kategorisch -> wird in der Prep via OHE gemacht
        "tod_sin": tod_sin,
        "tod_cos": tod_cos,
        "sched_elapsed_min": sched_elapsed,
        "flight_combo_freq": flight_combo_freq,
        "dest_freq": dest_freq,
        "origin_daily_arrival_delay_mean": float(arrival_stats_1d["mean"]),
        "origin_daily_arrival_n": float(arrival_stats_1d["n"]),
        "origin_7d_arrival_delay_mean": float(arrival_stats_7d["mean"]),
        "origin_7d_arrival_n": float(arrival_stats_7d["n"]),
        "cancellations_on_day": cancellations_on_day,
    }


# ----------------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------------
def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    raise ValueError(f"Unbekanntes Datumsformat: {value!r}")


def _coerce_time(value: Any) -> tuple[int, int]:
    """Akzeptiert 'HH:MM' oder int HHMM (BTS-Format)."""
    if value is None:
        return 12, 0
    if isinstance(value, str):
        s = value.strip()
        if ":" in s:
            h, m = s.split(":")[:2]
            return int(h), int(m)
        if s.isdigit() and len(s) in (3, 4):
            s = s.zfill(4)
            return int(s[:2]), int(s[2:])
        raise ValueError(f"Unbekanntes Zeitformat: {value!r}")
    if isinstance(value, (int, float)):
        n = int(value)
        return n // 100, n % 100
    raise ValueError(f"Unbekanntes Zeitformat: {value!r}")


# ----------------------------------------------------------------------------
# Lookup-Tabellen aus Trainingsdaten (verhindert Leakage + erspart Re-Training
# der Aggregat-Werte)
# ----------------------------------------------------------------------------
@dataclass
class LookupTables:
    flight_combo_freq: dict[str, float]
    dest_freq: dict[str, float]
    origin_arrival_1d: dict[date, dict[str, float]]
    origin_arrival_7d: dict[date, dict[str, float]]
    cancellations_per_day: dict[date, float]


def build_lookups(train_parquet: Path) -> LookupTables:
    """Berechnet alle Lookup-Tabellen aus den Trainingsdaten.

    `origin_arrival_*` ist hier der Mittelwert über alle Carrier am Origin IAD –
    repliziert die Definition der Prep-Pipeline. Für die Inferenz wird für
    den Flug am Tag D der Wert vom Vortag D-1 nachgeschlagen.
    """
    df = pd.read_parquet(train_parquet)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    flight_combo_freq = (
        df.groupby(["carrier_code", "flight_number"]).size().to_dict()
        if {"carrier_code", "flight_number"}.issubset(df.columns)
        else {}
    )
    # Schlüssel vereinheitlichen -> "UA123"
    flight_combo_freq = {
        f"{c}{fn}": float(v) for (c, fn), v in flight_combo_freq.items()
    }

    dest_freq = (
        df.groupby("dest_airport").size().to_dict() if "dest_airport" in df.columns else {}
    )
    dest_freq = {str(k).upper(): float(v) for k, v in dest_freq.items()}

    # 1d-Aggregat: pro Tag Mittelwert & Anzahl der Arrival-Delays
    if {"date", "arrival_delay_min"}.issubset(df.columns):
        g1 = df.groupby("date")["arrival_delay_min"].agg(["mean", "count"])
        arrival_1d = {idx: {"mean": float(r["mean"]), "n": float(r["count"])} for idx, r in g1.iterrows()}
    else:
        arrival_1d = {}

    # 7d-Aggregat: rollender Mittelwert pro Tag
    if {"date", "arrival_delay_min"}.issubset(df.columns):
        s = (
            df.groupby("date")["arrival_delay_min"]
            .mean()
            .sort_index()
            .rolling(window=7, min_periods=1)
            .mean()
        )
        cnt = df.groupby("date")["arrival_delay_min"].count().rolling(window=7, min_periods=1).sum()
        arrival_7d = {
            idx: {"mean": float(s.loc[idx]), "n": float(cnt.loc[idx])} for idx in s.index
        }
    else:
        arrival_7d = {}

    # Cancellation-Counts pro Tag
    if "is_cancelled" in df.columns:
        cc = df.groupby("date")["is_cancelled"].sum()
        cancellations = {idx: float(v) for idx, v in cc.items()}
    else:
        cancellations = {}

    return LookupTables(
        flight_combo_freq=flight_combo_freq,
        dest_freq={k: v for k, v in dest_freq.items()},
        origin_arrival_1d=arrival_1d,
        origin_arrival_7d=arrival_7d,
        cancellations_per_day=cancellations,
    )


# ----------------------------------------------------------------------------
# Predictor
# ----------------------------------------------------------------------------
class FlightPredictor:
    """Kapselt Modell + Lookups."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
        train_parquet: Path = DEFAULT_TRAIN_PARQUET,
        threshold: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        loaded = joblib.load(self.model_path)
        # Wrapper-Dict oder nacktes Modell akzeptieren
        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded["model"]
            self.scaler = loaded.get("scaler")
            self.wrapped_metrics = loaded.get("metrics")
        else:
            self.model = loaded
            self.scaler = None
            self.wrapped_metrics = None
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.feature_columns: list[str] = self.meta["feature_columns"]
        self.metadata_columns: list[str] = self.meta.get("metadata_columns", [])
        self.delay_threshold_min: int = int(self.meta.get("delay_threshold_min", 15))
        self.threshold = threshold
        self.lookups = build_lookups(train_parquet)
        self.logger = logging.getLogger("inference")

    # ---- intern ---------------------------------------------------------
    def _vectorize(self, flight: dict[str, Any]) -> pd.DataFrame:
        feats = build_features(flight, self.lookups)
        # Reihenfolge exakt wie im Training
        row = {col: feats.get(col, 0.0) for col in self.feature_columns}
        df = pd.DataFrame([row], columns=self.feature_columns)
        # XGBoost (und andere Booster) wollen numerische dtypes – nullable
        # Int64 aus dem Prep-Schritt auf float64 bringen, damit auch das
        # frisch erzeugte Inferenz-Dataframe sauber durchgeht.
        for col in df.columns:
            if df[col].dtype.name in ("Int64", "boolean"):
                df[col] = df[col].astype("float64")
        return df

    def _prepare_for_model(self, X: pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Scaler anwenden, sofern das Modell einen gespeichert hat (LogReg)."""
        if self.scaler is not None:
            return self.scaler.transform(X.values.astype("float64"))
        return X

    # ---- public ---------------------------------------------------------
    def predict_one(self, flight: dict[str, Any]) -> dict[str, Any]:
        """Gibt Wahrscheinlichkeit, Label, Eingabeflug, Wetter-Hinweise zurück."""
        X = self._vectorize(flight)
        X_in = self._prepare_for_model(X)
        prob = float(self.model.predict_proba(X_in)[:, 1][0])
        label = int(prob >= self.threshold)
        out = {
            "input": _sanitize_input(flight),
            "prob_delay": round(prob, 4),
            "threshold": self.threshold,
            "label": label,
            "model": type(self.model).__name__,
            "model_path": str(self.model_path),
            "feature_version": {
                "n_features": self.meta.get("n_features"),
                "delay_threshold_min": self.delay_threshold_min,
            },
            "weather_used": _summarize_weather(flight.get("weather")),
        }
        return out

    def predict_batch(self, flights: list[dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for f in flights:
            rows.append(self.predict_one(f))
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def _sanitize_input(flight: dict[str, Any]) -> dict[str, Any]:
    """Entfernt verbotene Felder (post-hoc) und gibt nur Eingabe zurück."""
    forbidden = {
        "departure_delay_min",
        "delay_label",
        "is_cancelled",
        "cancellation_code",
        "actual_elapsed_time",
        "arrival_delay_min",
        "wheels_off",
        "wheels_on",
        "taxi_out",
        "taxi_in",
    }
    return {k: v for k, v in flight.items() if k not in forbidden}


def _summarize_weather(weather: Any) -> dict[str, Any] | None:
    """Aktuell nur Echo – vorbereitet für künftige Wetterfeatures."""
    if weather is None:
        return None
    if not isinstance(weather, dict):
        return {"raw": str(weather)}
    return {
        "source": weather.get("source"),
        "hour_utc": weather.get("hour_utc"),
        "temp_c": weather.get("temp_c"),
        "wind_kts": weather.get("wind_kts"),
        "vis_m": weather.get("vis_m"),
        "precip_mm": weather.get("precip_mm"),
        "pressure_hpa": weather.get("pressure_hpa"),
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _cli(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Flight-Delay Inference")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    p.add_argument("--train-parquet", type=Path, default=DEFAULT_TRAIN_PARQUET)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--flight-json", type=str, help="JSON-String mit Flugdaten")
    p.add_argument("--flight-csv", type=Path, help="CSV-Datei mit Flugdaten")
    p.add_argument("--output-csv", type=Path, help="Optional: predictions als CSV")
    p.add_argument("--demo", action="store_true", help="Lädt einen Demo-Flug und prediceted")
    args = p.parse_args(argv)

    predictor = FlightPredictor(
        model_path=args.model,
        metadata_path=args.metadata,
        train_parquet=args.train_parquet,
        threshold=args.threshold,
    )

    if args.demo:
        demo = {
            "carrier_code": "UA",
            "date": "2026-06-15",
            "flight_number": "944",
            "dest_airport": "LHR",
            "sched_dep_time": "18:30",
            "sched_elapsed_min": 420,
            "weather": {
                "source": "demo",
                "hour_utc": "2026-06-15T22:30Z",
                "temp_c": 18.0,
                "wind_kts": 7.0,
                "vis_m": 10000,
                "precip_mm": 0.0,
                "pressure_hpa": 1015,
            },
        }
        print(json.dumps(predictor.predict_one(demo), indent=2, ensure_ascii=False))
        return 0

    if args.flight_json:
        try:
            flight = json.loads(args.flight_json)
        except json.JSONDecodeError:
            # Fallback: Datei?
            flight = json.loads(Path(args.flight_json).read_text(encoding="utf-8"))
        print(json.dumps(predictor.predict_one(flight), indent=2, ensure_ascii=False))
        return 0

    if args.flight_csv:
        df = pd.read_csv(args.flight_csv)
        flights = df.to_dict(orient="records")
        out = predictor.predict_batch(flights)
        if args.output_csv:
            out.to_csv(args.output_csv, index=False)
            print(f"{len(out)} Predictions geschrieben: {args.output_csv}")
        else:
            print(out.to_string(index=False))
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(_cli())
