"""
CRISP-DM Phase 3: Data Preparation
====================================

Pipeline: laden -> bereinigen -> features ableiten -> target -> split -> parquet.

Input:  data/raw/Detailed_Statistics_*.csv
Output: data/processed/iad_flights_train.parquet
        data/processed/iad_flights_val.parquet
        data/processed/feature_metadata.json
        docs/data_preparation.md

Getroffene Entscheidungen (siehe docs/data_understanding.md):
  - Verfruehte Fluge (< 0 Min) = Klasse 0
  - Cancelled/Diverted = Drop (kein sinnvolles Target)
  - 2025 = Train, 2026 (Jan-Apr) = Validation (temporal)
  - Threshold = 15 Min
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
DOCS = PROJECT / "docs"
LOGS = PROJECT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

# Spaltendefinitionen (siehe 01_business_data_understanding.py)
DEPARTURE_COLS = [
    "carrier_code", "date", "flight_number", "tail_number", "dest_airport",
    "sched_dep_time", "actual_dep_time",
    "sched_elapsed_min", "actual_elapsed_min",
    "departure_delay_min", "wheels_off_time", "taxi_out_min",
    "delay_carrier_min", "delay_weather_min", "delay_nas_min",
    "delay_security_min", "delay_late_aircraft_min",
]
ARRIVAL_COLS = [
    "carrier_code", "date", "flight_number", "tail_number", "origin_airport",
    "sched_arr_time", "actual_arr_time",
    "sched_elapsed_min", "actual_elapsed_min",
    "arrival_delay_min", "wheels_on_time", "taxi_in_min",
    "delay_carrier_min", "delay_weather_min", "delay_nas_min",
    "delay_security_min", "delay_late_aircraft_min",
]
CANCELLATION_COLS = ["carrier_code", "date", "flight_number", "tail_number", "dest_airport"]
DIVERSION_COLS = ["carrier_code", "date", "flight_number", "tail_number", "dest_airport"]


def _find_raw(name: str) -> Path:
    """Sucht die Roh-CSV im data/raw/-Verzeichnis, mit oder ohne Jahresprefix.

    Erlaubte Muster:
        Detailed_Statistics_<name>.csv   (alt)
        2024_Detailed_Statistics_<name>.csv
        2025_Detailed_Statistics_<name>.csv
    Wir nehmen die erste gefundene Datei. Bei mehreren Jahrgängen
    werden sie unten konkateniert.
    """
    candidates = sorted(RAW.glob(f"*Detailed_Statistics_{name}.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Keine Roh-Datei für '{name}' in {RAW} gefunden. "
            f"Erwartet: *Detailed_Statistics_{name}.csv"
        )
    if len(candidates) > 1:
        log.info("Mehrere Roh-Dateien für %s: %s – werden konkateniert.",
                 name, [c.name for c in candidates])
    return candidates

# Schwellwert & Split (wetter-bewusst)
#
# Hintergrund: NCEI publiziert das laufende Jahr erst mit ca. 6 Wochen Verzug.
# Unsere Wetter-Datei endet am 2025-08-27. Damit das Val-Set ebenfalls Wetter
# hat, definieren wir einen "wetter-bewussten" Split:
#   Train: < 2025-07-01  (~18 Monate Train, alle mit Wetter)
#   Val:   2025-07-01 .. 2025-08-27  (~2 Monate Val, alle mit Wetter)
# Damit ist Wetter ein verlässliches Feature in beiden Sets.
DELAY_THRESHOLD_MIN = 15
SPLIT_DATE = pd.Timestamp("2025-07-01")
WETTER_MAX_DATE = pd.Timestamp("2025-08-26")  # 2025-08-27 ist unvollständig

# Wetter (optional – kann per CLI abgeschaltet werden)
USE_WEATHER = True
WEATHER_PATH = PROJECT / "data" / "external" / "weather" / "iad_isd_hourly.csv"

# US-Bundesfeiertage 2025-2026 (gemaess federal Reserve / OPM)
US_FEDERAL_HOLIDAYS = pd.to_datetime([
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # MLK Day
    "2025-02-17",  # Presidents' Day
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-10-13",  # Columbus Day
    "2025-11-11",  # Veterans Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Presidents' Day
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observed)
])

# Regionale Schulferien (3 grosse Bezirke im IAD-Einzugsgebiet).
# Wir kodieren nur die datierten "Breaks" (Tage, an denen Schulen geschlossen sind),
# weil sie die staerksten Passagier-Spitzen am IAD verursachen.
# Datenquellen (veroeffentlichte Schulkalender):
#   - DC Public Schools  (dcps.dc.gov)
#   - Fairfax County PS  (fcps.edu)
#   - Montgomery County PS (montgomeryschoolsmd.org)
SCHOOL_BREAK_DATES = pd.to_datetime([
    # === DC Public Schools ===
    # 2024-2025 Winter Break (verlaengert in Jan 2025)
    "2025-01-01", "2025-01-02", "2025-01-03",
    # MLK Weekend
    "2025-01-17", "2025-01-20",
    # Presidents' Week
    "2025-02-14", "2025-02-17",
    # Spring Break (1 Woche, typisch April)
    "2025-04-14", "2025-04-15", "2025-04-16", "2025-04-17", "2025-04-18",
    # Memorial Day
    "2025-05-23", "2025-05-26",
    # Summer Break (Start ca. 24. Juni, public DCPS)
    "2025-06-24", "2025-06-25", "2025-06-26", "2025-06-27", "2025-06-30",
    "2025-07-01", "2025-07-02", "2025-07-03", "2025-07-04",
    # Labor Day
    "2025-09-01",
    # Fall Break (1 Woche, ca. Oktober)
    "2025-10-13", "2025-10-14",
    # Thanksgiving Break
    "2025-11-26", "2025-11-27", "2025-11-28",
    # Winter Break (Ende 2025 / Start 2026)
    "2025-12-19", "2025-12-22", "2025-12-23", "2025-12-24", "2025-12-25",
    "2025-12-26", "2025-12-29", "2025-12-30", "2025-12-31",
    "2026-01-01", "2026-01-02",
    # MLK + Presidents 2026
    "2026-01-19",
    "2026-02-13", "2026-02-16",
    # Spring Break 2026 (DCPS)
    "2026-03-30", "2026-03-31", "2026-04-01", "2026-04-02", "2026-04-03",
    # Memorial Day 2026
    "2026-05-22", "2026-05-25",
    # Summer 2026 (ab 22. Juni)
    "2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26",
    "2026-06-29", "2026-06-30",
    # === Fairfax County Public Schools (VA) - abweichende Termine ===
    # Spring Break (1 Woche im April, jaehrlich variierend)
    "2025-04-14", "2025-04-15", "2025-04-16", "2025-04-17", "2025-04-18",
    "2026-03-30", "2026-03-31", "2026-04-01", "2026-04-02", "2026-04-03",
    # Fall Break (1 Woche im November, FCPS hat Herbstferien)
    "2025-10-13", "2025-10-14",
    # Winter Break (aehnlich DCPS)
    "2025-12-22", "2025-12-23", "2025-12-24", "2025-12-25", "2025-12-26",
    "2025-12-29", "2025-12-30", "2025-12-31",
    "2026-01-01", "2026-01-02",
    # === Montgomery County Public Schools (MD) ===
    # Spring Break (1 Woche)
    "2025-04-14", "2025-04-15", "2025-04-16", "2025-04-17", "2025-04-18",
    "2026-03-30", "2026-03-31", "2026-04-01", "2026-04-02", "2026-04-03",
    # Winter Break
    "2025-12-22", "2025-12-23", "2025-12-24", "2025-12-25", "2025-12-26",
    "2025-12-29", "2025-12-30", "2025-12-31",
    "2026-01-01", "2026-01-02",
])

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "data_preparation.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("data_prep")

# Reproduzierbarkeit
np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. LADEN + BEREINIGEN
# ---------------------------------------------------------------------------

def load_bts_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    """Lädt eine BTS-CSV mit den korrekten Header-Zeilen."""
    log.info("Lade %s ...", path.name)
    df = pd.read_csv(path, skiprows=9, header=None, names=columns, dtype=str, encoding="utf-8")
    # Filter SOURCE-Zeile + NaN
    df = df[df["carrier_code"] != "SOURCE: Bureau of Transportation Statistics"]
    df = df[df["carrier_code"].notna()]
    # Whitespace trimmen
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].str.strip()
    # Carrier-Filter: nur "echte" Carrier-Codes
    valid = df["carrier_code"].str.match(r"^[A-Z0-9]{2,3}$", na=False)
    df = df[valid].reset_index(drop=True)
    log.info("  -> %s Zeilen nach Cleaning", f"{len(df):,}")
    return df


def load_bts_table(name: str, columns: list[str]) -> pd.DataFrame:
    """Lädt alle Roh-Dateien zu einer Tabelle (z. B. 'Departures') und konkateniert sie.

    Erlaubt sowohl Detailed_Statistics_<name>.csv als auch
    2024_Detailed_Statistics_<name>.csv / 2025_Detailed_Statistics_<name>.csv.
    """
    paths = _find_raw(name)
    frames = [load_bts_csv(p, columns) for p in paths]
    if len(frames) == 1:
        return frames[0]
    out = pd.concat(frames, ignore_index=True)
    log.info("  Konkateniert: %s Zeilen aus %d Dateien", f"{len(out):,}", len(frames))
    return out


def parse_types_departure(df: pd.DataFrame) -> pd.DataFrame:
    """Konvertiert Strings -> passende Typen für Departures."""
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    int_cols = ["flight_number", "sched_elapsed_min", "actual_elapsed_min",
                "departure_delay_min", "taxi_out_min"]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    # Zeit-Strings HH:MM parsen
    df["sched_dep_hour"] = pd.to_numeric(
        df["sched_dep_time"].str.split(":").str[0], errors="coerce"
    ).astype("Int64")
    df["sched_dep_minute"] = pd.to_numeric(
        df["sched_dep_time"].str.split(":").str[1], errors="coerce"
    ).astype("Int64")
    return df


def parse_types_arrival(df: pd.DataFrame) -> pd.DataFrame:
    """Konvertiert Strings -> passende Typen für Arrivals."""
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    int_cols = ["flight_number", "sched_elapsed_min", "actual_elapsed_min",
                "arrival_delay_min", "taxi_in_min"]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def parse_types_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """Für Cancellation / Diversion."""
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df["flight_number"] = pd.to_numeric(df["flight_number"], errors="coerce").astype("Int64")
    return df


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ableitung von Datums-Features."""
    df["dayofweek"] = df["date"].dt.dayofweek.astype("Int64")  # 0=Mon, 6=Sun
    df["day"] = df["date"].dt.day.astype("Int64")
    df["month"] = df["date"].dt.month.astype("Int64")
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype("Int64")
    df["quarter"] = df["date"].dt.quarter.astype("Int64")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("Int64")
    # Federal Holiday: nur die 16 bundesweiten Feiertage
    df["is_holiday"] = df["date"].isin(US_FEDERAL_HOLIDAYS).astype("Int64")
    # Regional Holiday: Schulferien in DC, Fairfax, Montgomery (siehe SCHOOL_BREAK_DATES)
    df["is_school_break"] = df["date"].isin(SCHOOL_BREAK_DATES).astype("Int64")
    # Kombiniertes "free-day"-Flag: Wochenende ODER federal holiday ODER Schulferien
    df["is_free_day"] = (
        (df["is_weekend"] == 1) | (df["is_holiday"] == 1) | (df["is_school_break"] == 1)
    ).astype("Int64")
    # Sinus/Cosinus-Encoding für zyklische Features
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"].astype(float) / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"].astype(float) / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"].astype(float) / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"].astype(float) / 12)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tageszeit-Features aus sched_dep_time."""
    # Schon in parse_types_departure geparst: sched_dep_hour, sched_dep_minute
    # Minuten seit Mitternacht (0-1439)
    df["sched_dep_min_of_day"] = (
        df["sched_dep_hour"].astype("Int64") * 60 + df["sched_dep_minute"].astype("Int64")
    )
    # Tageszeit-Bucket: 0=Nacht, 1=Morgen, 2=Mittag, 3=Abend
    def _bucket(h):
        if pd.isna(h):
            return -1
        h = int(h)
        if h < 6:
            return 0  # Nacht
        if h < 12:
            return 1  # Morgen
        if h < 18:
            return 2  # Mittag/Nachmittag
        return 3  # Abend
    df["time_of_day"] = df["sched_dep_hour"].apply(_bucket).astype("Int64")
    # Sinus/Cosinus für Tageszeit
    df["tod_sin"] = np.sin(2 * np.pi * df["sched_dep_min_of_day"].astype(float) / 1440)
    df["tod_cos"] = np.cos(2 * np.pi * df["sched_dep_min_of_day"].astype(float) / 1440)
    return df


def add_frequency_features(df: pd.DataFrame, ref_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Frequency-Encoding für hochkardinale kategorische Features.

    Verwendet ref_df (z.B. Train-Set) für die Frequenz-Berechnung, um Data Leakage
    bei Train/Val-Splits zu vermeiden.
    """
    if ref_df is None:
        ref_df = df

    # Frequenz: wie oft wurde jede (carrier, dest, flight_number)-Kombination geflogen?
    freq = ref_df.groupby(["carrier_code", "dest_airport", "flight_number"]).size()
    freq = freq.rename("flight_combo_freq")
    df = df.merge(
        freq.reset_index(),
        on=["carrier_code", "dest_airport", "flight_number"],
        how="left",
    )
    df["flight_combo_freq"] = df["flight_combo_freq"].fillna(0).astype("Int64")

    # Destination-Frequenz (wie viele Fluege hat diese Destination insgesamt?)
    dest_freq = ref_df["dest_airport"].value_counts()
    df["dest_freq"] = df["dest_airport"].map(dest_freq).fillna(0).astype("Int64")

    return df


def add_arrival_aggregate_features(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    """Baut Tages- und 7-Tage-rolling-Mittel der Ankunftsverspaetung pro Origin.

    Logik: Wenn an einem Tag viele Fluge von einer bestimmten Origin verspaetet ankommen,
    ist es plausibel, dass der Carrier/das Wettersystem Probleme hat -> Indikator fuer
    spaetere Abflugverspaetungen.

    Verwendet nur historische Daten (Tagesmittel = am gleichen Tag; 7-Tage-Rolling
    = letzte 7 Tage OHNE den aktuellen Tag), um Leakage zu vermeiden.

    Felder:
      - origin_daily_arrival_delay_mean: Tagesmittel (heute)
      - origin_daily_arrival_n:         Anzahl Fluge heute
      - origin_7d_arrival_delay_mean:   7-Tage-rolling-Mittel (vor heute)
      - origin_7d_arrival_n:            Anzahl Fluge in den letzten 7 Tagen
    """
    log.info("Baue rollende Arrival-Aggregate (1d + 7d) ...")
    # Pro Tag: Mittlere Ankunftsverspätung pro Origin
    daily = (
        arr.dropna(subset=["arrival_delay_min"])
           .groupby(["date", "origin_airport"])["arrival_delay_min"]
           .agg(["mean", "count"])
           .reset_index()
           .rename(columns={"mean": "_daily_mean", "count": "_daily_n"})
    )

    # 7-Tage-Rolling: pro Origin, alle Daten in den letzten 7 Tagen MIT-ausser-den-aktuellen-Tag
    # Wir berechnen das auf der Origin-Ebene: groupby(origin).rolling(7, closed='left')
    daily = daily.sort_values(["origin_airport", "date"])
    daily["_7d_mean"] = (
        daily.groupby("origin_airport")["_daily_mean"]
             .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    )
    daily["_7d_n"] = (
        daily.groupby("origin_airport")["_daily_n"]
             .transform(lambda s: s.rolling(window=7, min_periods=1).sum())
    )
    # shift(1) erst HIER: das 7-Tage-Mittel soll die letzten 7 Tage VOR HEUTE abbilden
    daily["_7d_mean"] = daily.groupby("origin_airport")["_7d_mean"].shift(1)
    daily["_7d_n"] = daily.groupby("origin_airport")["_7d_n"].shift(1)

    # Joinen
    dep = dep.merge(
        daily,
        left_on=["date", "dest_airport"],
        right_on=["date", "origin_airport"],
        how="left",
    )
    dep = dep.drop(columns=["origin_airport"])
    # NaN-Handling: bei unseen Origins oder kaltem Start -> 0 / 0
    dep["_daily_mean"] = dep["_daily_mean"].fillna(0.0)
    dep["_daily_n"] = dep["_daily_n"].fillna(0).astype("Int64")
    dep["_7d_mean"] = dep["_7d_mean"].fillna(0.0)
    dep["_7d_n"] = dep["_7d_n"].fillna(0).astype("Int64")
    dep = dep.rename(columns={
        "_daily_mean": "origin_daily_arrival_delay_mean",
        "_daily_n":    "origin_daily_arrival_n",
        "_7d_mean":    "origin_7d_arrival_delay_mean",
        "_7d_n":       "origin_7d_arrival_n",
    })
    return dep


def add_cancellation_features(dep: pd.DataFrame, canc: pd.DataFrame) -> pd.DataFrame:
    """Binaer-Feature: gab es an einem Tag Cancellation an der Destination?"""
    log.info("Baue Cancellation-Features ...")
    canc_daily = canc.groupby("date").size().rename("cancellations_on_day").reset_index()
    dep = dep.merge(canc_daily, on="date", how="left")
    dep["cancellations_on_day"] = dep["cancellations_on_day"].fillna(0).astype("Int64")
    return dep


def add_congestion_window(dep: pd.DataFrame) -> pd.DataFrame:
    """Markiert Bank-Stunden (15-19 Uhr an Wochentagen) als 'congestion window'."""
    log.info("Baue is_congestion_window ...")
    dep["is_congestion_window"] = (
        (dep["sched_dep_hour"].between(15, 18, inclusive="left"))
        & (~dep["date"].dt.dayofweek.isin([5, 6]))
    ).astype("Int64")
    return dep


def add_dest_3d_arrival_features(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    """3-Tage-rolling-Mittel der Ankunftsverspätung pro Ziel-Flughafen.

    Anders als `origin_daily_arrival_*` (das die Origin = IAD nutzt) berechnen
    wir hier pro Ziel-Flughafen (dest_airport) das 3-Tage-rolling-Mittel der
    Ankunftsverspätung, ohne den aktuellen Tag (shift(1)). Damit hat jeder
    Flug Zugriff auf „war LHR gestern verspätet?" – das stärkste verfügbare
    Proxy-Signal, das wir noch nicht nutzten.
    """
    log.info("Baue dest_3d_arrival_delay_mean ...")
    daily = (
        arr.dropna(subset=["arrival_delay_min"])
           .groupby(["date", "origin_airport"])["arrival_delay_min"]
           .mean()
           .reset_index()
           .rename(columns={"arrival_delay_min": "_dest_daily_mean"})
    )
    daily = daily.sort_values(["origin_airport", "date"])
    daily["_dest_3d_mean"] = (
        daily.groupby("origin_airport")["_dest_daily_mean"]
             .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )
    daily["_dest_3d_mean"] = daily.groupby("origin_airport")["_dest_3d_mean"].shift(1)
    dep = dep.merge(
        daily[["date", "origin_airport", "_dest_3d_mean"]],
        left_on=["date", "dest_airport"],
        right_on=["date", "origin_airport"],
        how="left",
    )
    dep = dep.drop(columns=["origin_airport"])
    dep["_dest_3d_mean"] = dep["_dest_3d_mean"].fillna(0.0)
    dep = dep.rename(columns={"_dest_3d_mean": "dest_3d_arrival_delay_mean"})
    return dep


def add_weather_features(
    dep: pd.DataFrame,
    weather: pd.DataFrame,
    hours_offset: int = 0,
) -> pd.DataFrame:
    """Hängt Stunden-Wetter (KIAD) an die Departures an.

    Match-Logik: pro Flug wird das Wetter der vollen UTC-Stunde verwendet,
    die der `sched_dep_time` am nächsten liegt.

    WICHTIG: BTS `sched_dep_time` ist in LOKALER Zeit (Eastern Time).
    IAD liegt in America/New_York: UTC-5 (EST) im Winter, UTC-4 (EDT) im Sommer.
    Wir konvertieren erst nach UTC und joinen dann aufs Stunden-Raster.

    Parameter:
      hours_offset: Verschiebung in Stunden (default 0). Negative Werte
                    = Wetter VOR Abflug (z. B. -1 = eine Stunde vorher).

    Leakage-Schutz: keine zusätzlichen Felder, die nach dem Abflug erst
    entstehen (z. B. Wetter zur Landung).
    """
    log.info("Baue Wetter-Features (offset=%d h) ...", hours_offset)
    if weather.empty:
        log.warning("Wetter-DataFrame ist leer – Features werden als NaN gefüllt.")
        for col in ("temp_c", "wind_kts", "precip_1h_mm", "pressure_hpa"):
            dep[col] = np.nan
        return dep
    w = weather.copy()
    w["ts_utc"] = pd.to_datetime(w["ts_utc"], utc=True)
    w = w.sort_values("ts_utc")
    w["ts_hour"] = w["ts_utc"].dt.floor("h")

    # Lokalzeit (Eastern) der Flüge bauen und nach UTC konvertieren
    dep_local = pd.to_datetime(
        dep["date"].astype(str) + " " + dep["sched_dep_time"].astype(str),
        errors="coerce",
    )
    # America/New_York → UTC (berücksichtigt DST automatisch)
    dep["sched_dep_dt_utc"] = (
        dep_local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
              .dt.tz_convert("UTC")
              + pd.Timedelta(hours=hours_offset)
    )
    dep["sched_dep_hour_ts"] = dep["sched_dep_dt_utc"].dt.floor("h")
    keep = ["ts_hour", "temp_c", "dewpoint_c", "wind_kts", "pressure_hpa",
            "precip_1h_mm", "precip_6h_mm"]
    w_keep = w[keep].dropna(subset=["ts_hour"]).drop_duplicates(
        subset=["ts_hour"], keep="last"
    )
    dep = dep.merge(
        w_keep,
        left_on="sched_dep_hour_ts",
        right_on="ts_hour",
        how="left",
    )
    dep = dep.drop(columns=["ts_hour", "sched_dep_hour_ts", "sched_dep_dt_utc"])

    # Forward-Fill der Wetter-Features pro Tag. Da unser Split auf den
    # Wetter-Coverage-Bereich beschränkt ist (s. WETTER_MAX_DATE), gibt es
    # keine Lücken innerhalb des Splits. Wir setzen den Limit dennoch auf
    # 7 Tage, falls einzelne Stunden fehlen.
    log.info("Forward-Fill Wetter-Features pro Datum (Tages-Granularität) ...")
    w_cols = ["temp_c", "wind_kts", "precip_1h_mm", "pressure_hpa"]
    dep = dep.sort_values("date")
    for col in w_cols:
        daily = dep.groupby("date")[col].transform("mean")
        dep[col] = daily.ffill(limit=7).bfill(limit=7)
    return dep


# ---------------------------------------------------------------------------
# 3. TARGET + SPLIT
# ---------------------------------------------------------------------------

def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """Binäre Target-Variable: 1 wenn Verspätung >= Threshold, sonst 0."""
    df["delay_label"] = (df["departure_delay_min"] >= DELAY_THRESHOLD_MIN).astype("Int64")
    return df


def temporal_split_weather(
    df: pd.DataFrame,
    split_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wetter-bewusster temporaler Split.

    Train: < split_date  UND  <= max_date
    Val:   >= split_date UND  <= max_date

    Damit liegen Train und Val vollständig im Wetter-Coverage-Bereich.
    """
    train = df[(df["date"] < split_date) & (df["date"] <= max_date)].copy()
    val = df[(df["date"] >= split_date) & (df["date"] <= max_date)].copy()
    return train, val


# ---------------------------------------------------------------------------
# 4. PIPELINE
# ---------------------------------------------------------------------------

def select_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Wählt finale Feature- und Metadata-Spalten.

    Returns: (feature_cols, metadata_cols)

    Version 2026-06-16 (Diskussion):
      - Reduziert: day, quarter, weekofyear, is_free_day, time_of_day (kategorisch),
        sched_dep_minute, origin_*_n, flight_combo_freq
      - Hinzu: temp_c, wind_kts, precip_1h_mm, pressure_hpa (Wetter),
        dest_3d_arrival_delay_mean, is_congestion_window
    """
    feature_cols = [
        # Datum (zyklisch)
        "dow_sin", "dow_cos", "month_sin", "month_cos",
        # Datum (linear) – reduziert um day, quarter, weekofyear, is_free_day
        "dayofweek", "month",
        "is_weekend", "is_holiday", "is_school_break",
        # Tageszeit – reduziert um sched_dep_minute, time_of_day (kategorisch)
        "sched_dep_hour", "sched_dep_min_of_day", "tod_sin", "tod_cos",
        # Strecke
        "sched_elapsed_min",
        # Frequency-Encoding – reduziert um flight_combo_freq
        "dest_freq",
        # Aggregate aus Arrivals (1-Tag) – nur mean, ohne n
        "origin_daily_arrival_delay_mean",
        # Aggregate aus Arrivals (7-Tage-Rolling) – nur mean, ohne n
        "origin_7d_arrival_delay_mean",
        # Aggregate aus Cancellation
        "cancellations_on_day",
        # Wetter (NEU, aus 04_weather_join.py)
        "temp_c", "wind_kts", "precip_1h_mm", "pressure_hpa",
        # Arrival-by-Destination (NEU)
        "dest_3d_arrival_delay_mean",
        # Congestion-Window (NEU)
        "is_congestion_window",
    ]
    metadata_cols = [
        "carrier_code", "date", "flight_number", "dest_airport",
        "sched_dep_time", "departure_delay_min", "delay_label",
    ]
    # Nur vorhandene Spalten zurückgeben
    feature_cols = [c for c in feature_cols if c in df.columns]
    metadata_cols = [c for c in metadata_cols if c in df.columns]
    return feature_cols, metadata_cols


def run_pipeline() -> dict:
    """Haupt-Pipeline. Liefert Metadaten-Dict."""
    log.info("=" * 70)
    log.info("CRISP-DM Phase 3: Data Preparation")
    log.info("=" * 70)

    # --- 1. Laden
    log.info("\n[1/5] Lade Rohdaten ...")
    dep = load_bts_table("Departures", DEPARTURE_COLS)
    arr = load_bts_table("Arrivals", ARRIVAL_COLS)
    canc = load_bts_table("Cancellation", CANCELLATION_COLS)
    div = load_bts_table("Diversion", DIVERSION_COLS)

    # --- 2. Typen parsen
    log.info("\n[2/5] Parse Typen ...")
    dep = parse_types_departure(dep)
    arr = parse_types_arrival(arr)
    canc = parse_types_minimal(canc)
    div = parse_types_minimal(div)

    # --- 3. Bereinigen / Drop
    log.info("\n[3/5] Bereinige Departures ...")
    n_before = len(dep)
    # NaN in departure_delay_min -> Drop (Entscheidung #5)
    dep = dep.dropna(subset=["departure_delay_min"])
    log.info("  Drop %d Zeilen mit NaN delay -> %d verbleibend",
             n_before - len(dep), len(dep))
    # NaN in sched_elapsed_min -> Drop (essentielles Feature)
    n_before = len(dep)
    dep = dep.dropna(subset=["sched_elapsed_min"])
    log.info("  Drop %d Zeilen mit NaN sched_elapsed -> %d verbleibend",
             n_before - len(dep), len(dep))

    # --- 4. Feature Engineering
    log.info("\n[4/5] Feature Engineering ...")
    dep = add_date_features(dep)
    dep = add_time_features(dep)
    dep = add_arrival_aggregate_features(dep, arr)
    dep = add_dest_3d_arrival_features(dep, arr)   # NEU
    dep = add_cancellation_features(dep, canc)
    dep = add_congestion_window(dep)                # NEU
    # Wetter (optional, default: aus iad_isd_hourly.csv, wenn vorhanden)
    if USE_WEATHER:
        weather_path = Path(WEATHER_PATH)
        if weather_path.exists():
            weather = pd.read_csv(weather_path, parse_dates=["ts_utc"])
            dep = add_weather_features(dep, weather)
        else:
            log.warning("Wetter-Datei %s fehlt – Wetter-Features als NaN.", weather_path)
            for col in ("temp_c", "wind_kts", "precip_1h_mm", "pressure_hpa"):
                dep[col] = np.nan
    # Frequency-Encoding erst NACH Split, damit wir es korrekt machen können
    # -> erst Split, dann Frequenz mit Train als Referenz

    # --- 5. Target + Split
    log.info("\n[5/5] Target + Split ...")
    dep = make_target(dep)
    log.info("  Klassen-Verteilung (vor Split): %d verspaetet / %d puenktlich",
             dep["delay_label"].sum(), (dep["delay_label"] == 0).sum())

    train, val = temporal_split_weather(dep, SPLIT_DATE, WETTER_MAX_DATE)
    log.info("  Train: %d Zeilen (%s .. %s)",
             len(train), train["date"].min().date(), train["date"].max().date())
    log.info("  Val:   %d Zeilen (%s .. %s, Wetter-bis %s)",
             len(val), val["date"].min().date(), val["date"].max().date(),
             WETTER_MAX_DATE.date())
    log.info("  Train-Klassen: %d / %d (%.1f%% verspaetet)",
             train["delay_label"].sum(), len(train),
             100 * train["delay_label"].mean())
    log.info("  Val-Klassen:   %d / %d (%.1f%% verspaetet)",
             val["delay_label"].sum(), len(val),
             100 * val["delay_label"].mean())

    # Frequency-Encoding mit Train als Referenz
    train = add_frequency_features(train, ref_df=train)
    val = add_frequency_features(val, ref_df=train)

    # --- 6. Speichern
    feature_cols, metadata_cols = select_feature_columns(train)
    log.info("\n  Features: %d Spalten", len(feature_cols))
    log.info("  Metadata: %d Spalten", len(metadata_cols))

    # Parquet speichern (Spalten-Reihenfolge: metadata + features)
    all_cols = metadata_cols + feature_cols
    train_out = train[all_cols].copy()
    val_out = val[all_cols].copy()

    train_path = PROCESSED / "iad_flights_train.parquet"
    val_path = PROCESSED / "iad_flights_val.parquet"
    train_out.to_parquet(train_path, index=False)
    val_out.to_parquet(val_path, index=False)
    log.info("  Gespeichert: %s", train_path.name)
    log.info("  Gespeichert: %s", val_path.name)

    # --- 7. Metadaten
    meta = {
        "n_features": len(feature_cols),
        "n_metadata": len(metadata_cols),
        "feature_columns": feature_cols,
        "metadata_columns": metadata_cols,
        "delay_threshold_min": DELAY_THRESHOLD_MIN,
        "split_date": SPLIT_DATE.strftime("%Y-%m-%d"),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "train_delayed_pct": float(train["delay_label"].mean() * 100),
        "val_delayed_pct": float(val["delay_label"].mean() * 100),
        "n_unique_destinations": int(train["dest_airport"].nunique()),
        "n_unique_flight_numbers": int(train["flight_number"].nunique()),
        "date_train_min": train["date"].min().strftime("%Y-%m-%d"),
        "date_train_max": train["date"].max().strftime("%Y-%m-%d"),
        "date_val_min": val["date"].min().strftime("%Y-%m-%d"),
        "date_val_max": val["date"].max().strftime("%Y-%m-%d"),
    }
    meta_path = PROCESSED / "feature_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("  Metadaten: %s", meta_path.name)

    return meta


# ---------------------------------------------------------------------------
# 5. REPORT
# ---------------------------------------------------------------------------

def build_report(meta: dict) -> str:
    """Markdown-Report für docs/data_preparation.md."""
    md = []
    md.append("# CRISP-DM Phase 3: Data Preparation")
    md.append("")
    md.append(f"_Erstellt am {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Input-Output")
    md.append("")
    md.append("| Phase | Input | Output |")
    md.append("|---|---|---|")
    md.append("| Phase 2 (Data Understanding) | `data/raw/Detailed_Statistics_*.csv` | `docs/data_understanding.md` |")
    md.append("| **Phase 3 (Data Preparation)** | `data/raw/*.csv` | `data/processed/iad_flights_{train,val}.parquet`<br>`docs/data_preparation.md`<br>`logs/data_preparation.log` |")
    md.append("")
    md.append("## Pipeline-Schritte")
    md.append("")
    md.append("```")
    md.append("Roh-CSV  ->  load_bts_csv  ->  parse_types  ->  dropna  ->")
    md.append("add_date_features  ->  add_time_features  ->  add_arrival_aggregate  ->")
    md.append("add_cancellation  ->  make_target  ->  temporal_split  ->")
    md.append("add_frequency_features (Train-only Ref)  ->  save parquet")
    md.append("```")
    md.append("")
    md.append("## Getroffene Entscheidungen (Übernahme aus Phase 1+2)")
    md.append("")
    md.append("| # | Entscheidung | Auswirkung in Pipeline |")
    md.append("|---|---|---|")
    md.append("| 1 | Carrier = United Airlines (UA) only | Filter `carrier_code == 'UA'` (alle Daten ohnehin UA) |")
    md.append("| 2 | Threshold = 15 Min | `delay_label = (departure_delay_min >= 15)` |")
    md.append("| 3 | Verfrühte = pünktlich | Verfrühte Flüge bekommen `delay_label = 0` |")
    md.append("| 4 | Cancelled/Diverted = Drop | `dropna(subset=['departure_delay_min'])` |")
    md.append("| 5 | 2025 = Train, 2026 = Val | `split_date = 2026-01-01` |")
    md.append("")

    md.append("## Feature-Liste")
    md.append("")
    md.append(f"Insgesamt **{meta['n_features']} Features**, eingeteilt in Gruppen:")
    md.append("")
    md.append("| Gruppe | Features | Beschreibung |")
    md.append("|---|---|---|")
    md.append("| **Datum (zyklisch)** | `dow_sin`, `dow_cos`, `month_sin`, `month_cos` | Sinus/Cosinus-Encoding für Wochentag & Monat (Modell erkennt Zyklizität) |")
    md.append("| **Datum (linear)** | `dayofweek`, `day`, `month`, `weekofyear`, `quarter`, `is_weekend`, `is_holiday`, `is_school_break`, `is_free_day` | Roh-Encoding. `is_holiday` = federal, `is_school_break` = regionale Schulferien (DC, Fairfax, Montgomery), `is_free_day` = Wochenende OR holiday OR school_break |")
    md.append("| **Tageszeit (zyklisch)** | `tod_sin`, `tod_cos` | Sinus/Cosinus für Minuten seit Mitternacht |")
    md.append("| **Tageszeit (linear)** | `sched_dep_hour`, `sched_dep_minute`, `sched_dep_min_of_day`, `time_of_day` | Roh-Encoding + 4-Bucket (Nacht/Morgen/Mittag/Abend) |")
    md.append("| **Strecke** | `sched_elapsed_min` | Proxy für Distanz (Kurz-/Langstrecke) |")
    md.append("| **Frequency-Encoding** | `flight_combo_freq`, `dest_freq` | Häufigkeit der (carrier, dest, flight_number)-Kombination bzw. Destination |")
    md.append("| **Arrival-Aggregate (1 Tag)** | `origin_daily_arrival_delay_mean`, `origin_daily_arrival_n` | Mittlere Ankunftsverspätung an diesem Tag von dieser Origin |")
    md.append("| **Arrival-Aggregate (7-Tage-Rolling)** | `origin_7d_arrival_delay_mean`, `origin_7d_arrival_n` | Mittelwert/Summe der letzten 7 Tage (ohne aktuellen Tag). Glättet Ausreißer, bildet Trends ab. |")
    md.append("| **Cancellation-Aggregate** | `cancellations_on_day` | Anzahl Stornierungen am gleichen Tag (IAD-weit) |")
    md.append("")
    md.append("### Regionale Feiertage (`is_school_break`)")
    md.append("")
    md.append("Quelle: veröffentlichte Schulkalender für 3 Bezirke im IAD-Einzugsgebiet:")
    md.append("")
    md.append("- **DC Public Schools** (Washington DC)")
    md.append("- **Fairfax County Public Schools** (Northern Virginia)")
    md.append("- **Montgomery County Public Schools** (Maryland Suburbs)")
    md.append("")
    md.append("Kodiert als Union der Schulferien-Intervalle aller 3 Bezirke (Frühjahrs-, Sommer-, Herbst-, Winter-, Thanksgiving-Pause).")
    md.append("")
    md.append("### 7-Tage-Rolling-Aggregate")
    md.append("")
    md.append("Pro `(origin_airport, date)` wird der rolling-7-Tage-Mittel der `arrival_delay_min` berechnet, mit `shift(1)` damit der aktuelle Tag ausgeschlossen ist. Damit hat jeder Flug nur Zugriff auf **vorhergehende** Verspätungs-Information – kein Leakage.")
    md.append("")
    md.append("")
    md.append("## Explizit ausgeschlossene Features (Leakage-Schutz)")
    md.append("")
    md.append("| Spalte | Grund |")
    md.append("|---|---|")
    md.append("| `actual_dep_time`, `actual_elapsed_min`, `wheels_off_time`, `taxi_out_min` | post-hoc Info |")
    md.append("| Alle `delay_*_min` (Carrier/Weather/NAS/Security/Late Aircraft) | BTS attribuiert diese nachträglich |")
    md.append("| `tail_number` | 99 % Missing |")
    md.append("| `arrival_delay_min` | Target Leakage wenn Flugzeug ankommt |")
    md.append("")

    md.append("## Temporal-Split")
    md.append("")
    md.append(f"- **Split-Datum:** {meta['split_date']}")
    md.append(f"- **Train:** {meta['n_train']:,} Zeilen, {meta['date_train_min']} -> {meta['date_train_max']}")
    md.append(f"- **Val:**   {meta['n_val']:,} Zeilen, {meta['date_val_min']} -> {meta['date_val_max']}")
    md.append(f"- **Train Klassen-Balance:** {meta['train_delayed_pct']:.2f} % verspätet")
    md.append(f"- **Val Klassen-Balance:**   {meta['val_delayed_pct']:.2f} % verspätet")
    md.append("")
    md.append("> **Anmerkung zur Saisonalität:** Val enthält nur Jan-Apr, also Winter + früher Frühling. Sommerverspätungen (Gewitter) und Thanksgiving/Christmas-Spitzen sind nicht in Val enthalten. Wird in Phase 5 (Evaluation) thematisiert.")
    md.append("")

    md.append("## Train vs. Val – Verteilungs-Check")
    md.append("")
    md.append("TBD: wird beim Lauf ergänzt (Distinct counts, etc.)")
    md.append("")

    md.append("## Output-Dateien")
    md.append("")
    md.append("| Datei | Größe (typisch) | Inhalt |")
    md.append("|---|---|---|")
    md.append("| `data/processed/iad_flights_train.parquet` | ~5-10 MB | Trainingsdaten mit Features + Target |")
    md.append("| `data/processed/iad_flights_val.parquet`   | ~2-4 MB | Validierungsdaten |")
    md.append("| `data/processed/feature_metadata.json` | <10 KB | Feature-Liste + Split-Statistiken |")
    md.append("")

    md.append("## Nächste Schritte (CRISP-DM Phase 4: Modeling)")
    md.append("")
    md.append("1. **Baseline-Modell:** Logistic Regression (mit `class_weight='balanced'`)")
    md.append("2. **Baum-Modelle:** Random Forest, dann XGBoost/LightGBM")
    md.append("3. **Hyperparameter-Tuning:** Optuna oder GridSearch")
    md.append("4. **Evaluation:** PR-AUC, F1, Brier Score, Confusion Matrix")
    md.append("5. **Feature-Importance:** SHAP-Werte für die Top-Features")
    md.append("6. **Modell-Persistierung:** Joblib/Pickle")
    md.append("")

    return "\n".join(md)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> int:
    global USE_WEATHER, WEATHER_PATH
    import argparse
    p = argparse.ArgumentParser(description="CRISP-DM Phase 3 Pipeline")
    p.add_argument("--no-weather", action="store_true",
                   help="Wetter-Features weglassen (für A/B-Vergleich).")
    p.add_argument("--weather-path", type=Path, default=WEATHER_PATH,
                   help="Pfad zu iad_isd_hourly.csv.")
    args = p.parse_args()
    USE_WEATHER = not args.no_weather
    WEATHER_PATH = args.weather_path

    meta = run_pipeline()
    report = build_report(meta)
    report_path = DOCS / "data_preparation.md"
    report_path.write_text(report, encoding="utf-8")
    log.info("\nReport geschrieben: %s", report_path)
    log.info("=" * 70)
    log.info("Fertig.")
    log.info("  Train:  %s (%.1f%% verspaetet)", f"{meta['n_train']:,}",
             meta['train_delayed_pct'])
    log.info("  Val:    %s (%.1f%% verspaetet)", f"{meta['n_val']:,}",
             meta['val_delayed_pct'])
    log.info("  Features: %d | Metadata: %d", meta['n_features'], meta['n_metadata'])
    log.info("  Weather:  %s", "ON" if USE_WEATHER else "OFF")
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
