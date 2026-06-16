"""
Wetter-Download für IAD (Washington Dulles) via NCEI ISD-Lite (Bulk).

Quelle
-------
ISD-Lite ist die jährliche Stunden-Snapshot-Datei der Integrated Surface
Database. Sie ist klein (~0.5–1 MB/Jahr als .gz), enthält die Stunden-
Felder, die wir für unser Modell brauchen, und ist Public Domain.

URL-Schema:
    https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/<YYYY>/<USAF>-<WMO>-<YYYY>.gz

IAD:
    USAF = 724050, WMO = 13743 (KIAD)

Spalten-Format (Whitespace-getrennt, kein Header):
    YYYY MM DD HH   temp_tF   dew_tF   slp_tenth_hPa   wind_dir_deg
    wind_tenth_kt   precip_6h_tenth_mm   cloud_cover_flag
Eine Zeile pro Stunde UTC. Sentinel für fehlende Werte: -9999.

Lizenz: Public Domain (US-Regierung).

Was passiert hier?
------------------
1. Iteriert Jahr-für-Jahr (resumefähig – nur fehlende Jahre werden geladen).
2. Schreibt pro Jahr die dekomprimierte Stunden-CSV nach
   `data/external/weather/raw/isd_lite_KIAD_<YYYY>.csv`.
3. Vereinheitlicht alle Jahre zu
   `data/external/weather/iad_isd_hourly.csv` mit Spalten
   `station, ts_utc, temp_c, dewpoint_c, wind_kts, wind_dir, pressure_hpa,
    precip_mm, cloud_cover_flag`.

Nutzung
------
CLI:
    # Standard: 2024..aktuelles Jahr
    python -m src.weather_download

    # Zeitraum explizit
    python -m src.weather_download --start-year 2024 --end-year 2026

Programmatisch:
    from src.weather_download import download_year, merge_raw_to_hourly
"""
from __future__ import annotations

import argparse
import gzip
import io
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

# ----------------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "external" / "weather" / "raw"
DEFAULT_MERGED = PROJECT_ROOT / "data" / "external" / "weather" / "iad_isd_hourly.csv"

# IAD – Dulles International Airport
# Korrekte USAF-ID = 724050 (nicht 724030 – das war falsch verifiziert)
ISD_LITE_USAF = "724050"
ISD_LITE_WMO = "13743"
ISD_LITE_BASE = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite"
ISD_LITE_SENTINEL = -9999

REQUEST_TIMEOUT = 60
RETRY_BACKOFF = 5
USER_AGENT = "ISE-02-Mini-Case/1.0 (research; contact: noreply@example.org)"

# ISD-Lite-Format (offizielle Doku, 12 numerische Felder pro Zeile):
#
#   Feld  1-4  : Year, Month, Day, Hour (UTC)
#   Feld  5    : Air Temperature (°C, Skalierung 10, Sentinel -9999)
#   Feld  6    : Dew Point (°C, Skalierung 10)
#   Feld  7    : Sea Level Pressure (hPa, Skalierung 10)
#   Feld  8    : Wind Direction (deg, Skalierung 1, 0 = calm)
#   Feld  9    : Wind Speed (m/s, Skalierung 10)
#   Feld 10    : Sky Condition Total Coverage (Code 0–19)
#   Feld 11    : Liquid Precip 1h (mm, Skalierung 10, -1 = Trace)
#   Feld 12    : Liquid Precip 6h (mm, Skalierung 10, -1 = Trace)
_ISD_LINE = re.compile(
    r"^\s*"
    r"(?P<year>\d{4})\s+"
    r"(?P<month>\d{1,2})\s+"
    r"(?P<day>\d{1,2})\s+"
    r"(?P<hour>\d{1,2})\s+"
    r"(?P<temp_tC>-?\d+)\s+"
    r"(?P<dew_tC>-?\d+)\s+"
    r"(?P<slp_tenth_hpa>-?\d+)\s+"
    r"(?P<wind_dir_deg>-?\d+)\s+"
    r"(?P<wind_tenth_ms>-?\d+)\s+"
    r"(?P<sky_cov_code>-?\d+)\s+"
    r"(?P<precip_1h_tenth_mm>-?\d+)\s+"
    r"(?P<precip_6h_tenth_mm>-?\d+)"
)


# ----------------------------------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------------------------------
def _tenth_to_value(t: float) -> float | None:
    """Zehntel-Wert -> normaler Wert. Sentinel (-9999) -> None.
    Trace (-1) wird als 0 behandelt.
    """
    if pd.isna(t) or t == ISD_LITE_SENTINEL:
        return None
    if t == -1:  # Trace
        return 0.0
    return float(t) / 10.0


def _resolve_sentinel(v: float) -> float | None:
    """Sentinel -> None, sonst Originalwert."""
    if pd.isna(v) or v == ISD_LITE_SENTINEL:
        return None
    return float(v)


# ----------------------------------------------------------------------------
# Download-Logik
# ----------------------------------------------------------------------------
def _build_url(year: int) -> str:
    return f"{ISD_LITE_BASE}/{year}/{ISD_LITE_USAF}-{ISD_LITE_WMO}-{year}.gz"


def download_year(year: int, out_dir: Path, logger: logging.Logger) -> Path:
    """Lädt ein Kalenderjahr IAD ISD-Lite (.gz) und dekomprimiert es."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"isd_lite_KIAD_{year}.csv"
    if out_path.exists() and out_path.stat().st_size > 100:
        logger.info(
            "Überspringe %s – Datei existiert bereits (%d B).",
            out_path.name, out_path.stat().st_size,
        )
        return out_path

    url = _build_url(year)
    logger.info("GET %s ...", url)

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"},
            )
            if resp.status_code == 200 and len(resp.content) > 500:
                with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as gz:
                    out_path.write_bytes(gz.read())
                logger.info(
                    "OK %s (%d B aus %d B .gz).",
                    out_path.name, out_path.stat().st_size, len(resp.content),
                )
                return out_path
            logger.warning(
                "Versuch %d: HTTP %d, %d B. Warte %ds...",
                attempt + 1, resp.status_code, len(resp.content), RETRY_BACKOFF,
            )
            last_exc = RuntimeError(f"HTTP {resp.status_code}")
        except requests.RequestException as exc:
            logger.warning("Versuch %d: %s. Warte %ds...", attempt + 1, exc, RETRY_BACKOFF)
            last_exc = exc
        time.sleep(RETRY_BACKOFF)

    raise RuntimeError(f"Download {year} fehlgeschlagen: {last_exc}")


# ----------------------------------------------------------------------------
# Parsing / Vereinheitlichung
# ----------------------------------------------------------------------------
def parse_year_csv(csv_path: Path) -> pd.DataFrame:
    """Liest eine ISD-Lite Stunden-CSV per Regex und gibt einen normalisierten DataFrame zurück."""
    rows: list[dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            m = _ISD_LINE.match(line)
            if m:
                rows.append(m.groupdict())
    if not rows:
        return pd.DataFrame(columns=[
            "station", "ts_utc", "temp_c", "dewpoint_c", "wind_kts",
            "wind_dir", "pressure_hpa", "precip_1h_mm", "precip_6h_mm",
            "sky_cov_code",
        ])

    df = pd.DataFrame(rows)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ts = pd.to_datetime(
        df[["year", "month", "day", "hour"]],
        utc=True,
        errors="coerce",
    )

    # Wind: Zehntel-m/s -> Knoten
    wind_kts = df["wind_tenth_ms"].apply(_tenth_to_value).apply(
        lambda ms: ms * 1.94384 if ms is not None else None
    )

    out = pd.DataFrame({
        "station": "KIAD",
        "ts_utc": ts,
        "temp_c": df["temp_tC"].apply(_tenth_to_value),       # °C × 0.1
        "dewpoint_c": df["dew_tC"].apply(_tenth_to_value),     # °C × 0.1
        "wind_kts": wind_kts,
        "wind_dir": df["wind_dir_deg"].apply(_resolve_sentinel),
        "pressure_hpa": df["slp_tenth_hpa"].apply(_tenth_to_value),
        "precip_1h_mm": df["precip_1h_tenth_mm"].apply(_tenth_to_value),
        "precip_6h_mm": df["precip_6h_tenth_mm"].apply(_tenth_to_value),
        "sky_cov_code": df["sky_cov_code"].apply(_resolve_sentinel),
    })
    return out


def merge_raw_to_hourly(raw_dir: Path, out_path: Path,
                        logger: logging.Logger) -> pd.DataFrame:
    """Liest alle `isd_lite_KIAD_*.csv` und vereinheitlicht sie."""
    files = sorted(raw_dir.glob("isd_lite_KIAD_*.csv"))
    if not files:
        logger.warning("Keine Roh-Dateien in %s gefunden.", raw_dir)
        return pd.DataFrame()

    frames = [parse_year_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc").drop_duplicates(
        subset=["ts_utc"], keep="last"
    )
    df = df.reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Merged %d Stundenzeilen -> %s", len(df), out_path)
    return df


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _years(start: int, end: int) -> Iterable[int]:
    return range(start, end + 1)


def _cli(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("weather_download")

    p = argparse.ArgumentParser(description="NCEI ISD-Lite Hourly Download für IAD")
    p.add_argument("--start-year", type=int, default=2024)
    p.add_argument("--end-year", type=int, default=datetime.now().year)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    p.add_argument("--no-merge", action="store_true",
                   help="Nur Roh-Dateien herunterladen, kein Merge")
    args = p.parse_args(argv)

    for y in _years(args.start_year, args.end_year):
        try:
            download_year(y, args.out_dir, logger)
        except RuntimeError as exc:
            # Fehlende Jahre (z. B. 2026 noch nicht publiziert) sind nicht
            # fatal – wir loggen eine Warnung und machen mit dem nächsten
            # Jahr weiter, solange mindestens ein Jahr Erfolg hat.
            logger.warning(
                "Jahr %d nicht verfügbar (%s) – wird übersprungen.",
                y, exc,
            )

    if not args.no_merge:
        df = merge_raw_to_hourly(args.out_dir, args.merged, logger)
        if df.empty:
            logger.error("Keine Roh-Dateien zum Mergen gefunden.")
            return 1
        print(df.head(5).to_string(index=False))
        print(
            f"\nGesamt: {len(df)} Stunden, "
            f"{df['ts_utc'].min()} -> {df['ts_utc'].max()}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(_cli())
