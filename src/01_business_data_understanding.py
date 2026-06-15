"""
CRISP-DM Phase 1+2: Business Understanding & Data Understanding
================================================================

Erzeugt einen strukturierten Bericht für die BTS-Daten.
- Phase 1: Business Understanding (Aufgabe, Ziel, Erfolgsmetriken, Limitationen)
- Phase 2: Data Understanding (Schema, Qualität, Verteilungen, Auffälligkeiten)

Input:  data/raw/Detailed_Statistics_*.csv
Output: docs/data_understanding.md
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
REPORT_PATH = PROJECT / "docs" / "data_understanding.md"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Echte Header aus den BTS-CSV-Dateien (Zeile 10, die pandas nicht erkennt)
DEPARTURE_COLS = [
    "carrier_code", "date", "flight_number", "tail_number", "dest_airport",
    "sched_dep_time", "actual_dep_time",
    "sched_elapsed_min", "actual_elapsed_min",
    "departure_delay_min", "wheels_off_time", "taxi_out_min",
    "delay_carrier_min", "delay_weather_min", "delay_nas_min",
    "delay_security_min", "delay_late_aircraft_min",
]
CANCELLATION_COLS = ["carrier_code", "date", "flight_number", "tail_number", "dest_airport"]
DIVERSION_COLS = ["carrier_code", "date", "flight_number", "tail_number", "dest_airport"]

# Mindestanzahl Verspätung, ab der Flug als "verspätet" gilt (BTS-Konvention)
DELAY_THRESHOLD_MIN = 15


# ---------------------------------------------------------------------------
# Lade-Funktionen
# ---------------------------------------------------------------------------

def load_bts_csv(path: Path, columns: list[str], filename: str) -> pd.DataFrame:
    """Lädt eine BTS-CSV mit den korrekten Header-Zeilen."""
    # Erste 9 Zeilen sind Metadaten, Zeile 10 ist Header
    df = pd.read_csv(path, skiprows=9, header=None, names=columns, dtype=str, encoding="utf-8")
    # Letzte Zeile ist "SOURCE: Bureau of Transportation Statistics"
    df = df[df["carrier_code"] != "SOURCE: Bureau of Transportation Statistics"]
    df = df[df["carrier_code"].notna()]
    # Whitespace trimmen in allen String-Spalten
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].str.strip()
    # Carrier-Filter: nur "echte" Carrier-Codes (Großbuchstaben, 2-3 Zeichen, keine Leerzeichen)
    valid_carriers = df["carrier_code"].str.match(r"^[A-Z0-9]{2,3}$", na=False)
    df = df[valid_carriers]
    return df.reset_index(drop=True)


def parse_departure_types(df: pd.DataFrame) -> pd.DataFrame:
    """Konvertiert Strings in passende Typen für Departures."""
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    int_cols = ["flight_number", "sched_elapsed_min", "actual_elapsed_min",
                "departure_delay_min", "taxi_out_min",
                "delay_carrier_min", "delay_weather_min", "delay_nas_min",
                "delay_security_min", "delay_late_aircraft_min"]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    # Zeiten als HH:MM String belassen, hier nicht weiter parsen
    return df


def parse_minimal_types(df: pd.DataFrame) -> pd.DataFrame:
    """Für Cancellation / Diversion: nur Date parsen."""
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df["flight_number"] = pd.to_numeric(df["flight_number"], errors="coerce").astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------

def analyse_departures(df: pd.DataFrame) -> dict:
    """Bereitet Kennzahlen für die Departure-Tabelle auf."""
    n = len(df)
    delay = df["departure_delay_min"]
    on_time = (delay < DELAY_THRESHOLD_MIN).sum()
    delayed = (delay >= DELAY_THRESHOLD_MIN).sum()
    cancelled_proxy = df["departure_delay_min"].isna().sum()  # NaN = ggf. storniert
    early = (delay < 0).sum()

    return {
        "n": n,
        "date_min": df["date"].min().strftime("%Y-%m-%d") if df["date"].notna().any() else "n/a",
        "date_max": df["date"].max().strftime("%Y-%m-%d") if df["date"].notna().any() else "n/a",
        "unique_carriers": sorted(df["carrier_code"].dropna().unique().tolist()),
        "n_unique_destinations": df["dest_airport"].nunique(),
        "top_destinations": df["dest_airport"].value_counts().head(10).to_dict(),
        "delay_mean": float(delay.mean()) if delay.notna().any() else None,
        "delay_median": float(delay.median()) if delay.notna().any() else None,
        "delay_std": float(delay.std()) if delay.notna().any() else None,
        "delay_min": float(delay.min()) if delay.notna().any() else None,
        "delay_max": float(delay.max()) if delay.notna().any() else None,
        "on_time_count": int(on_time),
        "on_time_pct": float(on_time / n * 100) if n else 0.0,
        "delayed_count": int(delayed),
        "delayed_pct": float(delayed / n * 100) if n else 0.0,
        "early_count": int(early),
        "early_pct": float(early / n * 100) if n else 0.0,
        "missing_delay": int(cancelled_proxy),
        "missing_delay_pct": float(cancelled_proxy / n * 100) if n else 0.0,
        "missing_tail_number": int(df["tail_number"].isna().sum()),
        "missing_tail_pct": float(df["tail_number"].isna().mean() * 100) if n else 0.0,
    }


def analyse_minimal(df: pd.DataFrame, name: str) -> dict:
    """Kennzahlen für Cancellation / Diversion."""
    n = len(df)
    return {
        "name": name,
        "n": n,
        "date_min": df["date"].min().strftime("%Y-%m-%d") if df["date"].notna().any() else "n/a",
        "date_max": df["date"].max().strftime("%Y-%m-%d") if df["date"].notna().any() else "n/a",
        "unique_carriers": sorted(df["carrier_code"].dropna().unique().tolist()),
        "n_unique_destinations": df["dest_airport"].nunique(),
        "missing_tail_number": int(df["tail_number"].isna().sum()),
        "missing_tail_pct": float(df["tail_number"].isna().mean() * 100) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Report-Generator (Markdown)
# ---------------------------------------------------------------------------

def build_report(dep_stats, canc_stats, div_stats, dep_df) -> str:
    md = []
    md.append("# CRISP-DM Phase 1 & 2: Business- und Data-Understanding")
    md.append("")
    md.append(f"_Erstellt am {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    md.append("")
    md.append("---")
    md.append("")

    # === Phase 1: Business Understanding ===
    md.append("## Phase 1 – Business Understanding")
    md.append("")
    md.append("### 1.1 Ausgangslage (Aufgabenstellung)")
    md.append("")
    md.append("> **CASE 12: VORHERSAGE VON FLUGVERSPÄTUNGEN**")
    md.append(">")
    md.append("> - **Business Problem**: Verspätungen frühzeitig erkennen.")
    md.append("> - **ML-Typ**: Klassifikation")
    md.append("> - **Datensatz**: Airline Delay Dataset")
    md.append("> - **Business Mehrwert**: Bessere Ressourcenplanung.")
    md.append("")
    md.append("### 1.2 Data-Science-Ziel")
    md.append("")
    md.append("Binäre Klassifikation: Wird ein Flug von IAD **≥ 15 Minuten verspätet** abheben")
    md.append("(gemäß BTS-Standarddefinition), oder startet er pünktlich / verfrüht.")
    md.append("")
    md.append("**Target-Variable (vorzuschlagen):**")
    md.append("```")
    md.append("delay_label = 1  falls departure_delay_min >= 15")
    md.append("             0  sonst (pünktlich ODER verfrüht, oder NaN/cancelled/diverted)")
    md.append("```")
    md.append("")
    md.append("### 1.3 Vorgeschlagene Erfolgsmetriken")
    md.append("")
    md.append("| Metrik | Begründung |")
    md.append("|---|---|")
    md.append("| **PR-AUC (Precision-Recall)** | Verspätungen sind das seltenere Ereignis, PR-AUC ist aussagekräftiger als ROC-AUC. |")
    md.append("| **F1-Score (Klasse 1)** | Ausbalanciertes Maß für Precision und Recall. |")
    md.append("| **Recall@Precision≥0.7** | Business-relevant: Welchen Anteil der echten Verspätungen erwischen wir bei akzeptabler False-Positive-Rate? |")
    md.append("| **Brier Score** | Kalibrierung der Wahrscheinlichkeiten (für Ressourcenplanung wichtig). |")
    md.append("")
    md.append("### 1.4 Business-Mehrwert & Handlungsempfehlung")
    md.append("")
    md.append("- **Ressourcenplanung am Gate**: Bei vorhergesagter Verspätung kann Personal/Equipment umgeplant werden.")
    md.append("- **Passagier-Information**: Proaktive Benachrichtigungen reduzieren Entschädigungs-/Hotelkosten.")
    md.append("- **Crews**: Schichtpläne anpassen, um regulatorische Ruhezeiten einzuhalten.")
    md.append("")
    md.append("### 1.5 Limitationen & Annahmen (selbst getroffene Entscheidungen)")
    md.append("")
    md.append("| # | Entscheidung | Begründung |")
    md.append("|---|---|---|")
    md.append("| 1 | **Carrier = United Airlines (UA) only** | Die heruntergeladene `Departures.csv` enthält ausschließlich UA-Flüge (vermutlich Auswahl im Web-UI). Wir behandeln das als Feature, nicht als Bug. |")
    md.append("| 2 | **Zeitraum = 2025-01 bis 2026-04** | `Departures.csv` reicht bis 30.04.2026; damit haben wir 16 Monate (≈ 47.000 Flüge). 2025 dient als Trainingsbasis, 2026 als Validation. |")
    md.append("| 3 | **Verfrühte Flüge = pünktlich (Klasse 0)** | BTS-Konvention; -5 Min ist operativ eine Punktlandung. |")
    md.append("| 4 | **Stornierte/Umgeleitete Flüge = Ausschluss** | Sie haben keine sinnvolle Verspätungs-Target; werden separat über Cancellation/Diversion-Tabellen analysiert. |")
    md.append("")

    # === Phase 2: Data Understanding ===
    md.append("---")
    md.append("")
    md.append("## Phase 2 – Data Understanding")
    md.append("")
    md.append("### 2.1 Datenquellen & Dateigrößen")
    md.append("")
    md.append("| Datei | Größe | Zeilen | Spalten |")
    md.append("|---|---|---|---|")
    for f in RAW.glob("Detailed_Statistics_*.csv"):
        size_kb = f.stat().st_size / 1024
        md.append(f"| `data/raw/{f.name}` | {size_kb:,.1f} KB | – | – |")
    md.append("")

    md.append("### 2.2 Departure-Tabelle – Hauptmetriken")
    md.append("")
    md.append(f"- **Zeilen gesamt:** {dep_stats['n']:,}")
    md.append(f"- **Zeitraum:** {dep_stats['date_min']} → {dep_stats['date_max']}")
    md.append(f"- **Carrier (Unique):** {dep_stats['unique_carriers']}")
    md.append(f"- **Zielflughäfen (Unique):** {dep_stats['n_unique_destinations']}")
    md.append("")

    md.append("**Top-10 Destinationen (von IAD aus):**")
    md.append("")
    md.append("| Rang | IATA-Code | Anzahl Flüge |")
    md.append("|---:|---|---:|")
    for i, (dest, cnt) in enumerate(dep_stats["top_destinations"].items(), 1):
        md.append(f"| {i} | {dest} | {cnt:,} |")
    md.append("")

    md.append("**Verspätungs-Verteilung (Minuten, alle Flüge):**")
    md.append("")
    md.append(f"| Kennzahl | Wert |")
    md.append(f"|---|---|")
    md.append(f"| Mittelwert | {dep_stats['delay_mean']:.2f} |")
    md.append(f"| Median | {dep_stats['delay_median']:.2f} |")
    md.append(f"| Std.-Abw. | {dep_stats['delay_std']:.2f} |")
    md.append(f"| Min | {dep_stats['delay_min']:.2f} |")
    md.append(f"| Max | {dep_stats['delay_max']:.2f} |")
    md.append("")

    md.append("**Klassen-Verteilung (Delay ≥ 15 Min):**")
    md.append("")
    md.append(f"| Klasse | Bedeutung | Anzahl | Anteil |")
    md.append(f"|---:|---|---:|---:|")
    md.append(f"| 1 | Verspätet (≥ 15 Min) | {dep_stats['delayed_count']:,} | {dep_stats['delayed_pct']:.1f} % |")
    md.append(f"| 0 | Pünktlich (< 15 Min) | {dep_stats['on_time_count']:,} | {dep_stats['on_time_pct']:.1f} % |")
    md.append(f"| (negativ) | Davon verfrüht (< 0 Min) | {dep_stats['early_count']:,} | {dep_stats['early_pct']:.1f} % der Gesamtflüge |")
    md.append(f"| NaN | Missing (storniert/diverted?) | {dep_stats['missing_delay']:,} | {dep_stats['missing_delay_pct']:.1f} % |")
    md.append("")
    md.append(f"> ⚠️ **Klassen-Imbalance:** {dep_stats['delayed_pct']:.1f} % verspätet vs. {dep_stats['on_time_pct']:.1f} % pünktlich. Modell muss Stratifizierung & Class-Weights berücksichtigen.")
    md.append("")

    md.append("### 2.3 Datenqualität – Auffälligkeiten")
    md.append("")
    md.append(f"| Problem | Wert | Implikation |")
    md.append(f"|---|---|---|")
    md.append(f"| `tail_number` Missing | {dep_stats['missing_tail_number']:,} / {dep_stats['n']:,} ({dep_stats['missing_tail_pct']:.1f} %) | Feature nicht nutzbar |")
    md.append(f"| `departure_delay_min` NaN | {dep_stats['missing_delay']:,} ({dep_stats['missing_delay_pct']:.1f} %) | Diese Zeilen für Klassifikation ausschließen |")
    md.append("")

    md.append("### 2.4 Cancellation & Diversion – Ergänzende Tabellen")
    md.append("")
    md.append("| Tabelle | n | Zeitraum | Carrier | Fehlende Tail-Numbers |")
    md.append(f"|---|---:|---|---|---|")
    for s in (canc_stats, div_stats):
        md.append(f"| {s['name']} | {s['n']:,} | {s['date_min']} → {s['date_max']} | {s['unique_carriers']} | {s['missing_tail_number']:,} ({s['missing_tail_pct']:.1f} %) |")
    md.append("")
    md.append("**Beobachtung:** Cancellation & Diversion sind **sehr seltene Ereignisse** (< 0.5 % der Flüge).")
    md.append("Sie fließen als **zusätzliche Features** ein (z. B. „Carrier hatte gestern Cancellation an diesem Flughafen“),")
    md.append("aber **nicht in die Target-Variable** der Hauptaufgabe.")
    md.append("")

    md.append("### 2.5 Schema – Spalten-Definitionen (Departures)")
    md.append("")
    md.append("| Spalte | Typ | Bedeutung | Nutzung in Modellierung |")
    md.append("|---|---|---|---|")
    md.append("| `carrier_code` | str (UA) | Carrier-Code | Kategorisch (konstant) |")
    md.append("| `date` | date | Flugdatum | Feature → Wochentag, Monat, Feiertag |")
    md.append("| `flight_number` | int | Flugnummer (UA-intern) | Optional (viele unique) |")
    md.append("| `tail_number` | str | Flugzeug-Kennung | ❌ 99.9 % Missing → nicht nutzbar |")
    md.append("| `dest_airport` | str (IATA) | Ziel-Flughafen | Feature (One-Hot / Embedding) |")
    md.append("| `sched_dep_time` | str (HH:MM) | Geplante Abflugzeit (lokal) | Feature → Stunde, Tageszeit-Bucket |")
    md.append("| `actual_dep_time` | str (HH:MM) | Tatsächliche Abflugzeit | **Nicht als Feature** (post-hoc Info) |")
    md.append("| `sched_elapsed_min` | int | Geplante Flugdauer | Feature (proxy für Distanz) |")
    md.append("| `actual_elapsed_min` | int | Tatsächliche Flugdauer | **Nicht als Feature** |")
    md.append("| `departure_delay_min` | int | **TARGET** (Verspätung) | y |")
    md.append("| `wheels_off_time` | str | Rollzeit-Ende | **Nicht als Feature** |")
    md.append("| `taxi_out_min` | int | Rollzeit zum Start | ⚠️ Borderline – wird *vor* tatsächlichem Start gemessen, ggf. Feature |")
    md.append("| `delay_carrier_min` | int | Verspätung Airline (offiziell zugeordnet) | ⚠️ Leakage-Risiko! Wird von BTS erst *nach* dem Ereignis attributiert |")
    md.append("| `delay_weather_min` | int | Verspätung Wetter | ⚠️ Ebenfalls nachträglich attributiert |")
    md.append("| `delay_nas_min` | int | Verspätung National Aviation System | ⚠️ Ebenfalls nachträglich |")
    md.append("| `delay_security_min` | int | Verspätung Sicherheit | ⚠️ Ebenfalls nachträglich |")
    md.append("| `delay_late_aircraft_min` | int | Verspätung durch Vorgänger-Flugzeug | ⚠️ Ebenfalls nachträglich |")
    md.append("")
    md.append("### 2.6 Feature-Kandidaten (Finale Vorauswahl)")
    md.append("")
    md.append("**Zugelassene Features (pre-departure, kein Leakage):**")
    md.append("1. `date` → abgeleitet: `dayofweek`, `month`, `day`, `is_weekend`, `is_holiday` (US-Feiertage)")
    md.append("2. `sched_dep_time` → `hour_of_day`, `minute_of_hour`, `time_block`")
    md.append("3. `dest_airport` (One-Hot-Encoding, ~100 unique)")
    md.append("4. `sched_elapsed_min` (proxy für Distanz / Kurz- vs. Langstrecke)")
    md.append("5. `flight_number` (optional: Frequenz-Count-Encoding)")
    md.append("")
    md.append("**Ausgeschlossene Features (Leakage / post-hoc):**")
    md.append("- `actual_dep_time`, `actual_elapsed_min`, `wheels_off_time`")
    md.append("- `taxi_out_min` (zwar pre-departure gemessen, aber **nicht** vorhersagbar)")
    md.append("- Alle 5 `delay_*_min` (von BTS nachträglich attribuiert → Leakage)")
    md.append("- `tail_number` (99.9 % Missing)")
    md.append("")
    md.append("**Externe Features (Phase 3, falls Zeit reicht):**")
    md.append("- Wetter (IAD METAR, Schneefall, Gewitter)")
    md.append("- Vorgänger-Flugzeug-Verspätung (über `tail_number` nicht möglich → verwerfen)")
    md.append("- Feiertage in Zieldestination")
    md.append("")

    md.append("### 2.7 Daten-Hypothesen für Phase 3 (Data Preparation)")
    md.append("")
    md.append("1. **Saisonalität**: Verspätungen sind im Winter (Schnee) und Sommer (Gewitter) höher.")
    md.append("2. **Tageszeit**: Frühe Flüge und Stoßzeiten am Abend sind verspätungsanfälliger.")
    md.append("3. **Streckeneffekt**: Langstreckenflüge haben tendenziell höhere durchschnittliche Verspätungen.")
    md.append("4. **Klassen-Imbalance**: ~15–20 % Verspätungen → `class_weight='balanced'` oder SMOTE.")
    md.append("5. **NaN-Strategie**: `departure_delay_min` NaN → Drop (nur 0.1 % der Daten).")
    md.append("")

    md.append("---")
    md.append("")
    md.append("## Nächste Schritte (CRISP-DM Phase 3: Data Preparation)")
    md.append("")
    md.append("1. Saubere Pipeline zum Einlesen aller 3 Tabellen (mit korrekten Headern).")
    md.append("2. Feature-Engineering: Datums-Features, Strecken-Features, Holiday-Flags.")
    md.append("3. Train/Validation-Split: 2025 ganzjährig → temporal 80/20 (Q1+Q2 vs. Q3+Q4) oder stratifiziert.")
    md.append("4. Baseline-Modell: Logistic Regression → F1 ~0.30–0.40 erwartbar.")
    md.append("5. Iteration: Random Forest → XGBoost/LightGBM → Hyperparameter-Tuning.")
    md.append("")

    return "\n".join(md)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("CRISP-DM Phase 1+2 – Business & Data Understanding")
    print("=" * 70)

    # 1. Lade Daten
    print("\n[1/3] Lade Departures …")
    dep_raw = load_bts_csv(RAW / "Detailed_Statistics_Departures.csv", DEPARTURE_COLS, "Departures")
    dep = parse_departure_types(dep_raw)
    print(f"      -> {len(dep):,} Zeilen geladen, "
          f"{dep['date'].min():%Y-%m-%d} bis {dep['date'].max():%Y-%m-%d}")

    print("[2/3] Lade Cancellation …")
    canc_raw = load_bts_csv(RAW / "Detailed_Statistics_Cancellation.csv", CANCELLATION_COLS, "Cancellation")
    canc = parse_minimal_types(canc_raw)
    print(f"      → {len(canc):,} Zeilen geladen")

    print("[3/3] Lade Diversion …")
    div_raw = load_bts_csv(RAW / "Detailed_Statistics_Diversion.csv", DIVERSION_COLS, "Diversion")
    div = parse_minimal_types(div_raw)
    print(f"      → {len(div):,} Zeilen geladen")

    # 2. Analysen
    print("\nAnalysiere Departures …")
    dep_stats = analyse_departures(dep)
    canc_stats = analyse_minimal(canc, "Cancellation")
    div_stats = analyse_minimal(div, "Diversion")

    # 3. Report schreiben
    print(f"\nSchreibe Report: {REPORT_PATH}")
    report = build_report(dep_stats, canc_stats, div_stats, dep)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"      → {len(report):,} Zeichen geschrieben")

    print("\n" + "=" * 70)
    print("✓ Fertig.")
    print(f"  Report:     {REPORT_PATH}")
    print(f"  Departures: {dep_stats['n']:,} Flüge, "
          f"{dep_stats['delayed_pct']:.1f}% verspätet (≥15 Min)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
