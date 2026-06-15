# ISE-02-Mini-Case – Flugverspätungs-Vorhersage (IAD)

Mini-Case zur Vorhersage von Flugverspätungen am Washington Dulles International (IAD).

## Datenquelle

- **BTS TranStats – On-Time Performance** (Tabelle 236)
- Web: <https://www.transtats.bts.gov/ONTIME/>
- Zeitraum: 2025-01 bis 2026-05
- Rohformat: ZIP mit monatlicher CSV (alle Felder, ~200 MB pro Monat)

## Projektstruktur

```
ISE-02-Mini-Case/
├── data/
│   ├── raw/         # Heruntergeladene Monats-CSVs
│   └── processed/   # Gefilterte/vereinigte Daten
├── docs/            # Aufgabenstellung
├── logs/            # Log-Files der Skripte
└── src/             # Python-Skripte
```

## Setup

```powershell
# 1. Virtuelle Umgebung anlegen (einmalig)
python -m venv .venv

# 2. Aktivieren
.\.venv\Scripts\Activate.ps1

# 3. Abhängigkeiten installieren
pip install requests pandas pyarrow
```

## Skripte

### 1. `src/download_bts_data.py`

Lädt monatliche On-Time-Daten von BTS herunter und entpackt sie nach
`data/raw/`. Lädt **alle** Felder, damit das Feature-Engineering flexibel
gestaltet werden kann.

**Aufruf (mit aktivierter venv):**

```powershell
python src/download_bts_data.py
```

**Alternative** (ohne Aktivierung, direkter Aufruf):

```powershell
.\.venv\Scripts\python.exe src\download_bts_data.py
```

**Wichtige Parameter** (im Skript anpassbar):
- `YEARS = [2025, 2026]`
- `MONTHS = list(range(1, 13))` → wird automatisch für 2026 auf Mai begrenzt
- `REQUEST_TIMEOUT = 300` (Sekunden pro Datei)
- `RETRY_COUNT = 3`

## Datenaufkommen

| Zeitraum | Monate | ~Zeilen gesamt | Größe Roh |
|----------|--------|----------------|-----------|
| 2025-01 – 2026-05 | 17 | ~14 Mio. | ~3,5 GB |

> **Hinweis**: Wir laden **alle** US-Flüge herunter, da die BTS-API keine
> Flughafen-Vorabfilterung erlaubt. Das Filtern auf IAD passiert
> **nach** dem Download in einem separaten Skript.

## Nächste Schritte (geplant)

1. ✓ Download-Skript (dieses Skript)
2. ⬜ Filter-Skript: nur IAD-Flüge, relevante Felder
3. ⬜ Merge-Skript: alle Monate zu einer Datei zusammenfügen
4. ⬜ EDA & Feature Engineering
5. ⬜ Train/Val-Split (Temporal: 2025 = Train, 2026 Q1-Q2 = Val)
6. ⬜ Modell-Training & Evaluation
