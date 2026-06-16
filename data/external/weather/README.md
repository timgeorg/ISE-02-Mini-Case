# Wetter-Snapshot (IAD, NCEI ISD-Lite)

**Datum:** 2026-06-16 09:30
**Quelle:** NCEI ISD-Lite Bulk, Station USAF `724050` / WMO `13743` (KIAD)

## Inhalt

- `raw/isd_lite_KIAD_2024.csv` (544 KB)
- `raw/isd_lite_KIAD_2025.csv` (354 KB; endet 2025-08-27, danach noch nicht publiziert)
- `iad_isd_hourly.csv` (1.27 MB, 14 497 Stundenzeilen)

## Coverage

- 2024-01-01 00:00 UTC → 2025-08-27 04:00 UTC
- 2026 noch nicht bei NCEI publiziert (Stand 2026-06-16). Skript
  überspringt das Jahr automatisch (HTTP 404 → Soft-Warning).

## Schema (offizielle ISD-Lite-Doku, 12 numerische Felder)

| # | Pos | Name | Einheit | Skalierung |
|---|---|---|---|---|
| 1-4 | 1-13 | Year/Month/Day/Hour | – | – |
| 5 | 14-19 | Air Temperature | °C | × 0.1 |
| 6 | 20-24 | Dew Point | °C | × 0.1 |
| 7 | 26-31 | Sea Level Pressure | hPa | × 0.1 |
| 8 | 32-37 | Wind Direction | deg | 1 |
| 9 | 38-43 | Wind Speed | m/s | × 0.1 |
| 10 | 44-49 | Sky Condition Total Coverage | Code 0-19 | 1 |
| 11 | 50-55 | Liquid Precip 1h | mm | × 0.1 |
| 12 | 56-61 | Liquid Precip 6h | mm | × 0.1 |

Sentinel `-9999` (fehlend) → `NaN`; Trace-Precipitation `-1` → `0.0`.

## Vereinheitlicht (`iad_isd_hourly.csv`)

| Spalte | Einheit | Quelle |
|---|---|---|
| `station` | – | konstant "KIAD" |
| `ts_utc` | UTC | aus year/month/day/hour |
| `temp_c` | °C | `temp_tC` × 0.1 |
| `dewpoint_c` | °C | `dew_tC` × 0.1 |
| `wind_kts` | Knoten | `wind_tenth_ms` × 0.1 m/s → × 1.944 |
| `wind_dir` | deg | `wind_dir_deg` (0 = calm) |
| `pressure_hpa` | hPa | `slp_tenth_hpa` × 0.1 |
| `precip_1h_mm` | mm | `precip_1h_tenth_mm` × 0.1 |
| `precip_6h_mm` | mm | `precip_6h_tenth_mm` × 0.1 |
| `sky_cov_code` | code | original |

## Plausi (Bereich)

- temp_c: −9.4 bis +39.4
- dewpoint_c: −22.8 bis +25.6
- wind_kts: 0.0 bis 28.0
- pressure_hpa: 985.4 bis 1047.4
- precip_1h_sum: 1352 mm total (≈ 9 mm/h im Mittel über nasse Stunden)
- sky_cov_code: 0 (klar) bis 9 (obscured)

## Nächster Schritt

Skript `src/04_weather_join.py` bauen: Stundenwetter → Flug-Feature
(`sched_dep_time` rounded auf volle Stunde, left-join, NaN-Forward-Fill).
