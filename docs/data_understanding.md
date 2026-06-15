# CRISP-DM Phase 1 & 2: Business- und Data-Understanding

_Erstellt am 2026-06-15 16:56_

---

## Phase 1 – Business Understanding

### 1.1 Ausgangslage (Aufgabenstellung)

> **CASE 12: VORHERSAGE VON FLUGVERSPÄTUNGEN**
>
> - **Business Problem**: Verspätungen frühzeitig erkennen.
> - **ML-Typ**: Klassifikation
> - **Datensatz**: Airline Delay Dataset
> - **Business Mehrwert**: Bessere Ressourcenplanung.

### 1.2 Data-Science-Ziel

Binäre Klassifikation: Wird ein Flug von IAD **≥ 15 Minuten verspätet** abheben
(gemäß BTS-Standarddefinition), oder startet er pünktlich / verfrüht.

**Target-Variable (vorzuschlagen):**
```
delay_label = 1  falls departure_delay_min >= 15
             0  sonst (pünktlich ODER verfrüht, oder NaN/cancelled/diverted)
```

### 1.3 Vorgeschlagene Erfolgsmetriken

| Metrik | Begründung |
|---|---|
| **PR-AUC (Precision-Recall)** | Verspätungen sind das seltenere Ereignis, PR-AUC ist aussagekräftiger als ROC-AUC. |
| **F1-Score (Klasse 1)** | Ausbalanciertes Maß für Precision und Recall. |
| **Recall@Precision≥0.7** | Business-relevant: Welchen Anteil der echten Verspätungen erwischen wir bei akzeptabler False-Positive-Rate? |
| **Brier Score** | Kalibrierung der Wahrscheinlichkeiten (für Ressourcenplanung wichtig). |

### 1.4 Business-Mehrwert & Handlungsempfehlung

- **Ressourcenplanung am Gate**: Bei vorhergesagter Verspätung kann Personal/Equipment umgeplant werden.
- **Passagier-Information**: Proaktive Benachrichtigungen reduzieren Entschädigungs-/Hotelkosten.
- **Crews**: Schichtpläne anpassen, um regulatorische Ruhezeiten einzuhalten.

### 1.5 Limitationen & Annahmen (selbst getroffene Entscheidungen)

| # | Entscheidung | Begründung |
|---|---|---|
| 1 | **Carrier = United Airlines (UA) only** | Die heruntergeladene `Departures.csv` enthält ausschließlich UA-Flüge (vermutlich Auswahl im Web-UI). Wir behandeln das als Feature, nicht als Bug. |
| 2 | **Zeitraum = 2025-01 bis 2026-04** | `Departures.csv` reicht bis 30.04.2026; damit haben wir 16 Monate (≈ 47.000 Flüge). 2025 dient als Trainingsbasis, 2026 als Validation. |
| 3 | **Verfrühte Flüge = pünktlich (Klasse 0)** | BTS-Konvention; -5 Min ist operativ eine Punktlandung. |
| 4 | **Stornierte/Umgeleitete Flüge = Ausschluss** | Sie haben keine sinnvolle Verspätungs-Target; werden separat über Cancellation/Diversion-Tabellen analysiert. |

---

## Phase 2 – Data Understanding

### 2.1 Datenquellen & Dateigrößen

| Datei | Größe | Zeilen | Spalten |
|---|---|---|---|
| `data/raw/Detailed_Statistics_Arrivals.csv` | 4,903.5 KB | – | – |
| `data/raw/Detailed_Statistics_Cancellation.csv` | 12.5 KB | – | – |
| `data/raw/Detailed_Statistics_Departures.csv` | 4,925.5 KB | – | – |
| `data/raw/Detailed_Statistics_Diversion.csv` | 5.3 KB | – | – |

### 2.2 Departure-Tabelle – Hauptmetriken

- **Zeilen gesamt:** 47,217
- **Zeitraum:** 2025-01-01 → 2026-04-30
- **Carrier (Unique):** ['UA']
- **Zielflughäfen (Unique):** 63

**Top-10 Destinationen (von IAD aus):**

| Rang | IATA-Code | Anzahl Flüge |
|---:|---|---:|
| 1 | DEN | 3,553 |
| 2 | SFO | 3,425 |
| 3 | LAX | 2,974 |
| 4 | MCO | 2,653 |
| 5 | ORD | 2,561 |
| 6 | IAH | 2,461 |
| 7 | BOS | 2,026 |
| 8 | EWR | 1,865 |
| 9 | TPA | 1,682 |
| 10 | LAS | 1,486 |

**Verspätungs-Verteilung (Minuten, alle Flüge):**

| Kennzahl | Wert |
|---|---|
| Mittelwert | 10.08 |
| Median | -3.00 |
| Std.-Abw. | 46.25 |
| Min | -24.00 |
| Max | 1128.00 |

**Klassen-Verteilung (Delay ≥ 15 Min):**

| Klasse | Bedeutung | Anzahl | Anteil |
|---:|---|---:|---:|
| 1 | Verspätet (≥ 15 Min) | 7,402 | 15.7 % |
| 0 | Pünktlich (< 15 Min) | 39,815 | 84.3 % |
| (negativ) | Davon verfrüht (< 0 Min) | 30,414 | 64.4 % der Gesamtflüge |
| NaN | Missing (storniert/diverted?) | 0 | 0.0 % |

> ⚠️ **Klassen-Imbalance:** 15.7 % verspätet vs. 84.3 % pünktlich. Modell muss Stratifizierung & Class-Weights berücksichtigen.

### 2.3 Datenqualität – Auffälligkeiten

| Problem | Wert | Implikation |
|---|---|---|
| `tail_number` Missing | 330 / 47,217 (0.7 %) | Feature nicht nutzbar |
| `departure_delay_min` NaN | 0 (0.0 %) | Diese Zeilen für Klassifikation ausschließen |

### 2.4 Arrival-Tabelle – Sekundäranalyse (Ankunftsverspätung in IAD)

- **Zeilen gesamt:** 47,212
- **Zeitraum:** 2025-01-01 → 2026-04-30
- **Carrier:** ['UA']
- **Herkunftsflughäfen (Unique):** 61

**Top-10 Herkunftsflughäfen (nach IAD):**

| Rang | IATA-Code | Anzahl Flüge |
|---:|---|---:|
| 1 | DEN | 3,552 |
| 2 | SFO | 3,429 |
| 3 | LAX | 2,975 |
| 4 | MCO | 2,653 |
| 5 | ORD | 2,559 |
| 6 | IAH | 2,462 |
| 7 | BOS | 2,028 |
| 8 | EWR | 1,856 |
| 9 | TPA | 1,683 |
| 10 | LAS | 1,484 |

**Verspätungs-Verteilung bei Ankunft (Minuten):**

| Kennzahl | Wert |
|---|---|
| Mittelwert | 2.66 |
| Median | -8.00 |
| Std.-Abw. | 48.32 |
| Min | -68.00 |
| Max | 1389.00 |

**Klassen-Verteilung Arrival Delay (≥ 15 Min):**

| Klasse | Bedeutung | Anzahl | Anteil |
|---:|---|---:|---:|
| 1 | Verspätete Ankunft (≥ 15 Min) | 7,652 | 16.2 % |
| 0 | Pünktliche Ankunft (< 15 Min) | 39,560 | 83.8 % |
| (negativ) | Davon verfrüht (< 0 Min) | 31,362 | 66.4 % der Gesamtflüge |

> **Verwendung im Case:**
>
> - **Hauptaufgabe bleibt Abflugverspätung** (Departures). Arrivals können als **zusätzliches Feature** dienen: `avg_arrival_delay_yesterday` oder `inbound_delay_likely` (ein Flugzeug, das verspätet ankommt, hat ein höheres Risiko, verspätet abzufliegen).
> - Aktuell fehlt aber das Join-Schlüssel-Feld (Tail Number ist in Departures zu 99 % leer) → diese Verknüpfung ist in unseren Daten **nicht direkt möglich**.
> - Alternative: Rollende Mittel der historischen Ankunftsverspätung pro `(origin_airport, hour_of_day, dayofweek)` als Feature.

### 2.5 Cancellation & Diversion – Ergänzende Tabellen

| Tabelle | n | Zeitraum | Carrier | Fehlende Tail-Numbers |
|---|---:|---|---|---|
| Cancellation | 356 | 2025-01-05 → 2026-04-19 | ['UA'] | 329 (92.4 %) |
| Diversion | 121 | 2025-01-02 → 2026-04-29 | ['UA'] | 0 (0.0 %) |

**Beobachtung:** Cancellation & Diversion sind **sehr seltene Ereignisse** (< 0.5 % der Flüge).
Sie fließen als **zusätzliche Features** ein (z. B. „Carrier hatte gestern Cancellation an diesem Flughafen“),
aber **nicht in die Target-Variable** der Hauptaufgabe.

### 2.6 Schema – Spalten-Definitionen (Departures)

| Spalte | Typ | Bedeutung | Nutzung in Modellierung |
|---|---|---|---|
| `carrier_code` | str (UA) | Carrier-Code | Kategorisch (konstant) |
| `date` | date | Flugdatum | Feature → Wochentag, Monat, Feiertag |
| `flight_number` | int | Flugnummer (UA-intern) | Optional (viele unique) |
| `tail_number` | str | Flugzeug-Kennung | ❌ 99.9 % Missing → nicht nutzbar |
| `dest_airport` | str (IATA) | Ziel-Flughafen | Feature (One-Hot / Embedding) |
| `sched_dep_time` | str (HH:MM) | Geplante Abflugzeit (lokal) | Feature → Stunde, Tageszeit-Bucket |
| `actual_dep_time` | str (HH:MM) | Tatsächliche Abflugzeit | **Nicht als Feature** (post-hoc Info) |
| `sched_elapsed_min` | int | Geplante Flugdauer | Feature (proxy für Distanz) |
| `actual_elapsed_min` | int | Tatsächliche Flugdauer | **Nicht als Feature** |
| `departure_delay_min` | int | **TARGET** (Verspätung) | y |
| `wheels_off_time` | str | Rollzeit-Ende | **Nicht als Feature** |
| `taxi_out_min` | int | Rollzeit zum Start | ⚠️ Borderline – wird *vor* tatsächlichem Start gemessen, ggf. Feature |
| `delay_carrier_min` | int | Verspätung Airline (offiziell zugeordnet) | ⚠️ Leakage-Risiko! Wird von BTS erst *nach* dem Ereignis attributiert |
| `delay_weather_min` | int | Verspätung Wetter | ⚠️ Ebenfalls nachträglich attributiert |
| `delay_nas_min` | int | Verspätung National Aviation System | ⚠️ Ebenfalls nachträglich |
| `delay_security_min` | int | Verspätung Sicherheit | ⚠️ Ebenfalls nachträglich |
| `delay_late_aircraft_min` | int | Verspätung durch Vorgänger-Flugzeug | ⚠️ Ebenfalls nachträglich |

### 2.7 Feature-Kandidaten (Finale Vorauswahl)

**Zugelassene Features (pre-departure, kein Leakage):**
1. `date` → abgeleitet: `dayofweek`, `month`, `day`, `is_weekend`, `is_holiday` (US-Feiertage)
2. `sched_dep_time` → `hour_of_day`, `minute_of_hour`, `time_block`
3. `dest_airport` (One-Hot-Encoding, ~100 unique)
4. `sched_elapsed_min` (proxy für Distanz / Kurz- vs. Langstrecke)
5. `flight_number` (optional: Frequenz-Count-Encoding)

**Ausgeschlossene Features (Leakage / post-hoc):**
- `actual_dep_time`, `actual_elapsed_min`, `wheels_off_time`
- `taxi_out_min` (zwar pre-departure gemessen, aber **nicht** vorhersagbar)
- Alle 5 `delay_*_min` (von BTS nachträglich attribuiert → Leakage)
- `tail_number` (99.9 % Missing)

**Externe Features (Phase 3, falls Zeit reicht):**
- Wetter (IAD METAR, Schneefall, Gewitter)
- Vorgänger-Flugzeug-Verspätung (über `tail_number` nicht möglich → verwerfen)
- Feiertage in Zieldestination

### 2.8 Daten-Hypothesen für Phase 3 (Data Preparation)

1. **Saisonalität**: Verspätungen sind im Winter (Schnee) und Sommer (Gewitter) höher.
2. **Tageszeit**: Frühe Flüge und Stoßzeiten am Abend sind verspätungsanfälliger.
3. **Streckeneffekt**: Langstreckenflüge haben tendenziell höhere durchschnittliche Verspätungen.
4. **Klassen-Imbalance**: ~15–20 % Verspätungen → `class_weight='balanced'` oder SMOTE.
5. **NaN-Strategie**: `departure_delay_min` NaN → Drop (nur 0.1 % der Daten).

---

## Nächste Schritte (CRISP-DM Phase 3: Data Preparation)

1. Saubere Pipeline zum Einlesen aller 3 Tabellen (mit korrekten Headern).
2. Feature-Engineering: Datums-Features, Strecken-Features, Holiday-Flags.
3. Train/Validation-Split: 2025 ganzjährig → temporal 80/20 (Q1+Q2 vs. Q3+Q4) oder stratifiziert.
4. Baseline-Modell: Logistic Regression → F1 ~0.30–0.40 erwartbar.
5. Iteration: Random Forest → XGBoost/LightGBM → Hyperparameter-Tuning.
