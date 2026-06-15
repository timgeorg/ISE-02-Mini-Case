"""Smoke-Test: fuehrt die Code-Zellen des Baseline-Notebooks als Skript aus.

Damit pruefen wir, dass alle Imports + Sklearn-Calls korrekt funktionieren,
bevor das Notebook im Jupyter-Editor geoeffnet wird.
"""
import json
import sys
from pathlib import Path

NB = Path("notebooks/01_baseline_logreg.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"Notebook: {NB.name} -> {len(code_cells)} Code-Zellen")

PROJECT = Path.cwd()
RESULTS = PROJECT / "results"
MODELS = PROJECT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

# Fuehre jede Code-Zelle im shared namespace aus
namespace = {"__name__": "__main__"}
for i, c in enumerate(code_cells, 1):
    src = "".join(c["source"])
    # Jupyter-Magic ueberspringen
    src_lines = [l for l in src.split("\n") if not l.strip().startswith("%")]
    src_clean = "\n".join(src_lines)
    print(f"\n--- Code-Zelle {i}/{len(code_cells)} ---")
    try:
        exec(src_clean, namespace)
    except Exception as e:
        print(f"FEHLER in Zelle {i}: {e}")
        # Nicht abbrechen, damit wir sehen, welche Zellen laufen
print("\n=== Smoke-Test fertig ===")
