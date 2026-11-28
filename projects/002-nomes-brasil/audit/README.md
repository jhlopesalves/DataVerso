# Independent audit

Recomputes the processed and results tables of DataVerso 002 directly from `data/raw/`, sharing no code with `build.py` or `dataverso_002.ipynb`, and checks every table and column against `data/processed/` and `results/`.
Written before reading the notebook: `recompute_tables.py`, the `blind` phase of `recompute_class_tables.py`, and the checks in `recompute_robustness.py`; written after reading the notebook: the `reconciled` phase and `life_cycle_robustness.csv`, the sex-class check's extension to all six features, diagnostic C's table, and `recompute_after_notebook.py`.
`compare.py` exits with status 1 on any disagreement or on any `results/` table or column that the audit does not cover; run from `projects/002-nomes-brasil/` with `data/raw/` present:
```bash
uv run python audit/recompute_tables.py
uv run python audit/recompute_class_tables.py reconciled
uv run python audit/recompute_robustness.py
uv run python audit/recompute_after_notebook.py
uv run python audit/compare.py
```
