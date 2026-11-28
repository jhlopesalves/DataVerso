# Independent audit

These scripts recompute every table in `data/processed/` and `results/` from IBGE's raw responses in `data/raw/`, without importing anything from `build.py` or the notebook. `compare.py` then checks the two sets of tables against each other.

`recompute_tables.py` and the first version of `recompute_class_tables.py` were written before reading the notebook. Reading it changed two things in `recompute_class_tables.py`: the tables' layout, and the life-cycle features, which now use the sample standard deviation. Everything else was written afterwards: `recompute_robustness.py`, `life_cycle_robustness.csv` in `recompute_class_tables.py`, and `recompute_after_notebook.py`.

Run from `projects/002-nomes-brasil/`, in this order:

```bash
uv run python audit/recompute_tables.py
uv run python audit/recompute_class_tables.py
uv run python audit/recompute_robustness.py
uv run python audit/recompute_after_notebook.py
uv run python audit/compare.py
```

`compare.py` exits with status 1 if any table disagrees, or if any table or column in `results/` has no counterpart in the audit.
