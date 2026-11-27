Independent audit for DataVerso 002: recomputes the results tables from IBGE's raw API responses, compares them with `results/`, and runs sensitivity checks.
The scripts share no code with `dataverso_002.ipynb`; the `life_cycle_robustness.csv` comparison in Part 5 was added after reading the notebook, unlike the earlier parts.
Each comparison script (`part2_compare.py` and `part5_compare.py`) exits with status 1 on any disagreement.
Requires `data/raw/`; run in order from `projects/002-nomes-brasil/`:
```bash
uv run python audit/part1_recompute.py
uv run python audit/part2_compare.py
uv run python audit/part3_sensitivity.py
uv run python audit/part5_recompute.py reconciled
uv run python audit/part5_compare.py
```
