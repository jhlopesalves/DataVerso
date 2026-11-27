Independent audit of DataVerso 002 ([AUDIT.md](AUDIT.md)) — recomputes results from `data/raw/`, compares them with `results/` and runs sensitivity checks. Requires `data/raw/`; run in order from the project directory:
```bash
uv run python audit/part1_recompute.py
uv run python audit/part2_compare.py
uv run python audit/part3_sensitivity.py
uv run python audit/part5_recompute.py reconciled
uv run python audit/part5_compare.py
```
