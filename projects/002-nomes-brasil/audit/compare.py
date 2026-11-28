"""Compare every audit processed and results table against the project outputs.

Compares audit/output/processed/ with data/processed/ and every table in
audit/output/results/ (mapped through results_renames.json) with results/.
Exits with status 1 on any disagreement, any uncovered results file, or any
results column missing from the audit table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = AUDIT_DIR.parent
AUDIT_PROCESSED_DIR = AUDIT_DIR / "output" / "processed"
AUDIT_RESULTS_DIR = AUDIT_DIR / "output" / "results"
REPO_PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPO_RESULTS_DIR = PROJECT_DIR / "results"
RENAMES_PATH = PROJECT_DIR / "results_renames.json"

FLOAT_RTOL = 1e-9
DISCRETE_FLOAT_KEYS = frozenset({"threshold", "min_share"})


def normalise_non_float(series: pd.Series[Any]) -> pd.Series[Any]:
    """Normalise integer and text columns so width or backend differences match."""
    if pd.api.types.is_integer_dtype(series):
        return series.astype("int64")
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        return series.astype("string")
    return series


def compare_frames(
    audit: pd.DataFrame, repo: pd.DataFrame, *, exact_columns: bool
) -> str | None:
    """Compare two DataFrames by value; return a failure reason or an ok note/None."""
    missing_in_audit = [col for col in repo.columns if col not in audit.columns]
    if missing_in_audit:
        return f"audit lacks columns {missing_in_audit}"
    if exact_columns:
        missing_in_repo = [col for col in audit.columns if col not in repo.columns]
        if missing_in_repo:
            return f"results lacks columns {missing_in_repo}"
    if len(audit) != len(repo):
        return f"{len(audit)} rows against {len(repo)}"

    cols = list(repo.columns)
    audit = audit[cols].copy()
    repo = repo[cols].copy()

    floats = [c for c in cols if pd.api.types.is_float_dtype(repo[c])]
    non_floats = [c for c in cols if c not in floats]
    for col in non_floats:
        audit[col] = normalise_non_float(audit[col])
        repo[col] = normalise_non_float(repo[col])

    sort_keys = [
        c for c in cols if c in non_floats or c in DISCRETE_FLOAT_KEYS
    ]
    audit = audit.sort_values(sort_keys, kind="stable", ignore_index=True)
    repo = repo.sort_values(sort_keys, kind="stable", ignore_index=True)

    for col in non_floats:
        if not audit[col].equals(repo[col]):
            return f"column {col} differs"

    worst = 0.0
    for col in floats:
        a = audit[col].to_numpy(dtype=np.float64)
        r = repo[col].to_numpy(dtype=np.float64)
        if not np.allclose(a, r, rtol=FLOAT_RTOL, atol=0.0, equal_nan=True):
            return f"column {col} differs beyond {FLOAT_RTOL:g}"
        finite = np.isfinite(r) & (r != 0)
        if finite.any():
            worst = max(worst, float(np.max(np.abs(a[finite] / r[finite] - 1))))
    return None if worst == 0.0 else f"ok, floats within {worst:.1e}"


def report_outcome(label: str, problem: str | None) -> bool:
    """Print one table line and return True if the comparison failed."""
    failed = problem is not None and not problem.startswith("ok")
    status = "FAIL" if failed else "same"
    print(f"{status}  {label}  {problem or ''}".rstrip())
    return failed


def main() -> None:
    renames: dict[str, Any] = (
        json.loads(RENAMES_PATH.read_text(encoding="utf-8"))
        if RENAMES_PATH.exists()
        else {}
    )
    file_renames: dict[str, str] = renames.get("files", {})
    column_renames: dict[str, dict[str, str]] = renames.get("columns", {})
    value_renames: dict[str, dict[str, dict[str, str]]] = renames.get("values", {})

    failures = 0
    total = 0

    for repo_path in sorted(REPO_PROCESSED_DIR.glob("*.parquet")):
        total += 1
        audit_path = AUDIT_PROCESSED_DIR / repo_path.name
        if not audit_path.exists():
            failures += report_outcome(
                f"data/processed/{repo_path.name}", "missing in audit"
            )
            continue
        audit_df = pd.read_parquet(audit_path).rename(columns={"api_rank": "rank"})
        repo_df = pd.read_parquet(repo_path)
        problem = compare_frames(audit_df, repo_df, exact_columns=False)
        failures += report_outcome(f"data/processed/{repo_path.name}", problem)

    audit_by_repo_name: dict[str, tuple[str, Path]] = {}
    for audit_path in sorted(AUDIT_RESULTS_DIR.glob("*.csv")):
        mapped = file_renames.get(audit_path.name, audit_path.name)
        audit_by_repo_name[mapped] = (audit_path.name, audit_path)

    repo_names = {p.name for p in REPO_RESULTS_DIR.glob("*.csv")}
    for name in sorted(repo_names | audit_by_repo_name.keys()):
        total += 1
        repo_path = REPO_RESULTS_DIR / name
        if name not in audit_by_repo_name:
            failures += report_outcome(f"results/{name}", "missing in audit")
            continue
        if not repo_path.exists():
            failures += report_outcome(f"results/{name}", "missing in results")
            continue
        old_name, audit_path = audit_by_repo_name[name]
        audit_df = pd.read_csv(audit_path).rename(
            columns=column_renames.get(old_name, {})
        )
        for col, mapping in value_renames.get(old_name, {}).items():
            audit_df[col] = audit_df[col].replace(mapping)
        repo_df = pd.read_csv(repo_path)
        problem = compare_frames(audit_df, repo_df, exact_columns=True)
        failures += report_outcome(f"results/{name}", problem)

    print(f"{failures} failure(s) across {total} tables")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
