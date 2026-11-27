"""Cell-by-cell comparison between Part 1 independent tables and repository results.

Compares output/part1/ against ../results/ using exact equality for integer and string
columns and 1e-9 relative tolerance for floating-point columns. Exits with status 1 on
any disagreement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent
PART1_DIR = AUDIT_DIR / "output" / "part1"
REPO_RESULTS_DIR = AUDIT_DIR.parent / "results"

TABLE_SPECS: dict[str, dict[str, list[str]]] = {
    "diversity_by_decade.csv": {
        "keys": ["decade", "set"],
        "ints": ["decade", "listed_people", "listed_names", "n50"],
        "strs": ["set", "top1_name"],
        "floats": [
            "top1_share",
            "top10_share",
            "top100_share",
            "hill_q1",
            "hill_q2",
        ],
    },
    "coverage_by_decade.csv": {
        "keys": ["decade"],
        "ints": ["decade", "listed_people", "census_people_plain"],
        "strs": [],
        "floats": ["coverage_plain"],
    },
    "distinctive_names.csv": {
        "keys": ["decade", "rank"],
        "ints": ["decade", "count", "rank"],
        "strs": ["name"],
        "floats": ["share_in_decade", "share_elsewhere", "log_ratio"],
    },
    "heads_by_decade.csv": {
        "keys": ["decade", "name"],
        "ints": ["decade", "count"],
        "strs": ["name"],
        "floats": ["share"],
    },
    "spelling_by_decade.csv": {
        "keys": ["decade", "pattern", "sex_class"],
        "ints": ["decade", "people", "names"],
        "strs": ["pattern", "sex_class"],
        "floats": ["share"],
    },
    "sex_class_coverage.csv": {
        "keys": ["decade", "sex_class"],
        "ints": ["decade", "people"],
        "strs": ["sex_class"],
        "floats": ["share"],
    },
}


def compare_table(
    filename: str, spec: dict[str, list[str]]
) -> tuple[list[dict[str, object]], bool]:
    """Compare one CSV table by key and report whether raw row order matches."""
    df_audit = pd.read_csv(PART1_DIR / filename)
    df_repo = pd.read_csv(REPO_RESULTS_DIR / filename)

    disagreements: list[dict[str, object]] = []

    missing = [col for col in df_audit.columns if col not in df_repo.columns]
    if missing:
        disagreements.append(
            {
                "table": filename,
                "kind": "columns",
                "key": "-",
                "column": "-",
                "audit_value": list(df_audit.columns),
                "repo_value": list(df_repo.columns),
            }
        )
        return disagreements, False

    if len(df_audit) != len(df_repo):
        disagreements.append(
            {
                "table": filename,
                "kind": "row_count",
                "key": "-",
                "column": "-",
                "audit_value": len(df_audit),
                "repo_value": len(df_repo),
            }
        )

    keys = spec["keys"]
    audit_key_tuples = list(df_audit[keys].itertuples(index=False, name=None))
    repo_key_tuples = list(df_repo[keys].itertuples(index=False, name=None))
    row_order_matches = audit_key_tuples == repo_key_tuples
    if not row_order_matches and set(audit_key_tuples) != set(repo_key_tuples):
        missing_in_repo = set(audit_key_tuples) - set(repo_key_tuples)
        missing_in_audit = set(repo_key_tuples) - set(audit_key_tuples)
        disagreements.append(
            {
                "table": filename,
                "kind": "key_mismatch",
                "key": "-",
                "column": ",".join(keys),
                "audit_value": sorted(missing_in_repo)[:10],
                "repo_value": sorted(missing_in_audit)[:10],
            }
        )

    # Align on keys for cell-by-cell value checks.
    merged = pd.merge(
        df_audit,
        df_repo,
        on=keys,
        how="outer",
        suffixes=("_audit", "_repo"),
        indicator=True,
    )
    both = merged[merged["_merge"] == "both"]

    non_key_ints = [c for c in spec["ints"] if c not in keys]
    non_key_strs = [c for c in spec["strs"] if c not in keys]
    non_key_floats = [c for c in spec["floats"] if c not in keys]

    for col in non_key_ints + non_key_strs:
        va = both[f"{col}_audit"]
        vr = both[f"{col}_repo"]
        diff_mask = va != vr
        for _, row in both[diff_mask].iterrows():
            key_str = "|".join(str(row[k]) for k in keys)
            disagreements.append(
                {
                    "table": filename,
                    "kind": "exact_mismatch",
                    "key": key_str,
                    "column": col,
                    "audit_value": row[f"{col}_audit"],
                    "repo_value": row[f"{col}_repo"],
                }
            )

    for col in non_key_floats:
        va = both[f"{col}_audit"].to_numpy(dtype=np.float64)
        vr = both[f"{col}_repo"].to_numpy(dtype=np.float64)
        # Agree to relative tolerance 1e-9 (with exact 0.0 == 0.0 allowed).
        ok = np.isclose(va, vr, rtol=1e-9, atol=0.0)
        for idx in np.flatnonzero(~ok):
            row = both.iloc[idx]
            key_str = "|".join(str(row[k]) for k in keys)
            denom = max(abs(float(va[idx])), abs(float(vr[idx])), 1e-300)
            rel_err = abs(float(va[idx]) - float(vr[idx])) / denom
            disagreements.append(
                {
                    "table": filename,
                    "kind": f"float_mismatch(rel={rel_err:.3e})",
                    "key": key_str,
                    "column": col,
                    "audit_value": repr(float(va[idx])),
                    "repo_value": repr(float(vr[idx])),
                }
            )

    return disagreements, row_order_matches


def main() -> None:
    all_diffs: list[dict[str, object]] = []
    for fname, spec in TABLE_SPECS.items():
        diffs, order_ok = compare_table(fname, spec)
        order_note = "" if order_ok else " (raw row order differs)"
        print(f"=== {fname}: {len(diffs)} disagreement(s){order_note} ===")
        for d in diffs:
            print("  ", d)
        all_diffs.extend(diffs)

    print(
        f"\nTotal disagreements across all {len(TABLE_SPECS)} tables: {len(all_diffs)}"
    )
    if all_diffs:
        sys.exit(1)


if __name__ == "__main__":
    main()
