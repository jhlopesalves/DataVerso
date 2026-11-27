"""Compare Part 5 with the published CSVs using Part 2's tolerance."""

import csv
import json
from argparse import ArgumentParser
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = AUDIT_DIR.parent / "results"
KEYS = {
    "diversity_by_sex.csv": ("decade", "sex_class"),
    "heads_by_sex.csv": ("decade", "name"),
    "top_names_by_sex.csv": ("decade", "sex_class", "rank"),
    "generation_names.csv": ("decade", "rank"),
    "life_cycles.csv": ("name",),
    "life_cycle_robustness.csv": ("comparison", "k", "seed"),
}
INTS = {
    "decade",
    "listed_people",
    "listed_names",
    "n50",
    "count",
    "rank",
    "peak_decade",
    "bearers",
    "group",
    "k",
    "seed",
    "n_init",
    "comeback_size",
    "overlap_count",
    "comeback_trough_decade",
}
STRINGS = {"name", "top1_name", "sex_class", "set", "comparison"}
RTOL = 1e-9


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns or len(set(columns)) != len(columns):
            raise ValueError(f"Missing or duplicate columns in {path}")
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"Wrong number of fields in {path}")
    return list(columns), rows


def row_index(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict:
    def key(row: dict[str, str]) -> tuple:
        return tuple(int(row[col]) if col in INTS else row[col] for col in keys)

    index = {key(row): row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Duplicate keys on {keys}")
    return index


def compare(path: Path, reference: Path, keys: tuple[str, ...]) -> dict:
    columns, rows = read_csv(path)
    ref_columns, ref_rows = read_csv(reference)
    audit = row_index(rows, keys)
    repo = row_index(ref_rows, keys)
    common = [col for col in ref_columns if col in columns]
    diffs = []
    max_relative = 0.0
    max_column = None
    float_nonexact = 0
    for key in sorted(audit.keys() & repo.keys()):
        for col in common:
            left, right = audit[key][col], repo[key][col]
            if left == "" or right == "" or col in STRINGS:
                same = left == right
            elif col in INTS:
                same = int(left) == int(right)
            else:
                a, b = float(left), float(right)
                absolute = abs(a - b)
                same = absolute <= RTOL * abs(b)
                relative = absolute / max(abs(a), abs(b), 1e-300)
                float_nonexact += a != b
                if relative > max_relative:
                    max_relative, max_column = relative, col
            if not same:
                diffs.append({"key": key, "column": col, "audit": left, "repo": right})
    return {
        "audit_rows": len(audit),
        "repo_rows": len(repo),
        "audit_columns": columns,
        "repo_columns": ref_columns,
        "column_order_and_names_match": columns == ref_columns,
        "row_order_matches": list(audit) == list(repo),
        "audit_only_keys": sorted(audit.keys() - repo.keys()),
        "repo_only_keys": sorted(repo.keys() - audit.keys()),
        "cells_compared": len(audit.keys() & repo.keys()) * len(common),
        "cell_disagreements": diffs,
        "nonexact_float_cells": float_nonexact,
        "max_float_relative_difference": max_relative,
        "max_float_column": max_column,
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "audit_tables",
        nargs="?",
        type=Path,
        default=AUDIT_DIR / "output" / "part5" / "reconciled",
    )
    args = parser.parse_args()
    summaries = {
        name: compare(args.audit_tables / name, RESULTS_DIR / name, keys)
        for name, keys in KEYS.items()
    }
    print(json.dumps(summaries, indent=2))
    if any(
        not result["column_order_and_names_match"]
        or result["audit_only_keys"]
        or result["repo_only_keys"]
        or result["cell_disagreements"]
        for result in summaries.values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
