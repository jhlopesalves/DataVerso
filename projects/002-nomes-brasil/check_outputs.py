"""Compare this project's outputs with those of a reference commit.

The published results of DataVerso 002 are frozen: a refactor may change how the
notebook computes them, never what it computes. After executing the notebook, run

    uv run python check_outputs.py --reference c919f7b

Results tables must match the reference by value: integers and text exactly, floats
to a relative tolerance of 1e-12. Figures must match pixel for pixel, and the processed
tables byte for byte. A JSON rename map lets files, columns and labels be renamed
without loosening any comparison. New files and new columns are listed but allowed.
The script exits with status 1 when anything published differs.
"""

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
FLOAT_RTOL = 1e-12


def git(*args: str) -> bytes:
    """Run a git command in the project directory and return its standard output."""
    return subprocess.run(
        ["git", *args], cwd=PROJECT_DIR, capture_output=True, check=True
    ).stdout


def reference_files(reference: str, folder: str) -> dict[str, bytes]:
    """Return the files under ``folder`` at the reference commit, keyed by name."""
    listing = git(
        "ls-tree", "-r", "--name-only", "--full-name", reference, "--", folder
    )
    return {
        Path(path).name: git("show", f"{reference}:{path}")
        for path in listing.decode().splitlines()
    }


def compare_table(
    reference_csv: bytes, current_path: Path, renames: dict[str, Any], name: str
) -> str | None:
    """Compare one results table by value; return a problem, or None if it matches."""
    expected = pd.read_csv(io.BytesIO(reference_csv))
    actual = pd.read_csv(current_path)
    expected = expected.rename(columns=renames.get("columns", {}).get(name, {}))
    for column, mapping in renames.get("values", {}).get(name, {}).items():
        expected[column] = expected[column].replace(mapping)

    missing = [column for column in expected.columns if column not in actual.columns]
    if missing:
        return f"missing columns {missing}"
    if len(actual) != len(expected):
        return f"{len(actual)} rows against {len(expected)}"

    # Align rows on every column that is not a float, so a rename that changes the
    # sort order of a label does not count as a difference.
    floats = [c for c in expected.columns if expected[c].dtype.kind == "f"]
    keys = [c for c in expected.columns if c not in floats]
    expected = expected.sort_values(keys, kind="stable", ignore_index=True)
    actual = actual[expected.columns].sort_values(
        keys, kind="stable", ignore_index=True
    )

    for column in keys:
        if not expected[column].equals(actual[column]):
            return f"column {column} differs"
    worst = 0.0
    for column in floats:
        e, a = expected[column].to_numpy(float), actual[column].to_numpy(float)
        if not np.allclose(a, e, rtol=FLOAT_RTOL, atol=0.0, equal_nan=True):
            return f"column {column} differs beyond {FLOAT_RTOL:g}"
        finite = np.isfinite(e) & (e != 0)
        if finite.any():
            worst = max(worst, float(np.max(np.abs(a[finite] / e[finite] - 1))))
    return None if worst == 0.0 else f"ok, floats within {worst:.1e}"


def compare_figure(reference_png: bytes, current_path: Path) -> str | None:
    """Compare one figure pixel for pixel and return a problem, or None if identical."""
    expected = mpimg.imread(io.BytesIO(reference_png), format="png")
    actual = mpimg.imread(current_path)
    if expected.shape != actual.shape:
        return f"shape {actual.shape} against {expected.shape}"
    changed = np.any(expected != actual, axis=-1)
    return f"{int(changed.sum())} pixels differ" if changed.any() else None


def main() -> None:
    """Compare results, figures and processed tables, and exit 1 on any difference."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", required=True, help="commit to compare with")
    parser.add_argument("--rename-map", type=Path, help="JSON map of renames")
    args = parser.parse_args()
    renames = json.loads(args.rename_map.read_text()) if args.rename_map else {}
    file_renames = renames.get("files", {})

    failures = 0
    for folder in ("results", "figures", "data/processed"):
        expected_files = reference_files(args.reference, folder)
        current_dir = PROJECT_DIR / folder
        for name, content in sorted(expected_files.items()):
            current_name = file_renames.get(name, name)
            path = current_dir / current_name
            problem: str | None
            if not path.exists():
                problem = "missing"
            elif folder == "results":
                problem = compare_table(content, path, renames, name)
            elif folder == "figures":
                problem = compare_figure(content, path)
            elif name == "manifest.json":
                # Check messages may be reworded; the data they describe may not change.
                fields = ("outputs", "raw_digest", "queries")
                old, new = json.loads(content), json.loads(path.read_bytes())
                same = all(old[field] == new[field] for field in fields)
                problem = None if same else "outputs, raw digest or queries differ"
            else:
                problem = None if path.read_bytes() == content else "bytes differ"
            failed = problem is not None and not problem.startswith("ok")
            failures += failed
            status = "FAIL" if failed else "same"
            print(f"{status}  {folder}/{current_name}  {problem or ''}".rstrip())
        expected_names = {file_renames.get(name, name) for name in expected_files}
        for path in sorted(current_dir.glob("*")):
            if path.is_file() and path.name not in expected_names:
                print(f"new   {folder}/{path.name}")

    print(f"{failures} difference(s) in published outputs")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
