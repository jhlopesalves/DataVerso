"""Recompute tables and columns added after reading the notebook.

Reads only from audit/output/processed/ and updates audit/output/results/:
- concentration_common_cutoff.csv
- spelling_contributors.csv
- type_share column added to spelling_by_decade.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from recompute_tables import (
    DECADES,
    PROCESSED_DIR,
    RESULTS_DIR,
    SPELLING_PATTERNS,
    classify_names_by_sex,
)

FEATURE_LABELS = {
    "kwy": "contains_kwy",
    "th": "contains_th",
    "double_consonant": "doubled_letters",
    "son_ending": "ends_with_son",
    "ton_ending": "ends_with_ton",
}


def compute_concentration_common_cutoff(names_decade: pd.DataFrame) -> pd.DataFrame:
    """Recompute Hill numbers under one common share cut-off across all decades."""
    listed_by_dec = names_decade.groupby("decade")["count"].sum().to_dict()
    cutoff = max(20.0 / int(total) for total in listed_by_dec.values())

    rows: list[dict[str, object]] = []
    for dec in DECADES:
        l_d = int(listed_by_dec[dec])
        sub = names_decade[names_decade["decade"] == dec]
        kept = sub[sub["count"] / l_d >= cutoff]
        counts = kept["count"].to_numpy(dtype=np.int64)
        kept_people = int(counts.sum())
        shares = counts / kept_people
        rows.append(
            {
                "decade": dec,
                "cutoff": cutoff,
                "names_kept": len(counts),
                "share_kept": float(kept_people / l_d),
                "hill_q1": float(np.exp(-np.sum(shares * np.log(shares)))),
                "hill_q2": float(1.0 / np.sum(shares * shares)),
            }
        )
    return pd.DataFrame(rows)


def compute_spelling_contributors(names_decade: pd.DataFrame) -> pd.DataFrame:
    """Find the five commonest matching names per decade and feature (class all)."""
    rows: list[dict[str, object]] = []
    for dec in DECADES:
        df_dec = names_decade[names_decade["decade"] == dec].sort_values(
            ["count", "name"], ascending=[False, True]
        )
        for old_pat, pat_re in SPELLING_PATTERNS:
            label = FEATURE_LABELS[old_pat]
            mask = [pat_re.search(str(n)) is not None for n in df_dec["name"]]
            hits = df_dec[mask]
            feature_people = int(hits["count"].sum())
            top5 = hits.head(5)
            for rank_idx, (nm, cnt) in enumerate(
                zip(top5["name"], top5["count"], strict=True), start=1
            ):
                bearers = int(cnt)
                rows.append(
                    {
                        "decade": dec,
                        "pattern": label,
                        "rank": rank_idx,
                        "name": str(nm),
                        "bearers": bearers,
                        "share_of_feature": float(bearers / feature_people),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["decade", "pattern", "rank"], ignore_index=True
    )


def add_type_share_to_spelling(
    names_total: pd.DataFrame, names_decade: pd.DataFrame, names_sex: pd.DataFrame
) -> pd.DataFrame:
    """Add type_share (matching names over listed names) to audit spelling table."""
    spelling_path = RESULTS_DIR / "spelling_by_decade.csv"
    spelling = pd.read_csv(spelling_path).drop(columns=["type_share"], errors="ignore")

    sex_class_map = classify_names_by_sex(names_total, names_sex, threshold=0.95)
    df = names_decade.copy()
    df["sex_class"] = df["name"].map(sex_class_map)

    listed_names: dict[tuple[int, str], int] = {}
    for dec in DECADES:
        df_dec = df[df["decade"] == dec]
        listed_names[(dec, "all")] = len(df_dec)
        listed_names[(dec, "f")] = int((df_dec["sex_class"] == "f").sum())
        listed_names[(dec, "m")] = int((df_dec["sex_class"] == "m").sum())

    denoms = [
        listed_names[(int(dec), str(sc))]
        for dec, sc in zip(spelling["decade"], spelling["sex_class"], strict=True)
    ]
    spelling["type_share"] = spelling["names"] / pd.Series(denoms, dtype="float64")
    spelling.to_csv(spelling_path, index=False, encoding="utf-8")
    return spelling


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    names_total = pd.read_parquet(PROCESSED_DIR / "names_total.parquet")
    names_decade = pd.read_parquet(PROCESSED_DIR / "names_decade.parquet")
    names_sex = pd.read_parquet(PROCESSED_DIR / "names_sex.parquet")

    common_cutoff = compute_concentration_common_cutoff(names_decade)
    common_cutoff.to_csv(
        RESULTS_DIR / "concentration_common_cutoff.csv", index=False, encoding="utf-8"
    )
    print(f"Wrote concentration_common_cutoff.csv ({len(common_cutoff)} rows)")

    contributors = compute_spelling_contributors(names_decade)
    contributors.to_csv(
        RESULTS_DIR / "spelling_contributors.csv", index=False, encoding="utf-8"
    )
    print(f"Wrote spelling_contributors.csv ({len(contributors)} rows)")

    spelling = add_type_share_to_spelling(names_total, names_decade, names_sex)
    print(f"Updated spelling_by_decade.csv with type_share ({len(spelling)} rows)")


if __name__ == "__main__":
    main()
