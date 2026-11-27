"""Compute the Part 5 audit tables exclusively from Part 1 inputs."""

import hashlib
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from part1_recompute import (
    DECADES,
    HEADS_SET,
    classify_names_by_sex,
    compute_diversity_by_decade,
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

AUDIT_DIR = Path(__file__).resolve().parent
MIN_SHARE = 0.0005
MISSING_SHARE = 1e-6
SEX_CLASSES = ["f", "m"]
GROUP_KEYS = ["decade", "sex_class"]
DIVERSITY_COLUMNS = [
    *GROUP_KEYS,
    "listed_people",
    "listed_names",
    "top1_name",
    "top1_share",
    "top10_share",
    "n50",
    "hill_q2",
]


def sex_class_tables(decade: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reuse Part 1 diversity measures within each national name class."""
    names = decade.loc[decade["sex_class"].isin(SEX_CLASSES)].copy()
    names["share_within_sex"] = names["count"] / names.groupby(GROUP_KEYS)[
        "count"
    ].transform("sum")

    parts = []
    for sex_class in SEX_CLASSES:
        diversity = compute_diversity_by_decade(
            names.loc[names["sex_class"].eq(sex_class)]
        )
        diversity.insert(1, "sex_class", sex_class)
        parts.append(diversity)
    diversity = pd.concat(parts).sort_values([*GROUP_KEYS, "set"])

    heads = names.loc[
        names["name"].isin(HEADS_SET),
        ["decade", "sex_class", "name", "count", "share_within_sex"],
    ].sort_values(["decade", "name"])

    ranked = (
        names.sort_values(
            [*GROUP_KEYS, "count", "name"], ascending=[True, True, False, True]
        )
        .groupby(GROUP_KEYS)
        .head(10)
        .copy()
    )
    ranked["rank"] = ranked.groupby(GROUP_KEYS).cumcount() + 1
    top_names = ranked[[*GROUP_KEYS, "rank", "name", "count", "share_within_sex"]]
    assert top_names.groupby(GROUP_KEYS).size().eq(10).all()
    assert len(diversity) == 40 and len(heads) == 40
    return {
        "diversity_by_sex.csv": diversity,
        "heads_by_sex.csv": heads,
        "top_names_by_sex.csv": top_names,
    }


def generation_names(decade: pd.DataFrame, totals: pd.Series) -> pd.DataFrame:
    """Rank eligible peak-decade rows using Part 1's pooled-complement ratio."""
    peaks = decade.sort_values(
        ["share_in_decade", "decade"], ascending=[False, True]
    ).drop_duplicates("name")
    peaks = peaks.loc[peaks["share_in_decade"].ge(MIN_SHARE)].copy()
    decade_people = decade.groupby("decade")["count"].sum()
    name_people = decade.groupby("name")["count"].sum()
    all_decades_for_name = peaks["name"].map(name_people)
    complement_people = decade_people.sum() - peaks["decade"].map(decade_people)
    complement_for_name = all_decades_for_name - peaks["count"]
    complement_share = (complement_for_name + 0.5) / (complement_people + 0.5)
    peaks["log_ratio"] = np.log2(peaks["share_in_decade"] / complement_share)
    peaks["peak_decade"] = peaks["decade"]
    peaks["bearers"] = peaks["name"].map(totals)
    peaks["share_born_at_peak"] = peaks["count"] / all_decades_for_name
    ranked = (
        peaks.sort_values(
            ["decade", "log_ratio", "count", "name"],
            ascending=[True, False, False, True],
        )
        .groupby("decade")
        .head(4)
        .copy()
    )
    ranked["rank"] = ranked.groupby("decade").cumcount() + 1
    assert len(ranked) == 40
    return ranked[
        [
            "decade",
            "rank",
            "name",
            "count",
            "share_in_decade",
            "log_ratio",
            "peak_decade",
            "sex_class",
            "bearers",
            "share_born_at_peak",
        ]
    ]


def life_cycles(decade: pd.DataFrame, *, ddof: int) -> pd.DataFrame:
    """Fit KMeans to eligible names ordered alphabetically by name."""
    shares = (
        decade.pivot(index="name", columns="decade", values="share_in_decade")
        .reindex(columns=DECADES)
        .sort_index()
    )
    eligible = shares.loc[shares.max(axis=1).ge(MIN_SHARE)]
    log_shares = np.log10(eligible.fillna(MISSING_SHARE).to_numpy())
    centred = log_shares - log_shares.mean(axis=1, keepdims=True)
    features = centred / log_shares.std(axis=1, ddof=ddof, keepdims=True)
    labels = KMeans(n_clusters=6, n_init=100, random_state=0).fit_predict(features)

    # The missing-share proxy is used for the fit. An unlisted decade reaches
    # the lower display limit for relative share.
    relative_share = eligible.div(eligible.max(axis=1), axis="index").fillna(0)
    relative = np.log10(relative_share.clip(lower=0.001))
    medians = relative.groupby(labels).median()
    trough = medians.loc[:, 1960:1990].min(axis=1)
    comeback = (medians[1930] - trough + medians[2020] - trough).idxmax()
    other_peaks = medians.drop(index=comeback).idxmax(axis=1)
    display_order = [*other_peaks.sort_values(kind="stable").index, comeback]
    display_group = {label: number for number, label in enumerate(display_order, 1)}

    result = relative.add_prefix("relative_")
    result.insert(0, "group", [display_group[label] for label in labels])
    assert result["group"].nunique() == 6
    assert relative.max(axis=1).eq(0).all()
    return result.reset_index()


def fit_comeback(
    features: np.ndarray, relative: pd.DataFrame, *, k: int, n_init: int, seed: int
) -> tuple[np.ndarray, set[str], int]:
    """Fit k-means and return labels, comeback member names and trough decade."""
    with threadpool_limits(limits=1):
        labels = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(
            features
        )
    medians = relative.groupby(labels).median()
    trough = medians.loc[:, 1960:1990].min(axis=1)
    comeback = int((medians[1930] - trough + medians[2020] - trough).idxmax())
    members = set(relative.index[labels == comeback])
    trough_decade = int(medians.loc[comeback].idxmin())
    return labels, members, trough_decade


def overlap_metrics(
    members: set[str], reference: set[str], trough_decade: int
) -> dict[str, int | float]:
    """Compute comeback size, overlap count, Jaccard overlap and trough decade."""
    shared = len(members & reference)
    return {
        "comeback_size": len(members),
        "overlap_count": shared,
        "overlap_jaccard": shared / len(members | reference),
        "comeback_trough_decade": trough_decade,
    }


def life_cycle_robustness(decade: pd.DataFrame) -> pd.DataFrame:
    """Compute life_cycle_robustness.csv across cluster counts and random seeds."""
    shares = (
        decade.pivot(index="name", columns="decade", values="share_in_decade")
        .reindex(columns=DECADES)
        .sort_index()
    )
    eligible = shares.loc[shares.max(axis=1).ge(MIN_SHARE)]
    log_shares = np.log10(eligible.fillna(MISSING_SHARE).to_numpy())
    centred = log_shares - log_shares.mean(axis=1, keepdims=True)
    features = centred / log_shares.std(axis=1, ddof=1, keepdims=True)
    relative_share = eligible.div(eligible.max(axis=1), axis="index").fillna(0)
    relative = np.log10(relative_share.clip(lower=0.001))

    _, main_members, _ = fit_comeback(features, relative, k=6, n_init=100, seed=0)
    rows: list[dict[str, object]] = []
    for k in range(3, 11):
        labels, members, trough_decade = fit_comeback(
            features, relative, k=k, n_init=100, seed=0
        )
        row: dict[str, object] = {
            "comparison": "resolution",
            "k": k,
            "seed": 0,
            "n_init": 100,
            "silhouette": float(silhouette_score(features, labels)),
        }
        if 5 <= k <= 8:
            row.update(overlap_metrics(members, main_members, trough_decade))
        rows.append(row)

    _, seed0_members, _ = fit_comeback(features, relative, k=6, n_init=20, seed=0)
    for seed in range(10):
        _, members, trough_decade = fit_comeback(
            features, relative, k=6, n_init=20, seed=seed
        )
        rows.append(
            {
                "comparison": "seed",
                "k": 6,
                "seed": seed,
                "n_init": 20,
                **overlap_metrics(members, seed0_members, trough_decade),
            }
        )

    return pd.DataFrame(rows).astype(
        {
            "comeback_size": "Int64",
            "overlap_count": "Int64",
            "comeback_trough_decade": "Int64",
        }
    )


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("blind", "reconciled"))
    parser.add_argument("--output", type=Path, help="Write to a new directory.")
    args = parser.parse_args()
    output = args.output or AUDIT_DIR / "output" / "part5" / args.phase
    inputs = AUDIT_DIR / "output" / "tables"
    totals = pd.read_parquet(inputs / "names_total.parquet")
    decade = pd.read_parquet(inputs / "names_decade.parquet")
    sex = pd.read_parquet(inputs / "names_sex.parquet")
    decade["sex_class"] = decade["name"].map(classify_names_by_sex(totals, sex))
    assert decade["sex_class"].notna().all()
    class_counts = decade.groupby(GROUP_KEYS)["count"].sum().sort_index()
    previous_class_counts = (
        pd.read_csv(AUDIT_DIR / "output" / "part1" / "sex_class_coverage.csv")
        .set_index(GROUP_KEYS)["people"]
        .sort_index()
    )
    pd.testing.assert_series_equal(
        class_counts, previous_class_counts, check_names=False
    )
    decade["share_in_decade"] = decade["count"] / decade.groupby("decade")[
        "count"
    ].transform("sum")

    results = sex_class_tables(decade)
    results["generation_names.csv"] = generation_names(
        decade, totals.set_index("name")["count"]
    )
    ddof = 0
    if args.phase == "reconciled":
        ddof = 1
        diversity = results["diversity_by_sex.csv"]
        results["diversity_by_sex.csv"] = diversity.loc[
            diversity["set"].eq("all"), DIVERSITY_COLUMNS
        ]
        heads = results["heads_by_sex.csv"]
        results["heads_by_sex.csv"] = heads[
            ["decade", "name", "sex_class", "count", "share_within_sex"]
        ]
    results["life_cycles.csv"] = life_cycles(decade, ddof=ddof)
    if args.phase == "reconciled":
        results["life_cycle_robustness.csv"] = life_cycle_robustness(decade)
    output.mkdir(parents=True, exist_ok=True)
    for filename, table in results.items():
        if filename != "life_cycle_robustness.csv":
            assert table.notna().all().all(), filename
        table.to_csv(output / filename, index=False)
        print(f"{filename}: {len(table)} rows, {len(table.columns)} columns")

    hashes = []
    for path in sorted(output.glob("*.csv")):
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        hashes.append(f"{digest}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
