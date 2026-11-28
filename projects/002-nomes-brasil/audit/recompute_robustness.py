"""Independent recomputation of DataVerso 002 robustness tables.

Reads the audit's rebuilt base tables from output/processed/ and core tables from
output/results/, and writes the five robustness tables to output/results/.
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

PATTERN_LABELS = {
    "kwy": "contains_kwy",
    "th": "contains_th",
    "double_consonant": "doubled_letters",
    "son_ending": "ends_with_son",
    "ton_ending": "ends_with_ton",
    "any": "any_feature",
}


def compute_concentration_with_unlisted(
    names_decade: pd.DataFrame, coverage_df: pd.DataFrame
) -> pd.DataFrame:
    """Recompute Hill numbers for 1940-2000 under 19- and 1-bearer unseen bounds."""
    target_decades = list(range(1940, 2010, 10))
    cov_map = coverage_df.set_index("decade").to_dict(orient="index")

    rows: list[dict[str, object]] = []
    for dec in target_decades:
        counts = names_decade.loc[names_decade["decade"] == dec, "count"].to_numpy(
            dtype=np.int64
        )
        l_d = int(counts.sum())
        c_d = int(cov_map[dec]["census_people_plain"])
        m_d = c_d - l_d
        if m_d <= 0:
            raise ValueError(f"Expected positive unseen count m_d for {dec}, got {m_d}")

        n_19, rem = divmod(m_d, 19)
        p_listed_full = counts / c_d
        ent_listed = -float(np.sum(p_listed_full * np.log(p_listed_full)))
        simp_listed = float(np.sum(p_listed_full * p_listed_full))

        p19 = 19.0 / c_d
        ent_a = ent_listed - n_19 * (p19 * np.log(p19))
        simp_a = simp_listed + n_19 * (p19 * p19)
        if rem > 0:
            p_rem = rem / c_d
            ent_a -= p_rem * np.log(p_rem)
            simp_a += p_rem * p_rem
        rows.append(
            {
                "decade": dec,
                "unlisted_as": "names_of_19",
                "unlisted_people": m_d,
                "hill_q1": float(np.exp(ent_a)),
                "hill_q2": float(1.0 / simp_a),
            }
        )

        p1 = 1.0 / c_d
        ent_b = ent_listed - m_d * (p1 * np.log(p1))
        simp_b = simp_listed + m_d * (p1 * p1)
        rows.append(
            {
                "decade": dec,
                "unlisted_as": "singletons",
                "unlisted_people": m_d,
                "hill_q1": float(np.exp(ent_b)),
                "hill_q2": float(1.0 / simp_b),
            }
        )

    return pd.DataFrame(rows)


def compute_concentration_without_top_names(names_decade: pd.DataFrame) -> pd.DataFrame:
    """Recompute n50, hill_q1 and hill_q2 after removing each decade's top 20 names."""
    rows: list[dict[str, object]] = []
    for dec in DECADES:
        sub = (
            names_decade[names_decade["decade"] == dec]
            .sort_values(["count", "name"], ascending=[False, True])
            .iloc[20:]
        )
        counts = sub["count"].to_numpy(dtype=np.int64)
        l_d = int(counts.sum())
        shares = counts / l_d
        n50 = int(np.flatnonzero(np.cumsum(counts) * 2 >= l_d)[0]) + 1
        q1 = float(np.exp(-np.sum(shares * np.log(shares))))
        q2 = float(1.0 / np.sum(shares * shares))
        rows.append(
            {
                "decade": dec,
                "first_kept_name": str(sub["name"].iloc[0]),
                "listed_people": l_d,
                "listed_names": len(counts),
                "n50": n50,
                "hill_q1": q1,
                "hill_q2": q2,
            }
        )
    return pd.DataFrame(rows)


def compute_spelling_threshold_check(
    names_total: pd.DataFrame, names_decade: pd.DataFrame, names_sex: pd.DataFrame
) -> pd.DataFrame:
    """Recompute spelling shares for all six features at 0.90, 0.95 and 0.99."""
    unique_names = names_total["name"].tolist()
    pat_sets: dict[str, set[str]] = {}
    any_set: set[str] = set()
    for pat_name, pat_re in SPELLING_PATTERNS:
        matched = {n for n in unique_names if pat_re.search(n) is not None}
        pat_sets[PATTERN_LABELS[pat_name]] = matched
        any_set |= matched
    pat_sets[PATTERN_LABELS["any"]] = any_set

    thresholds = (0.90, 0.95, 0.99)
    class_maps = {
        tau: classify_names_by_sex(names_total, names_sex, threshold=tau)
        for tau in thresholds
    }

    rows: list[dict[str, object]] = []
    for tau in thresholds:
        smap = class_maps[tau]
        df = names_decade.copy()
        df["sex_class"] = df["name"].map(smap)
        for dec in DECADES:
            df_dec = df[df["decade"] == dec]
            subsets = {sc: df_dec[df_dec["sex_class"] == sc] for sc in ("f", "m")}
            denoms = {sc: int(sub["count"].sum()) for sc, sub in subsets.items()}
            for pat_label in sorted(pat_sets):
                matched = pat_sets[pat_label]
                for sc in ("f", "m"):
                    sub = subsets[sc]
                    num = int(sub.loc[sub["name"].isin(matched), "count"].sum())
                    rows.append(
                        {
                            "threshold": tau,
                            "decade": dec,
                            "pattern": pat_label,
                            "sex_class": sc,
                            "share": float(num / denoms[sc]),
                        }
                    )
    return pd.DataFrame(rows)


def compute_distinctive_floor_check(names_decade: pd.DataFrame) -> pd.DataFrame:
    """Recompute top 5 distinctive names per decade at 0.0002, 0.0005 and 0.001."""
    listed_by_dec = names_decade.groupby("decade")["count"].sum().to_dict()
    total_listed = int(sum(listed_by_dec.values()))
    pooled_counts = names_decade.groupby("name")["count"].sum().to_dict()

    thresholds = (0.0002, 0.0005, 0.001)
    rows: list[dict[str, object]] = []
    for tau in thresholds:
        for dec in DECADES:
            l_d = int(listed_by_dec[dec])
            l_not_d = total_listed - l_d
            sub = names_decade[names_decade["decade"] == dec].copy()
            sub["share_in_decade"] = sub["count"] / l_d
            c_not_d = sub["name"].map(pooled_counts) - sub["count"]
            sub["share_elsewhere"] = (c_not_d + 0.5) / (l_not_d + 0.5)
            sub["log_ratio"] = np.log2(sub["share_in_decade"] / sub["share_elsewhere"])

            elig = (
                sub[sub["share_in_decade"] >= tau]
                .sort_values(
                    ["log_ratio", "count", "name"], ascending=[False, False, True]
                )
                .head(5)
            )
            for rank_idx, (nm, lr) in enumerate(
                zip(elig["name"], elig["log_ratio"], strict=True), start=1
            ):
                rows.append(
                    {
                        "min_share": tau,
                        "decade": dec,
                        "rank": rank_idx,
                        "name": str(nm),
                        "log_ratio": float(lr),
                    }
                )
    return pd.DataFrame(rows)


def compute_distinctive_reference_check(
    names_decade: pd.DataFrame, distinctive_df: pd.DataFrame
) -> pd.DataFrame:
    """Compare pooled vs unweighted-decade-average Log Ratio top 20 per decade."""
    listed_dec = names_decade.groupby("decade")["count"].sum()
    piv_counts = names_decade.pivot(
        index="name", columns="decade", values="count"
    ).fillna(0.0)
    piv_shares = piv_counts.div(listed_dec, axis=1)

    rows: list[dict[str, object]] = []
    for dec in DECADES:
        pooled_sub = distinctive_df[distinctive_df["decade"] == dec]
        for rank_val, name_val, lr_val in zip(
            pooled_sub["rank"],
            pooled_sub["name"],
            pooled_sub["log_ratio"],
            strict=True,
        ):
            rows.append(
                {
                    "decade": dec,
                    "reference": "pooled",
                    "rank": int(rank_val),
                    "name": str(name_val),
                    "log_ratio": float(lr_val),
                }
            )

        s_d = piv_shares[dec]
        elig_names = s_d[s_d >= 0.0005].index
        other_decs = [d for d in DECADES if d != dec]
        unw_other = (
            piv_shares.loc[elig_names, other_decs].sum(axis=1)
            + 0.5 / float(listed_dec.mean())
        ) / 9.0
        avg_df = pd.DataFrame(
            {
                "name": elig_names.astype(str),
                "count": piv_counts.loc[elig_names, dec].to_numpy(dtype=np.int64),
                "log_ratio": np.log2(
                    s_d.loc[elig_names].to_numpy(dtype=np.float64)
                    / unw_other.to_numpy(dtype=np.float64)
                ),
            }
        )
        top_avg = avg_df.sort_values(
            ["log_ratio", "count", "name"], ascending=[False, False, True]
        ).head(20)
        for rank_idx, (nm, lr) in enumerate(
            zip(top_avg["name"], top_avg["log_ratio"], strict=True), start=1
        ):
            rows.append(
                {
                    "decade": dec,
                    "reference": "averaged",
                    "rank": rank_idx,
                    "name": str(nm),
                    "log_ratio": float(lr),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["decade", "reference", "rank"], ignore_index=True
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    names_total = pd.read_parquet(PROCESSED_DIR / "names_total.parquet")
    names_decade = pd.read_parquet(PROCESSED_DIR / "names_decade.parquet")
    names_sex = pd.read_parquet(PROCESSED_DIR / "names_sex.parquet")
    coverage_df = pd.read_csv(RESULTS_DIR / "coverage_by_decade.csv")
    distinctive_df = pd.read_csv(RESULTS_DIR / "distinctive_names.csv")

    outputs = {
        "concentration_with_unlisted.csv": compute_concentration_with_unlisted(
            names_decade, coverage_df
        ),
        "concentration_without_top_names.csv": compute_concentration_without_top_names(
            names_decade
        ),
        "distinctive_floor_check.csv": compute_distinctive_floor_check(names_decade),
        "distinctive_reference_check.csv": compute_distinctive_reference_check(
            names_decade, distinctive_df
        ),
        "spelling_threshold_check.csv": compute_spelling_threshold_check(
            names_total, names_decade, names_sex
        ),
    }

    for filename, frame in outputs.items():
        out_path = RESULTS_DIR / filename
        frame.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Wrote {out_path} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
