"""Sensitivity analyses and methodological diagnostics for DataVerso 002.

Reads the rebuilt base tables from output/tables/ and Part 1 outputs from output/part1/,
and writes all Part 3 sensitivity tables to output/part3/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from part1_recompute import (
    DECADES,
    PART1_DIR,
    SPELLING_PATTERNS,
    TABLES_DIR,
    classify_names_by_sex,
)

AUDIT_DIR = Path(__file__).resolve().parent
PART3_DIR = AUDIT_DIR / "output" / "part3"


def sensitivity_1_suppressed_people(
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

        # Baseline: listed people only.
        p_base = counts / l_d
        q1_base = float(np.exp(-np.sum(p_base * np.log(p_base))))
        q2_base = float(1.0 / np.sum(p_base * p_base))

        # Assumption (a): unseen people hold names of 19 bearers each (plus at most one
        # remainder group of 1..18 bearers when m_d is not divisible by 19).
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
        q1_a = float(np.exp(ent_a))
        q2_a = float(1.0 / simp_a)

        # Assumption (b): unseen people hold names of 1 bearer each (m_d singletons).
        p1 = 1.0 / c_d
        ent_b = ent_listed - m_d * (p1 * np.log(p1))
        simp_b = simp_listed + m_d * (p1 * p1)
        q1_b = float(np.exp(ent_b))
        q2_b = float(1.0 / simp_b)

        rows.append(
            {
                "decade": dec,
                "listed_people": l_d,
                "census_people_plain": c_d,
                "unseen_people_m": m_d,
                "unseen_share": float(m_d / c_d),
                "hill_q1_base": q1_base,
                "hill_q1_19each": q1_a,
                "hill_q1_1each": q1_b,
                "hill_q2_base": q2_base,
                "hill_q2_19each": q2_a,
                "hill_q2_1each": q2_b,
            }
        )

    df = pd.DataFrame(rows)
    for col in (
        "hill_q1_base",
        "hill_q1_19each",
        "hill_q1_1each",
        "hill_q2_base",
        "hill_q2_19each",
        "hill_q2_1each",
    ):
        diffs = df[col].diff()
        df[f"{col}_dir"] = [
            "-" if pd.isna(v) else ("up" if v > 0 else ("down" if v < 0 else "flat"))
            for v in diffs
        ]
    return df


def sensitivity_2_beyond_top20(names_decade: pd.DataFrame) -> pd.DataFrame:
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
                "remaining_people": l_d,
                "remaining_names": len(counts),
                "top21_name": str(sub["name"].iloc[0]),
                "n50_without_top20": n50,
                "hill_q1_without_top20": q1,
                "hill_q2_without_top20": q2,
            }
        )
    return pd.DataFrame(rows)


def sensitivity_3_sex_threshold(
    names_total: pd.DataFrame, names_decade: pd.DataFrame, names_sex: pd.DataFrame
) -> pd.DataFrame:
    """Recompute spelling shares for 'any' and 'kwy' at 0.90, 0.95 and 0.99."""
    unique_names = names_total["name"].tolist()
    kwy_set = {
        n for n in unique_names if SPELLING_PATTERNS[0][1].search(n) is not None
    }
    any_set: set[str] = set()
    for _, pat_re in SPELLING_PATTERNS:
        any_set |= {n for n in unique_names if pat_re.search(n) is not None}
    pat_sets = {"kwy": kwy_set, "any": any_set}

    thresholds = (0.90, 0.95, 0.99)
    class_maps = {
        tau: classify_names_by_sex(names_total, names_sex, threshold=tau)
        for tau in thresholds
    }

    rows: list[dict[str, object]] = []
    for pat_name in ("any", "kwy"):
        matched = pat_sets[pat_name]
        for sc in ("all", "f", "m"):
            for dec in DECADES:
                df_dec = names_decade[names_decade["decade"] == dec]
                vals: dict[float, float] = {}
                for tau in thresholds:
                    if sc == "all":
                        sub = df_dec
                    else:
                        smap = class_maps[tau]
                        sub = df_dec[df_dec["name"].map(smap) == sc]
                    denom = int(sub["count"].sum())
                    num = int(sub.loc[sub["name"].isin(matched), "count"].sum())
                    vals[tau] = float(num / denom)
                rows.append(
                    {
                        "pattern": pat_name,
                        "sex_class": sc,
                        "decade": dec,
                        "share_090": vals[0.90],
                        "share_095": vals[0.95],
                        "share_099": vals[0.99],
                        "diff_090_vs_095_pp": (vals[0.90] - vals[0.95]) * 100.0,
                        "diff_099_vs_095_pp": (vals[0.99] - vals[0.95]) * 100.0,
                        "span_090_to_099_pp": abs(vals[0.99] - vals[0.90]) * 100.0,
                    }
                )
    return pd.DataFrame(rows)


def sensitivity_4_distinctiveness_threshold(
    names_decade: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute top 5 distinctive names per decade at 0.0002, 0.0005 and 0.001."""
    listed_by_dec = names_decade.groupby("decade")["count"].sum().to_dict()
    total_listed = int(sum(listed_by_dec.values()))
    pooled_counts = names_decade.groupby("name")["count"].sum().to_dict()

    thresholds = (0.0002, 0.0005, 0.001)
    top5_by_tau: dict[tuple[int, float], list[tuple[str, float, float]]] = {}

    detail_rows: list[dict[str, object]] = []
    for dec in DECADES:
        l_d = int(listed_by_dec[dec])
        l_not_d = total_listed - l_d
        sub = names_decade[names_decade["decade"] == dec].copy()
        sub["share_in_decade"] = sub["count"] / l_d
        c_not_d = sub["name"].map(pooled_counts) - sub["count"]
        sub["share_elsewhere"] = (c_not_d + 0.5) / (l_not_d + 0.5)
        sub["log_ratio"] = np.log2(sub["share_in_decade"] / sub["share_elsewhere"])

        for tau in thresholds:
            elig = (
                sub[sub["share_in_decade"] >= tau]
                .sort_values(
                    ["log_ratio", "count", "name"], ascending=[False, False, True]
                )
                .head(5)
            )
            items = [
                (str(r.name), float(r.log_ratio), float(r.share_in_decade))
                for r in elig.itertuples(index=False)
            ]
            top5_by_tau[(dec, tau)] = items
            for rank_idx, (nm, lr, sh) in enumerate(items, start=1):
                detail_rows.append(
                    {
                        "min_share": tau,
                        "decade": dec,
                        "rank": rank_idx,
                        "name": nm,
                        "log_ratio": lr,
                        "share_in_decade": sh,
                    }
                )

    change_rows: list[dict[str, object]] = []
    for dec in DECADES:
        base_names = [x[0] for x in top5_by_tau[(dec, 0.0005)]]
        low_names = [x[0] for x in top5_by_tau[(dec, 0.0002)]]
        high_names = [x[0] for x in top5_by_tau[(dec, 0.001)]]

        base_set = set(base_names)
        low_set = set(low_names)
        high_set = set(high_names)

        change_rows.append(
            {
                "decade": dec,
                "top5_00002": ", ".join(low_names),
                "top5_00005_base": ", ".join(base_names),
                "top5_0001": ", ".join(high_names),
                "enter_at_00002": ", ".join([n for n in low_names if n not in base_set])
                or "none",
                "leave_at_00002": ", ".join([n for n in base_names if n not in low_set])
                or "none",
                "enter_at_0001": ", ".join([n for n in high_names if n not in base_set])
                or "none",
                "leave_at_0001": ", ".join([n for n in base_names if n not in high_set])
                or "none",
            }
        )

    return pd.DataFrame(detail_rows), pd.DataFrame(change_rows)


def sensitivity_5_pattern_top15(
    names_total: pd.DataFrame, names_sex: pd.DataFrame
) -> pd.DataFrame:
    """List the 15 most frequent matching names overall for each spelling pattern."""
    sex_map = classify_names_by_sex(names_total, names_sex, threshold=0.95)
    total_listed = int(names_total["count"].sum())

    df = names_total.sort_values(["count", "name"], ascending=[False, True]).copy()
    df["sex_class"] = df["name"].map(sex_map)

    any_mask = np.zeros(len(df), dtype=bool)
    rows: list[dict[str, object]] = []

    for pat_name, pat_re in SPELLING_PATTERNS:
        mask = np.array([pat_re.search(n) is not None for n in df["name"]], dtype=bool)
        any_mask |= mask
        top15 = df[mask].head(15)
        for rank_idx, r in enumerate(top15.itertuples(index=False), start=1):
            rows.append(
                {
                    "pattern": pat_name,
                    "rank": rank_idx,
                    "name": str(r.name),
                    "count": int(r.count),
                    "national_share": float(r.count / total_listed),
                    "sex_class": str(r.sex_class),
                }
            )

    top15_any = df[any_mask].head(15)
    for rank_idx, r in enumerate(top15_any.itertuples(index=False), start=1):
        rows.append(
            {
                "pattern": "any",
                "rank": rank_idx,
                "name": str(r.name),
                "count": int(r.count),
                "national_share": float(r.count / total_listed),
                "sex_class": str(r.sex_class),
            }
        )

    return pd.DataFrame(rows)


def run_method_diagnostics(
    names_total: pd.DataFrame,
    names_decade: pd.DataFrame,
    names_sex: pd.DataFrame,
    census_age_sex: pd.DataFrame,
) -> None:
    """Print quantitative diagnostics supporting the Part 3 method review."""
    print("\n=== DIAGNOSTIC A: Census age-boundary hypotheses ===")
    tot_c = census_age_sex[census_age_sex["sex"] == "total"].set_index("age")[
        "population"
    ]
    listed_dec = names_decade.groupby("decade")["count"].sum()

    # Hypothesis H1 (plain reading): floor((2022 - age)/10)*10
    # (2020 = ages 0..2, 2010 = 3..12, ..., 1930 = 83..100).
    # The loop tests reference years 2021-2024 for the age-to-label mapping.
    for ref_y in (2021, 2022, 2023, 2024):
        decs = np.clip(((ref_y - tot_c.index.to_numpy()) // 10) * 10, 1930, 2020)
        c_sum = tot_c.groupby(decs).sum()
        cov = listed_dec / c_sum
        print(f"Ref year {ref_y}:")
        print(cov.round(4).to_dict())

    print("\n=== DIAGNOSTIC B: Survivorship sex ratio by decade ===")
    c_piv = census_age_sex.pivot(index="age", columns="sex", values="population")
    c_piv["decade_plain"] = np.clip(
        ((2022 - c_piv.index.to_numpy()) // 10) * 10, 1930, 2020
    )
    c_dec = c_piv.groupby("decade_plain")[["f", "m", "total"]].sum()
    c_dec["census_f_share"] = c_dec["f"] / c_dec["total"]
    print(c_dec[["f", "m", "total", "census_f_share"]].round(4))

    # Share of Maria among f-classed listed people vs all listed people:
    sex_map = classify_names_by_sex(names_total, names_sex, threshold=0.95)
    nd = names_decade.copy()
    nd["sex_class"] = nd["name"].map(sex_map)
    f_people = nd[nd["sex_class"] == "f"].groupby("decade")["count"].sum()
    m_people = nd[nd["sex_class"] == "m"].groupby("decade")["count"].sum()
    all_people = nd.groupby("decade")["count"].sum()
    maria_c = nd[nd["name"] == "maria"].set_index("decade")["count"]
    jose_c = nd[nd["name"] == "jose"].set_index("decade")["count"]
    joao_c = nd[nd["name"] == "joao"].set_index("decade")["count"]
    ana_c = nd[nd["name"] == "ana"].set_index("decade")["count"]
    head_sex_df = pd.DataFrame(
        {
            "maria_of_all": maria_c / all_people,
            "maria_of_f": maria_c / f_people,
            "ana_of_f": ana_c / f_people,
            "jose_of_all": jose_c / all_people,
            "jose_of_m": jose_c / m_people,
            "joao_of_m": joao_c / m_people,
        }
    )
    print("\nHead shares within own sex class vs all listed people:")
    print(head_sex_df.round(4))

    # Hill numbers within sex_class == 'f' and sex_class == 'm' separately:
    sex_div_rows = []
    for dec in DECADES:
        for sc in ("f", "m"):
            sub = nd[(nd["decade"] == dec) & (nd["sex_class"] == sc)].sort_values(
                ["count", "name"], ascending=[False, True]
            )
            c = sub["count"].to_numpy(dtype=np.int64)
            p = c / c.sum()
            n50 = int(np.flatnonzero(np.cumsum(c) * 2 >= c.sum())[0]) + 1
            q1 = float(np.exp(-np.sum(p * np.log(p))))
            q2 = float(1.0 / np.sum(p * p))
            sex_div_rows.append(
                {
                    "decade": dec,
                    "sex": sc,
                    "n50": n50,
                    "q1": round(q1, 1),
                    "q2": round(q2, 1),
                }
            )
    print("\nDiversity within f and m classes separately:")
    print(
        pd.DataFrame(sex_div_rows).pivot(
            index="decade", columns="sex", values=["n50", "q1", "q2"]
        )
    )

    print(
        "\n=== DIAGNOSTIC C: Distinctiveness pooled vs unweighted & suppression ==="
    )
    dist_df = pd.read_csv(PART1_DIR / "distinctive_names.csv")
    tot_map = names_total.set_index("name")["count"].to_dict()
    pooled_map = names_decade.groupby("name")["count"].sum().to_dict()
    dist_df["national_total"] = dist_df["name"].map(tot_map)
    dist_df["decade_sum"] = dist_df["name"].map(pooled_map)
    dist_df["unlisted_elsewhere"] = dist_df["national_total"] - dist_df["decade_sum"]
    print(
        "Max unlisted_elsewhere among all 200 distinctive names:",
        dist_df["unlisted_elsewhere"].max(),
    )
    cols_c = [
        "decade",
        "rank",
        "name",
        "count",
        "decade_sum",
        "national_total",
        "unlisted_elsewhere",
        "log_ratio",
    ]
    print(dist_df[dist_df["unlisted_elsewhere"] > 0][cols_c].head(15))

    # Compare max LR per decade under pooled p_{~d} vs unweighted mean across 9 decades:
    piv_counts = names_decade.pivot(
        index="name", columns="decade", values="count"
    ).fillna(0.0)
    piv_shares = piv_counts.div(listed_dec, axis=1)
    lr_comp = []
    for dec in DECADES:
        s_d = piv_shares[dec]
        elig_names = s_d[s_d >= 0.0005].index
        other_decs = [d for d in DECADES if d != dec]
        # Unweighted mean share across other 9 decades (with tiny smoothing equivalent)
        unw_other = (
            piv_shares.loc[elig_names, other_decs].sum(axis=1)
            + 0.5 / listed_dec.mean()
        ) / 9.0
        lr_unw = np.log2(s_d.loc[elig_names] / unw_other)
        lr_pooled = dist_df.loc[dist_df["decade"] == dec, "log_ratio"]
        lr_comp.append(
            {
                "decade": dec,
                "max_lr_pooled": round(float(lr_pooled.max()), 2),
                "top1_pooled": dist_df.loc[dist_df["decade"] == dec, "name"].iloc[0],
                "max_lr_unweighted": round(float(lr_unw.max()), 2),
                "top1_unweighted": str(lr_unw.idxmax()),
            }
        )
    print("\nPooled vs unweighted-decade-average Log Ratio:")
    print(pd.DataFrame(lr_comp))

    print("\n=== DIAGNOSTIC D: Top 'other' sex-class names & spelling matches ===")
    nt = names_total.copy()
    nt["sex_class"] = nt["name"].map(sex_map)
    f_map = names_sex[names_sex["sex"] == "f"].set_index("name")["count"].to_dict()
    m_map = names_sex[names_sex["sex"] == "m"].set_index("name")["count"].to_dict()
    nt["f"] = nt["name"].map(f_map).fillna(0).astype(int)
    nt["m"] = nt["name"].map(m_map).fillna(0).astype(int)
    nt["f_share"] = nt["f"] / nt["count"]
    other_top = nt[nt["sex_class"] == "other"].head(20)
    print(other_top[["name", "count", "f", "m", "f_share"]])

    # Also check distinctive names' sex classes (e.g. 'noa', 'ariel', 'eden', etc.)
    print("\nDistinctive top-20 names falling in sex_class == 'other':")
    dist_other = dist_df[dist_df["name"].map(sex_map) == "other"]
    print(dist_other[["decade", "rank", "name", "count", "log_ratio"]])


def main() -> None:
    PART3_DIR.mkdir(parents=True, exist_ok=True)

    names_total = pd.read_parquet(TABLES_DIR / "names_total.parquet")
    names_decade = pd.read_parquet(TABLES_DIR / "names_decade.parquet")
    names_sex = pd.read_parquet(TABLES_DIR / "names_sex.parquet")
    census_age_sex = pd.read_parquet(TABLES_DIR / "census_age_sex.parquet")
    coverage_df = pd.read_csv(PART1_DIR / "coverage_by_decade.csv")

    s1 = sensitivity_1_suppressed_people(names_decade, coverage_df)
    s2 = sensitivity_2_beyond_top20(names_decade)
    s3 = sensitivity_3_sex_threshold(names_total, names_decade, names_sex)
    s4_detail, s4_changes = sensitivity_4_distinctiveness_threshold(names_decade)
    s5 = sensitivity_5_pattern_top15(names_total, names_sex)

    s1.to_csv(PART3_DIR / "s1_suppressed_people.csv", index=False, encoding="utf-8")
    s2.to_csv(PART3_DIR / "s2_beyond_top20.csv", index=False, encoding="utf-8")
    s3.to_csv(PART3_DIR / "s3_sex_threshold.csv", index=False, encoding="utf-8")
    s4_detail.to_csv(
        PART3_DIR / "s4_distinctive_detail.csv", index=False, encoding="utf-8"
    )
    s4_changes.to_csv(
        PART3_DIR / "s4_distinctive_changes.csv", index=False, encoding="utf-8"
    )
    s5.to_csv(PART3_DIR / "s5_pattern_top15.csv", index=False, encoding="utf-8")

    print("=== S1: Suppressed people (1940-2000) ===")
    print(s1.to_string(index=False))

    print("\n=== S2: Concentration beyond top 20 ===")
    print(s2.to_string(index=False))

    print("\n=== S3: Sex-class threshold max shifts ===")
    print(
        s3.groupby(["pattern", "sex_class"])[
            ["diff_090_vs_095_pp", "diff_099_vs_095_pp", "span_090_to_099_pp"]
        ].agg(lambda s: float(np.max(np.abs(s))))
    )
    over_1pp = s3[
        (s3["diff_090_vs_095_pp"].abs() > 1.0)
        | (s3["diff_099_vs_095_pp"].abs() > 1.0)
        | (s3["span_090_to_099_pp"] > 1.0)
    ]
    print("Rows with > 1pp shift:", len(over_1pp))
    if len(over_1pp):
        print(over_1pp.to_string(index=False))
    else:
        print(s3[s3["sex_class"] != "all"].to_string(index=False))

    print("\n=== S4: Distinctiveness threshold changes ===")
    print(s4_changes.to_string(index=False))

    print("\n=== S5: Top 15 matching names per pattern ===")
    print(s5.to_string(index=False))

    run_method_diagnostics(names_total, names_decade, names_sex, census_age_sex)


if __name__ == "__main__":
    main()
