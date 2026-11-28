"""Independent recomputation of DataVerso 002 (Nomes no Brasil) results from raw JSON.

Reads only the raw IBGE API responses under data/raw/ and writes the six core CSVs
to output/results/, alongside intermediate rebuilt tables and query row-count checks
under output/processed/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent
RAW_DIR = AUDIT_DIR.parent / "data" / "raw"
RESULTS_DIR = AUDIT_DIR / "output" / "results"
PROCESSED_DIR = AUDIT_DIR / "output" / "processed"

DECADES = list(range(1930, 2030, 10))
HEADS = ["maria", "ana", "joao", "jose"]
HEADS_SET = set(HEADS)

SPELLING_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("kwy", re.compile(r"[kwy]")),
    ("th", re.compile(r"th")),
    ("double_consonant", re.compile(r"(bb|dd|ff|gg|ll|mm|nn|pp|tt|zz)")),
    ("son_ending", re.compile(r"son$")),
    ("ton_ending", re.compile(r"ton$")),
]


def parse_ranking_query(query_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse all paginated JSON files for a single IBGE Nomes ranking query."""
    page_files = sorted(query_dir.glob("page_*.json"))
    if not page_files:
        raise FileNotFoundError(f"No page files found in {query_dir}")

    api_counts: set[int] = set()
    api_total_pages: set[int] = set()
    seen_pages: list[int] = []
    rows: list[dict[str, object]] = []

    for page_path in page_files:
        payload = json.loads(page_path.read_text(encoding="utf-8"))
        api_counts.add(int(payload["count"]))
        api_total_pages.add(int(payload["totalPages"]))
        seen_pages.append(int(payload["page"]))
        for item in payload["items"]:
            rows.append(
                {
                    "name": str(item["nome"]),
                    "count": int(item["frequencia"]),
                    "api_rank": int(item["rank"]),
                    "api_percent": float(item["percent"]),
                }
            )

    if len(api_counts) != 1 or len(api_total_pages) != 1:
        raise ValueError(f"Inconsistent pagination metadata in {query_dir}")

    expected_count = next(iter(api_counts))
    expected_pages = next(iter(api_total_pages))
    if seen_pages != list(range(1, expected_pages + 1)):
        raise ValueError(f"Missing or out-of-order pages in {query_dir}")

    df = pd.DataFrame(rows)
    if df["name"].nunique() != len(df):
        raise ValueError(f"Duplicate names found in {query_dir}")

    # Standardise ordering: count descending, then name ascending.
    df = df.sort_values(["count", "name"], ascending=[False, True]).reset_index(
        drop=True
    )

    stats = {
        "pages_parsed": len(page_files),
        "api_total_pages": expected_pages,
        "rows_parsed": len(df),
        "api_count": expected_count,
        "listed_people": int(df["count"].sum()),
        "min_count": int(df["count"].min()),
        "max_count": int(df["count"].max()),
    }
    return df, stats


def parse_census_age_sex(census_dir: Path) -> pd.DataFrame:
    """Parse IBGE SIDRA 9514 resident population by single year of age and sex."""
    payload = json.loads((census_dir / "values.json").read_text(encoding="utf-8"))
    if len(payload) != 1 or str(payload[0]["id"]) != "93":
        raise ValueError("Unexpected variable layout in census/values.json")

    sex_map = {"Total": "total", "Homens": "m", "Mulheres": "f"}
    rows: list[dict[str, object]] = []

    for res in payload[0]["resultados"]:
        classif = {
            c["nome"]: next(iter(c["categoria"].values()))
            for c in res["classificacoes"]
        }
        forma = classif["Forma de declaração da idade"]
        if forma != "Total":
            raise ValueError(f"Unexpected age declaration category: {forma}")

        sex_label = classif["Sexo"]
        sex = sex_map[sex_label]
        age_label = classif["Idade"]

        if age_label == "Menos de 1 ano":
            age = 0
        elif age_label == "1 ano":
            age = 1
        elif age_label == "100 anos ou mais":
            age = 100
        else:
            match = re.fullmatch(r"(\d+) anos", age_label)
            if not match:
                raise ValueError(f"Unrecognised age label: {age_label!r}")
            age = int(match.group(1))

        pop_str = res["series"][0]["serie"]["2022"]
        rows.append(
            {
                "sex": sex,
                "age": age,
                "age_label": age_label,
                "population": int(pop_str),
            }
        )

    df = pd.DataFrame(rows).sort_values(["sex", "age"]).reset_index(drop=True)

    # Validate complete single-year coverage (0..100) and additive sex totals.
    for sex in ("total", "f", "m"):
        sub = df[df["sex"] == sex]
        if list(sub["age"]) != list(range(101)):
            raise ValueError(f"Incomplete age range for sex={sex}")

    piv = df.pivot(index="age", columns="sex", values="population")
    if not (piv["f"] + piv["m"] == piv["total"]).all():
        raise ValueError("Census male + female counts do not equal total")

    return df


def build_raw_tables() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Rebuild base name and census tables plus the query row-count audit."""
    ranking_dir = RAW_DIR / "ranking"
    audit_rows: list[dict[str, object]] = []

    names_total, total_stats = parse_ranking_query(ranking_dir / "total")
    audit_rows.append({"query_id": "total", **total_stats})

    decade_frames: list[pd.DataFrame] = []
    for dec in DECADES:
        qid = f"decade_{dec}"
        df_dec, dec_stats = parse_ranking_query(ranking_dir / qid)
        df_dec.insert(0, "decade", dec)
        decade_frames.append(df_dec)
        audit_rows.append({"query_id": qid, **dec_stats})
    names_decade = pd.concat(decade_frames, ignore_index=True)

    sex_frames: list[pd.DataFrame] = []
    for sex_code in ("f", "m"):
        qid = f"sex_{sex_code}"
        df_sex, sex_stats = parse_ranking_query(ranking_dir / qid)
        df_sex.insert(0, "sex", sex_code)
        sex_frames.append(df_sex)
        audit_rows.append({"query_id": qid, **sex_stats})
    names_sex = pd.concat(sex_frames, ignore_index=True)

    census_age_sex = parse_census_age_sex(RAW_DIR / "census")
    query_counts = pd.DataFrame(audit_rows)

    # Check that every name in decade and sex tables appears in names_total.
    known_names = set(names_total["name"])
    if not set(names_decade["name"]).issubset(known_names):
        raise ValueError("Found decade names absent from names_total")
    if not set(names_sex["name"]).issubset(known_names):
        raise ValueError("Found sex names absent from names_total")

    return names_total, names_decade, names_sex, census_age_sex, query_counts


def compute_diversity_by_decade(names_decade: pd.DataFrame) -> pd.DataFrame:
    """Compute Table 1: diversity_by_decade.csv."""
    rows: list[dict[str, object]] = []

    for dec in DECADES:
        df_dec = names_decade[names_decade["decade"] == dec]
        for set_name in ("all", "without_heads"):
            if set_name == "without_heads":
                sub = df_dec[~df_dec["name"].isin(HEADS_SET)]
            else:
                sub = df_dec

            # Ensure ordering: count descending, then name ascending.
            sub = sub.sort_values(["count", "name"], ascending=[False, True])
            counts = sub["count"].to_numpy(dtype=np.int64)
            listed_people = int(counts.sum())
            listed_names = len(counts)
            shares = counts / listed_people

            top1_name = str(sub["name"].iloc[0])
            top1_share = float(shares[0])
            top10_share = float(shares[:10].sum())
            top100_share = float(shares[:100].sum())

            n50_float = int(np.flatnonzero(np.cumsum(shares) >= 0.5)[0]) + 1
            n50_exact = (
                int(np.flatnonzero(np.cumsum(counts) * 2 >= listed_people)[0]) + 1
            )
            if n50_float != n50_exact:
                raise RuntimeError(
                    f"Floating-point boundary ambiguity in n50 for {dec} {set_name}: "
                    f"{n50_float} vs {n50_exact}"
                )

            hill_q1 = float(np.exp(-np.sum(shares * np.log(shares))))
            hill_q2 = float(1.0 / np.sum(shares * shares))

            rows.append(
                {
                    "decade": dec,
                    "set": set_name,
                    "listed_people": listed_people,
                    "listed_names": listed_names,
                    "top1_name": top1_name,
                    "top1_share": top1_share,
                    "top10_share": top10_share,
                    "top100_share": top100_share,
                    "n50": n50_float,
                    "hill_q1": hill_q1,
                    "hill_q2": hill_q2,
                }
            )

    return pd.DataFrame(rows)


def compute_coverage_by_decade(
    names_decade: pd.DataFrame, census_age_sex: pd.DataFrame
) -> pd.DataFrame:
    """Compute Table 2: coverage_by_decade.csv under the plain age-to-decade reading."""
    tot_census = census_age_sex[census_age_sex["sex"] == "total"].copy()
    tot_census["decade"] = (((2022 - tot_census["age"]) // 10) * 10).clip(
        lower=1930, upper=2020
    )
    census_by_dec = tot_census.groupby("decade")["population"].sum().to_dict()
    listed_by_dec = names_decade.groupby("decade")["count"].sum().to_dict()

    rows: list[dict[str, object]] = []
    for dec in DECADES:
        listed = int(listed_by_dec[dec])
        census_pop = int(census_by_dec[dec])
        rows.append(
            {
                "decade": dec,
                "listed_people": listed,
                "census_people_plain": census_pop,
                "coverage_plain": float(listed / census_pop),
            }
        )
    return pd.DataFrame(rows)


def compute_distinctive_names(names_decade: pd.DataFrame) -> pd.DataFrame:
    """Compute Table 3: distinctive_names.csv using Hardie's Log Ratio."""
    listed_by_dec = names_decade.groupby("decade")["count"].sum().to_dict()
    total_listed_all_decades = int(sum(listed_by_dec.values()))
    pooled_name_counts = names_decade.groupby("name")["count"].sum().to_dict()

    rows: list[pd.DataFrame] = []
    for dec in DECADES:
        l_d = int(listed_by_dec[dec])
        l_not_d = total_listed_all_decades - l_d
        sub = names_decade[names_decade["decade"] == dec].copy()

        c_d = sub["count"].to_numpy(dtype=np.int64)
        p_d = c_d / l_d
        mask = p_d >= 0.0005

        elig = sub.loc[mask, ["name", "count"]].copy()
        elig["share_in_decade"] = p_d[mask]

        c_total_ten = elig["name"].map(pooled_name_counts).to_numpy(dtype=np.int64)
        c_not_d = c_total_ten - elig["count"].to_numpy(dtype=np.int64)
        p_not_d = (c_not_d + 0.5) / (l_not_d + 0.5)

        elig["share_elsewhere"] = p_not_d
        elig["log_ratio"] = np.log2(elig["share_in_decade"] / elig["share_elsewhere"])

        # Rank eligible names by log_ratio desc, ties by count desc, then name asc.
        elig = (
            elig.sort_values(
                ["log_ratio", "count", "name"], ascending=[False, False, True]
            )
            .head(20)
            .reset_index(drop=True)
        )
        elig.insert(0, "decade", dec)
        elig["rank"] = np.arange(1, len(elig) + 1, dtype=np.int64)
        rows.append(
            elig[
                [
                    "decade",
                    "name",
                    "count",
                    "share_in_decade",
                    "share_elsewhere",
                    "log_ratio",
                    "rank",
                ]
            ]
        )

    return pd.concat(rows, ignore_index=True)


def compute_heads_by_decade(names_decade: pd.DataFrame) -> pd.DataFrame:
    """Compute Table 4: heads_by_decade.csv."""
    listed_by_dec = names_decade.groupby("decade")["count"].sum().to_dict()
    heads_df = names_decade[names_decade["name"].isin(HEADS_SET)]
    lookup = {
        (int(dec_val), str(name_val)): int(count_val)
        for dec_val, name_val, count_val in zip(
            heads_df["decade"], heads_df["name"], heads_df["count"], strict=True
        )
    }

    rows: list[dict[str, object]] = []
    for dec in DECADES:
        l_d = int(listed_by_dec[dec])
        dec_rows: list[dict[str, Any]] = []
        for h in HEADS:
            c = lookup.get((dec, h), 0)
            dec_rows.append(
                {
                    "decade": dec,
                    "name": h,
                    "count": c,
                    "share": float(c / l_d),
                }
            )
        # Apply general ordering within each decade: count desc, name asc.
        dec_rows.sort(key=lambda r: (-int(r["count"]), str(r["name"])))
        rows.extend(dec_rows)

    return pd.DataFrame(rows)


def classify_names_by_sex(
    names_total: pd.DataFrame, names_sex: pd.DataFrame, threshold: float = 0.95
) -> dict[str, str]:
    """Assign each name in names_total to 'f', 'm', or 'other'."""
    f_counts = names_sex[names_sex["sex"] == "f"].set_index("name")["count"].to_dict()
    m_counts = names_sex[names_sex["sex"] == "m"].set_index("name")["count"].to_dict()

    sex_class: dict[str, str] = {}
    for name_val, count_val in zip(
        names_total["name"], names_total["count"], strict=True
    ):
        name = str(name_val)
        t = int(count_val)
        f = int(f_counts.get(name, 0))
        m = int(m_counts.get(name, 0))
        if f + m > t:
            raise ValueError(f"Sex counts exceed total for {name}: f={f}, m={m}, t={t}")
        if f / t >= threshold:
            sex_class[name] = "f"
        elif m / t >= threshold:
            sex_class[name] = "m"
        else:
            sex_class[name] = "other"
    return sex_class


def compute_spelling_and_sex_coverage(
    names_total: pd.DataFrame, names_decade: pd.DataFrame, names_sex: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Table 5 (spelling_by_decade.csv) and Table 6 (sex_class_coverage.csv)."""
    sex_class_map = classify_names_by_sex(names_total, names_sex, threshold=0.95)

    # Precompute regex pattern matches over all unique names in names_total.
    unique_names = names_total["name"].tolist()
    pattern_matches: dict[str, set[str]] = {}
    any_match: set[str] = set()
    for pat_name, pat_re in SPELLING_PATTERNS:
        matched = {n for n in unique_names if pat_re.search(n) is not None}
        pattern_matches[pat_name] = matched
        any_match |= matched
    pattern_matches["any"] = any_match

    all_pattern_names = [*(p for p, _ in SPELLING_PATTERNS), "any"]

    df = names_decade.copy()
    df["sex_class"] = df["name"].map(sex_class_map)

    spelling_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for dec in DECADES:
        df_dec = df[df["decade"] == dec]
        total_dec_people = int(df_dec["count"].sum())

        # Table 6: sex_class_coverage.csv
        for sc in ("f", "m", "other"):
            sc_people = int(df_dec.loc[df_dec["sex_class"] == sc, "count"].sum())
            coverage_rows.append(
                {
                    "decade": dec,
                    "sex_class": sc,
                    "people": sc_people,
                    "share": float(sc_people / total_dec_people),
                }
            )

        # Table 5: spelling_by_decade.csv
        subsets = {
            "all": df_dec,
            "f": df_dec[df_dec["sex_class"] == "f"],
            "m": df_dec[df_dec["sex_class"] == "m"],
        }
        denom_map = {sc: int(sub["count"].sum()) for sc, sub in subsets.items()}

        for pat_name in all_pattern_names:
            matched_set = pattern_matches[pat_name]
            for sc in ("all", "f", "m"):
                sub = subsets[sc]
                denom = denom_map[sc]
                hits = sub[sub["name"].isin(matched_set)]
                people = int(hits["count"].sum())
                n_names = len(hits)
                spelling_rows.append(
                    {
                        "decade": dec,
                        "pattern": pat_name,
                        "sex_class": sc,
                        "people": people,
                        "share": float(people / denom),
                        "names": n_names,
                    }
                )

    return pd.DataFrame(spelling_rows), pd.DataFrame(coverage_rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    names_total, names_decade, names_sex, census_age_sex, query_counts = (
        build_raw_tables()
    )

    # Persist rebuilt base tables and query row-count audit.
    names_total.to_parquet(PROCESSED_DIR / "names_total.parquet", index=False)
    names_decade.to_parquet(PROCESSED_DIR / "names_decade.parquet", index=False)
    names_sex.to_parquet(PROCESSED_DIR / "names_sex.parquet", index=False)
    census_age_sex.to_parquet(PROCESSED_DIR / "census_age_sex.parquet", index=False)
    query_counts.to_csv(
        PROCESSED_DIR / "query_row_counts.csv", index=False, encoding="utf-8"
    )

    diversity_df = compute_diversity_by_decade(names_decade)
    coverage_df = compute_coverage_by_decade(names_decade, census_age_sex)
    distinctive_df = compute_distinctive_names(names_decade)
    heads_df = compute_heads_by_decade(names_decade)
    spelling_df, sex_cov_df = compute_spelling_and_sex_coverage(
        names_total, names_decade, names_sex
    )

    outputs = {
        "diversity_by_decade.csv": diversity_df,
        "coverage_by_decade.csv": coverage_df,
        "distinctive_names.csv": distinctive_df,
        "heads_by_decade.csv": heads_df,
        "spelling_by_decade.csv": spelling_df,
        "sex_class_coverage.csv": sex_cov_df,
    }

    for filename, frame in outputs.items():
        out_path = RESULTS_DIR / filename
        frame.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Wrote {out_path} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
