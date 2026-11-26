"""Build the processed name and census tables from IBGE's raw responses.

Run from the project directory with ``uv run python build.py`` after ``collect.py``. The script
reads ``data/raw/``, checks the data against the contract below, and writes four Parquet tables
and ``manifest.json`` to ``data/processed/``. It exits with status 1 when any check fails.
"""

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from collect import (
    AGE_CLASSIFICATION,
    CENSUS_AGGREGATE,
    CENSUS_METADATA_URL,
    CENSUS_SEX,
    CENSUS_VALUES_URL,
    CENSUS_VARIABLE,
    NAME_URL,
    PROJECT_DIR,
    RANKING_QUERIES,
    RANKING_URL,
    RAW_DIR,
    RETRIEVAL_PATH,
    SEX_CLASSIFICATION,
    SPOT_CHECK_NAMES,
    age_from_label,
    census_classification,
)

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

# Resident population of Brazil on 1 August 2022. IBGE's ``percent`` field divides by this
# figure in every ranking query, which is why the processed tables drop it.
POPULATION = 203_080_756
# The national name tables omit names given to fewer than 20 people.
MINIMUM_COUNT = 20

NAME_COLUMNS = {"name": "string", "count": "int64", "rank": "int32"}
CENSUS_CELLS = {(age, sex) for age in range(101) for sex in CENSUS_SEX.values()}


@dataclass(frozen=True)
class Check:
    """The outcome of one data check, as printed and recorded in the manifest."""

    name: str
    passed: bool
    detail: str


def load_pages(query_dir: Path) -> dict[int, dict[str, Any]]:
    """Read the saved pages of one ranking query, keyed by page number in ascending order."""
    paths = sorted(query_dir.glob("page_*.json"))
    return {int(path.stem.removeprefix("page_")): json.loads(path.read_bytes()) for path in paths}


def parse_ranking(query_id: str, pages: dict[int, dict[str, Any]]) -> pd.DataFrame:
    """Turn the pages of one ranking query into one row per name, in page order.

    Parameters
    ----------
    query_id
        A key of ``RANKING_QUERIES``. The query's own parameters give the ``decade`` or
        ``sex`` column, because the pages carry only the ranked items.
    pages
        The query's pages as returned by ``load_pages``.

    Returns
    -------
    pandas.DataFrame
        Columns ``name``, ``count`` and ``rank``, preceded by ``decade`` or ``sex`` for a
        filtered query.
    """
    items = [item for page in pages.values() for item in page["items"]]
    frame = pd.DataFrame(items, columns=["nome", "frequencia", "rank"]).rename(
        columns={"nome": "name", "frequencia": "count"}
    )
    columns = NAME_COLUMNS
    params = RANKING_QUERIES[query_id]
    if "decada" in params:
        frame.insert(0, "decade", int(params["decada"]))
        columns = {"decade": "int16", **NAME_COLUMNS}
    elif "sexo" in params:
        frame.insert(0, "sex", params["sexo"])
        columns = {"sex": "string", **NAME_COLUMNS}
    return frame.astype(columns)


def parse_census(values: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn the census values response into one row per age and sex.

    SIDRA sends each value as a string and labels each cell with its categories, so the age
    comes from the ``Idade`` label and the sex from the ``Sexo`` category id.
    """
    (variable,) = values
    rows = []
    for result in variable["resultados"]:
        # Each ``categoria`` holds a single {id: label} pair.
        categories = {int(c["id"]): c["categoria"] for c in result["classificacoes"]}
        (sex_id,) = categories[SEX_CLASSIFICATION]
        (age_label,) = categories[AGE_CLASSIFICATION].values()
        (series,) = result["series"]
        rows.append(
            {
                "age": age_from_label(age_label),
                "sex": CENSUS_SEX[sex_id],
                "population": int(series["serie"]["2022"]),
            }
        )
    return pd.DataFrame(rows).astype({"age": "int16", "sex": "string", "population": "int64"})


def check_completeness(
    query_id: str, pages: dict[int, dict[str, Any]], frame: pd.DataFrame
) -> Check:
    """Check that one ranking query is complete: every page on disk, every row, no repeats."""
    count, total_pages = pages[1]["count"], pages[1]["totalPages"]
    problems = []
    if sorted(pages) != list(range(1, total_pages + 1)):
        problems.append(f"{len(pages)} pages on disk for totalPages {total_pages}")
    if any(page["count"] != count or page["totalPages"] != total_pages for page in pages.values()):
        problems.append("pages disagree on count or totalPages")
    if len(frame) != count:
        problems.append(f"{len(frame):,} rows for count {count:,}")
    duplicated = int(frame["name"].duplicated().sum())
    if duplicated:
        problems.append(f"{duplicated} duplicated names")
    detail = "; ".join(problems) or f"{total_pages:,} pages, {count:,} rows, names unique"
    return Check(f"completeness {query_id}", not problems, detail)


def check_order(query_id: str, frame: pd.DataFrame) -> Check:
    """Check that, in page order, counts never rise, ranks never fall and none is below 20."""
    problems = []
    if not frame["count"].is_monotonic_decreasing:
        problems.append("count increases")
    if not frame["rank"].is_monotonic_increasing:
        problems.append("rank decreases")
    minimum = int(frame["count"].min())
    if minimum < MINIMUM_COUNT:
        problems.append(f"minimum count {minimum}")
    detail = "; ".join(problems) or f"ordered, minimum count {minimum}"
    return Check(f"order {query_id}", not problems, detail)


def check_decades(names_total: pd.DataFrame, names_decade: pd.DataFrame) -> Check:
    """Check that no name's decade counts add up to more than its total count."""
    decade_sums = names_decade.groupby("name")["count"].sum()
    totals = names_total.set_index("name")["count"].reindex(decade_sums.index)
    # A name missing from the total table compares as NaN, which ``le`` counts as a failure.
    over = int((~decade_sums.le(totals)).sum())
    people = int(decade_sums.sum())
    detail = (
        f"{len(decade_sums):,} names with decade rows, {over} above their total; "
        f"decade counts sum to {people:,} ({people / POPULATION:.1%} of {POPULATION:,})"
    )
    return Check("decades within totals", over == 0, detail)


def check_sexes(names_total: pd.DataFrame, names_sex: pd.DataFrame) -> Check:
    """Check that no name's female and male counts add up to more than its total count."""
    # Summing in the pivot keeps a duplicated row from stopping the build here; the
    # completeness check reports duplicates.
    by_sex = names_sex.pivot_table(index="name", columns="sex", values="count", aggfunc="sum")
    sex_sums = by_sex.sum(axis=1)
    totals = names_total.set_index("name")["count"].reindex(by_sex.index)
    over = int((~sex_sums.le(totals)).sum())
    in_both = by_sex.notna().all(axis=1)
    exact = int((sex_sums[in_both] == totals[in_both]).sum())
    detail = (
        f"{over} names above their total; {int(in_both.sum()):,} names in both sex tables, "
        f"{exact:,} of them with f + m == total"
    )
    return Check("sexes within totals", over == 0, detail)


def check_census(census: pd.DataFrame) -> Check:
    """Check the census table: 101 ages per sex, the national total, and m + f == total."""
    cells = list(zip(census["age"], census["sex"], strict=True))
    if len(cells) != len(CENSUS_CELLS) or set(cells) != CENSUS_CELLS:
        detail = f"{len(cells)} rows; expected one for each age 0-100 and each of total, m, f"
        return Check("census", False, detail)
    by_age = census.pivot(index="age", columns="sex", values="population")
    total = int(by_age["total"].sum())
    mismatched = int((by_age["m"] + by_age["f"] != by_age["total"]).sum())
    detail = (
        f"101 ages for each of total, m, f; total population {total:,}; "
        f"{mismatched} ages with m + f != total"
    )
    return Check("census", total == POPULATION and mismatched == 0, detail)


def check_spot_name(
    response: dict[str, Any],
    names_total: pd.DataFrame,
    names_decade: pd.DataFrame,
    names_sex: pd.DataFrame,
) -> Check:
    """Compare one name endpoint response with the rows built from the rankings."""
    name = response["nome"]
    total = names_total.loc[names_total["name"] == name, "count"].tolist()
    decade_rows = names_decade[names_decade["name"] == name]
    built_decades = dict(
        zip(decade_rows["decade"].tolist(), decade_rows["count"].tolist(), strict=True)
    )
    # ``decadas`` is a list of single-key objects, or null when every decade is suppressed.
    endpoint_decades = {int(d): n for entry in response["decadas"] or [] for d, n in entry.items()}
    sex_rows = names_sex[names_sex["name"] == name]
    built_sexes = dict(zip(sex_rows["sex"].tolist(), sex_rows["count"].tolist(), strict=True))
    endpoint_sexes = {"m": response["homens"], "f": response["mulheres"]}

    problems = []
    if total != [response["frequencia"]]:
        problems.append(f"total {response['frequencia']:,} against rows {total}")
    for decade, count in endpoint_decades.items():
        if built_decades.get(decade) != count:
            problems.append(f"{decade} {count:,} against {built_decades.get(decade)}")
    for sex, count in built_sexes.items():
        if endpoint_sexes[sex] != count:
            problems.append(f"{sex} {endpoint_sexes[sex]:,} against {count:,}")
    decade_sum = sum(endpoint_decades.values())
    detail = "; ".join(problems) or (
        f"total, {len(endpoint_decades)} decades and {len(built_sexes)} sexes match; "
        f"decade sum {decade_sum:,} of {response['frequencia']:,}"
    )
    return Check(f"spot check {name}", not problems, detail)


def raw_digest(raw_dir: Path) -> str:
    """Return one SHA-256 over every saved API response under ``raw_dir``.

    The digest covers a ``sha256sum``-style listing, one ``<file hash>  <relative path>`` line
    per response in sorted path order. It changes when any file's bytes, name or presence
    changes, whatever order the file system lists the files in. It covers the responses
    alone, leaving out the run times in ``retrieval.json``, so two downloads of identical
    responses share one digest.
    """
    paths = sorted(
        path.relative_to(raw_dir).as_posix()
        for path in raw_dir.rglob("*.json")
        if path != raw_dir / RETRIEVAL_PATH.name
    )
    listing = "".join(
        f"{hashlib.sha256((raw_dir / path).read_bytes()).hexdigest()}  {path}\n" for path in paths
    )
    return hashlib.sha256(listing.encode()).hexdigest()


def sort_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort a name table by its group column, then by count descending, then by name."""
    keys = [column for column in ("decade", "sex") if column in frame] + ["count", "name"]
    ascending = [column != "count" for column in keys]
    return frame.sort_values(keys, ascending=ascending, ignore_index=True)


def write_table(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    """Write one table to Parquet and return its row count and SHA-256."""
    frame.to_parquet(path, engine="pyarrow", index=False)
    return {"rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    """Build, check and write every processed table and the manifest."""
    pages = {query_id: load_pages(RAW_DIR / "ranking" / query_id) for query_id in RANKING_QUERIES}
    frames = {query_id: parse_ranking(query_id, pages[query_id]) for query_id in RANKING_QUERIES}
    names_total = frames["total"]
    names_decade = pd.concat(
        [frames[q] for q, params in RANKING_QUERIES.items() if "decada" in params],
        ignore_index=True,
    )
    names_sex = pd.concat(
        [frames[q] for q, params in RANKING_QUERIES.items() if "sexo" in params],
        ignore_index=True,
    )
    census = parse_census(json.loads((RAW_DIR / "census" / "values.json").read_bytes()))

    # The order checks read the frames in page order, so they run before any sorting.
    checks = [check_completeness(q, pages[q], frames[q]) for q in RANKING_QUERIES]
    checks += [check_order(q, frames[q]) for q in RANKING_QUERIES]
    checks += [
        check_decades(names_total, names_decade),
        check_sexes(names_total, names_sex),
        check_census(census),
    ]
    for name in SPOT_CHECK_NAMES:
        response = json.loads((RAW_DIR / "spot_checks" / f"{name}.json").read_bytes())
        checks.append(check_spot_name(response, names_total, names_decade, names_sex))

    width = max(len(check.name) for check in checks)
    for check in checks:
        print(f"{check.name:<{width}}  {'pass' if check.passed else 'FAIL'}  {check.detail}")
    passed = sum(check.passed for check in checks)
    print(f"{passed} of {len(checks)} checks passed")

    tables = {
        "names_total.parquet": sort_names(names_total),
        "names_decade.parquet": sort_names(names_decade),
        "names_sex.parquet": sort_names(names_sex),
        "census_age_sex.parquet": census.sort_values(["sex", "age"], ignore_index=True),
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {file: write_table(table, PROCESSED_DIR / file) for file, table in tables.items()}

    manifest = {
        "source": {
            "citation": [
                "IBGE, Censo Demográfico 2022, Nomes no Brasil",
                f"IBGE, Censo Demográfico 2022, tabela {CENSUS_AGGREGATE} (SIDRA)",
            ],
            "ranking_url": f"{RANKING_URL}?page={{page}}[&decada={{decade}}|&sexo={{sex}}]",
            "name_url": NAME_URL,
            "census_aggregate": CENSUS_AGGREGATE,
            "census_variable": CENSUS_VARIABLE,
            "census_metadata_url": CENSUS_METADATA_URL,
            "census_values_url": (
                f"{CENSUS_VALUES_URL}?localidades=N1[all]"
                f"&classificacao={census_classification('{single-year age ids}')}"
            ),
        },
        "retrieved": json.loads(RETRIEVAL_PATH.read_bytes()),
        "queries": {
            q: {
                "count": pages[q][1]["count"],
                "totalPages": pages[q][1]["totalPages"],
                "rows": len(frames[q]),
            }
            for q in RANKING_QUERIES
        },
        "raw_digest": raw_digest(RAW_DIR),
        "outputs": outputs,
        "checks": [
            {"name": c.name, "result": "pass" if c.passed else "fail", "detail": c.detail}
            for c in checks
        ],
    }
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (PROCESSED_DIR / "manifest.json").write_text(text, encoding="utf-8")

    if passed < len(checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
