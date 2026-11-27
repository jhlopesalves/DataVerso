"""Test the parsers and data checks in ``build.py`` on small hand-written responses."""

import json
from pathlib import Path
from typing import Any

import pytest

import build

FIXTURES = Path(__file__).parent / "fixtures"
QUERY_ID = "decade_1950"


def fixture_pages() -> dict[int, dict[str, Any]]:
    """Load a fresh copy of the two-page fixture query, so each test can alter it."""
    return build.load_pages(FIXTURES / "ranking" / QUERY_ID)


def test_parse_ranking_builds_typed_rows_with_the_decade_from_the_query_id() -> None:
    frame = build.parse_ranking(QUERY_ID, fixture_pages())

    assert list(frame.dtypes.astype(str).items()) == [
        ("decade", "int16"),
        ("name", "string"),
        ("count", "int64"),
        ("rank", "int32"),
    ]
    assert frame["decade"].tolist() == [1950] * 5
    assert frame["name"].tolist() == ["maria", "jose", "antonio", "joao", "francisco"]
    assert frame["rank"].tolist() == [1, 2, 3, 4, 4]


def test_checks_pass_on_a_complete_query() -> None:
    pages = fixture_pages()
    frame = build.parse_ranking(QUERY_ID, pages)

    assert build.check_completeness(QUERY_ID, pages, frame).passed
    assert build.check_order(QUERY_ID, frame).passed


def test_completeness_fails_when_rows_fall_short_of_count() -> None:
    pages = fixture_pages()
    pages[2]["items"].pop()
    check = build.check_completeness(
        QUERY_ID, pages, build.parse_ranking(QUERY_ID, pages)
    )

    assert not check.passed
    assert "4 rows for count 5" in check.detail


def test_completeness_fails_on_a_duplicated_name() -> None:
    pages = fixture_pages()
    pages[2]["items"][0]["nome"] = "maria"
    check = build.check_completeness(
        QUERY_ID, pages, build.parse_ranking(QUERY_ID, pages)
    )

    assert not check.passed
    assert "1 duplicated names" in check.detail


def test_order_fails_when_frequencia_increases() -> None:
    pages = fixture_pages()
    pages[2]["items"][1]["frequencia"] = 600_000
    check = build.check_order(QUERY_ID, build.parse_ranking(QUERY_ID, pages))

    assert not check.passed
    assert "count increases" in check.detail


def test_parse_census_maps_age_labels_and_converts_values() -> None:
    values = json.loads((FIXTURES / "census_values.json").read_bytes())
    census = build.parse_census(values)

    assert census.to_dict("records") == [
        {"age": 0, "sex": "total", "population": 2_358_265},
        {"age": 1, "sex": "m", "population": 1_203_456},
        {"age": 100, "sex": "f", "population": 25_000},
    ]
    assert census.dtypes.astype(str).tolist() == ["int16", "string", "int64"]


def test_raw_digest_is_independent_of_listing_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for relative in (
        "census/values.json",
        "ranking/total/page_0001.json",
        "spot_checks/enzo.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"file": relative}))
    digest = build.raw_digest(tmp_path)

    # Reverse whatever order the file system lists the files in.
    rglob = Path.rglob
    monkeypatch.setattr(
        Path, "rglob", lambda self, pattern: reversed(list(rglob(self, pattern)))
    )
    assert build.raw_digest(tmp_path) == digest

    # The run times are not part of the responses, so they leave the digest unchanged.
    (tmp_path / "retrieval.json").write_text('{"start": null, "end": null}')
    assert build.raw_digest(tmp_path) == digest
