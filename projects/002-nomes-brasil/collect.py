"""Download IBGE's Nomes no Brasil rankings and the Censo 2022 population by age and sex.

Run from the project directory with ``uv run python collect.py``. Each response body is saved
under ``data/raw/`` byte for byte as IBGE sent it, and ``build.py`` turns those files into the
processed tables. A rerun skips every file already saved, so an interrupted download resumes.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
RETRIEVAL_PATH = RAW_DIR / "retrieval.json"

API = "https://servicodados.ibge.gov.br/api/v3"
RANKING_URL = f"{API}/nomes/2022/localidade/0/ranking/nome"  # localidade 0 is Brazil
NAME_URL = f"{API}/nomes/2022/nome/{{name}}"
CENSUS_AGGREGATE = 9514
CENSUS_VARIABLE = 93  # População residente
CENSUS_METADATA_URL = f"{API}/agregados/{CENSUS_AGGREGATE}/metadados"
CENSUS_VALUES_URL = f"{API}/agregados/{CENSUS_AGGREGATE}/periodos/2022/variaveis/{CENSUS_VARIABLE}"

USER_AGENT = "DataVerso/002 (+https://github.com/jhlopesalves/DataVerso)"
WORKERS = 6
TIMEOUT_SECONDS = 30

DECADES = range(1930, 2030, 10)

# Every ranking query, keyed by the short id that names its folder under data/raw/ranking/.
# The API ignores ``sexo`` when ``decada`` is set, so the two filters are never combined.
RANKING_QUERIES: dict[str, dict[str, str]] = {
    "total": {},
    **{f"decade_{decade}": {"decada": str(decade)} for decade in DECADES},
    "sex_f": {"sexo": "f"},
    "sex_m": {"sexo": "m"},
}

SPOT_CHECK_NAMES = ("maria", "jose", "wellington", "jessica", "enzo")

# SIDRA classification ids in aggregate 9514, and the categories requested from each.
SEX_CLASSIFICATION = 2
AGE_CLASSIFICATION = 287
DECLARATION_CLASSIFICATION = 286  # Forma de declaração da idade
DECLARATION_TOTAL = 113635
CENSUS_SEX = {"6794": "total", "4": "m", "5": "f"}


def age_from_label(label: str) -> int | None:
    """Return the single year of age a SIDRA ``Idade`` label denotes.

    Parameters
    ----------
    label
        A category label such as ``Menos de 1 ano``, ``7 anos`` or ``100 anos ou mais``.

    Returns
    -------
    int or None
        The age in completed years, with 100 standing for 100 or more, or None for a label
        that names months or an age group.
    """
    if label == "Menos de 1 ano":
        return 0
    if label == "100 anos ou mais":
        return 100
    # Matches "1 ano" and "2 anos" to "99 anos"; groups such as "0 a 4 anos" do not match.
    match = re.fullmatch(r"(\d+) anos?", label)
    return int(match.group(1)) if match else None


def make_session() -> requests.Session:
    """Create the shared HTTP session with the project's user agent and retry policy."""
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch(session: requests.Session, url: str, params: dict[str, str]) -> bytes:
    """Send one GET request and return the response body as received.

    Retries happen inside urllib3; their record travels with the final response, so each
    one is printed here.
    """
    response = session.get(url, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    retries = response.raw.retries
    for attempt in retries.history if retries else ():
        print(f"  retried {response.url}: {attempt.status or attempt.error}")
    return response.content


def write_atomic(path: Path, content: bytes) -> None:
    """Write ``content`` to ``path`` through a temporary file in the same directory.

    The rename replaces the target in one step, so an interrupted run leaves either the
    complete file or no file at all.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def load_saved(path: Path) -> object:
    """Return the parsed JSON saved at ``path``, or None when the file is absent or unreadable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_bytes())
    except json.JSONDecodeError:
        return None


def save_body(session: requests.Session, url: str, params: dict[str, str], path: Path) -> bool:
    """Download ``url`` to ``path`` unless a parseable body is saved there; report a download."""
    if load_saved(path) is not None:
        return False
    write_atomic(path, fetch(session, url, params))
    return True


def page_path(query_id: str, page: int) -> Path:
    """Return where one ranking page is saved."""
    return RAW_DIR / "ranking" / query_id / f"page_{page:04d}.json"


def save_page(session: requests.Session, query_id: str, page: int) -> bool:
    """Download one ranking page unless it is already saved, and report whether it was fetched."""
    path = page_path(query_id, page)
    saved = load_saved(path)
    if isinstance(saved, dict) and isinstance(saved.get("items"), list):
        return False
    params = {**RANKING_QUERIES[query_id], "page": str(page)}
    write_atomic(path, fetch(session, RANKING_URL, params))
    return True


def collect_ranking(
    session: requests.Session, pool: ThreadPoolExecutor, query_id: str
) -> tuple[int, int]:
    """Download every page of one ranking query.

    Returns
    -------
    tuple of int
        The query's ``totalPages`` and the number of pages fetched in this run.
    """
    # Page 1 is read from disk, fetched or not, so a resumed run takes totalPages from the
    # same body as the first run did.
    first_fetched = save_page(session, query_id, 1)
    total_pages: int = json.loads(page_path(query_id, 1).read_bytes())["totalPages"]
    fetched = pool.map(partial(save_page, session, query_id), range(2, total_pages + 1))
    return total_pages, first_fetched + sum(fetched)


def single_year_age_ids(metadata: dict[str, Any]) -> list[int]:
    """Select the 101 single-year ``Idade`` categories from the aggregate's metadata."""
    (ages,) = (c for c in metadata["classificacoes"] if c["id"] == AGE_CLASSIFICATION)
    return [c["id"] for c in ages["categorias"] if age_from_label(c["nome"]) is not None]


def census_classification(age_ids: str) -> str:
    """Return the ``classificacao`` parameter: each sex, the given ages, all declaration forms."""
    return (
        f"{SEX_CLASSIFICATION}[{','.join(CENSUS_SEX)}]"
        f"|{AGE_CLASSIFICATION}[{age_ids}]"
        f"|{DECLARATION_CLASSIFICATION}[{DECLARATION_TOTAL}]"
    )


def collect_census(session: requests.Session) -> int:
    """Download the aggregate's metadata, then the population for every single year of age.

    Returns
    -------
    int
        The number of files downloaded in this run.
    """
    metadata_path = RAW_DIR / "census" / "metadata.json"
    metadata_fetched = save_body(session, CENSUS_METADATA_URL, {}, metadata_path)
    age_ids = ",".join(map(str, single_year_age_ids(json.loads(metadata_path.read_bytes()))))
    # N1 is the national level; requests percent-encodes the brackets, commas and pipes.
    params = {"localidades": "N1[all]", "classificacao": census_classification(age_ids)}
    values_path = RAW_DIR / "census" / "values.json"
    return metadata_fetched + save_body(session, CENSUS_VALUES_URL, params, values_path)


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_retrieval(retrieval: dict[str, str | None]) -> None:
    """Save the retrieval times next to the raw responses."""
    write_atomic(RETRIEVAL_PATH, (json.dumps(retrieval, indent=2) + "\n").encode())


def main() -> None:
    """Download every raw response that is not yet saved."""
    run_start = time.perf_counter()
    retrieval: dict[str, str | None] = {"start": None, "end": None}
    if RETRIEVAL_PATH.exists():
        retrieval = json.loads(RETRIEVAL_PATH.read_bytes())
    # Any earlier run, complete or interrupted, started before this one, so its start is kept.
    retrieval["start"] = retrieval["start"] or utc_now()
    write_retrieval(retrieval)

    session = make_session()
    pages_fetched = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for query_id in RANKING_QUERIES:
            query_start = time.perf_counter()
            total_pages, fetched = collect_ranking(session, pool, query_id)
            pages_fetched += fetched
            elapsed = time.perf_counter() - query_start
            print(f"{query_id:<12} {total_pages:>5} pages {fetched:>5} fetched {elapsed:>7.1f} s")

    files_fetched = pages_fetched + collect_census(session)
    for name in SPOT_CHECK_NAMES:
        path = RAW_DIR / "spot_checks" / f"{name}.json"
        files_fetched += save_body(session, NAME_URL.format(name=name), {}, path)

    # The end marks the last download, so a run that only finds every file saved keeps the
    # end of the run that completed them. A run that downloads ends after every earlier one.
    if files_fetched:
        retrieval["end"] = utc_now()
        write_retrieval(retrieval)
    elapsed = time.perf_counter() - run_start
    print(f"{pages_fetched} ranking pages fetched in {elapsed:.1f} s", end="")
    print(f" ({pages_fetched / elapsed:.1f} pages per second)")


if __name__ == "__main__":
    main()
