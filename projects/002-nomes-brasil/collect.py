"""Download IBGE's name rankings and Censo 2022 population by age and sex.

Run from the project directory with ``uv run python collect.py``. Each response body
is saved under ``data/raw/`` byte for byte as IBGE sent it, and ``build.py`` turns those
files into the processed tables. A rerun skips every file already saved, so an
interrupted download resumes.
"""

import json
import re
import threading
import time
from collections.abc import Callable
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
CENSUS_VALUES_URL = (
    f"{API}/agregados/{CENSUS_AGGREGATE}/periodos/2022/variaveis/{CENSUS_VARIABLE}"
)

USER_AGENT = "DataVerso/002 (+https://github.com/jhlopesalves/DataVerso)"
WORKERS = 6
TIMEOUT_SECONDS = 30

DECADES = range(1930, 2030, 10)

# Every ranking query, keyed by its folder name under data/raw/ranking/.
# The API ignores ``sexo`` when ``decada`` is set, so these filters are never combined.
RANKING_QUERIES: dict[str, dict[str, str]] = {
    "total": {},
    **{f"decade_{decade}": {"decada": str(decade)} for decade in DECADES},
    "sex_f": {"sexo": "f"},
    "sex_m": {"sexo": "m"},
}

# Two very common old names, a -ton name, a 1990s name and a recent name.
SPOT_CHECK_NAMES = ("maria", "jose", "wellington", "jessica", "enzo")

# SIDRA classification ids in aggregate 9514, and the categories requested from each.
SEX_CLASSIFICATION = 2
AGE_CLASSIFICATION = 287
DECLARATION_CLASSIFICATION = 286  # Forma de declaração da idade
DECLARATION_TOTAL = 113635
CENSUS_SEX = {"6794": "total", "4": "m", "5": "f"}

_thread_local = threading.local()


def age_from_label(label: str) -> int | None:
    """Return the completed age (100 for 100+) of a SIDRA ``Idade`` label, or None."""
    if label == "Menos de 1 ano":
        return 0
    if label == "100 anos ou mais":
        return 100
    # Matches "1 ano" and "2 anos" to "99 anos". Groups such as "0 a 4 anos" do not
    # match.
    match = re.fullmatch(r"(\d+) anos?", label)
    return int(match.group(1)) if match else None


def get_session() -> requests.Session:
    """Return this thread's HTTP session, creating it with retries on first use."""
    session: requests.Session | None = getattr(_thread_local, "session", None)
    if session is None:
        retry = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            respect_retry_after_header=True,
        )
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
        session.mount("https://", HTTPAdapter(max_retries=retry))
        _thread_local.session = session
    return session


def fetch(url: str, params: dict[str, str]) -> bytes:
    """Send one GET request on this thread's session and return the response bytes."""
    response = get_session().get(url, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    # Retries happen inside urllib3; their record travels with the final response.
    retries = response.raw.retries
    for attempt in retries.history if retries else ():
        print(f"  retried {response.url}: {attempt.status or attempt.error}")
    return response.content


def write_atomic(path: Path, content: bytes) -> None:
    """Write ``content`` to ``path`` atomically via a sibling temporary file."""
    # Renaming replaces the target in one step, so an interrupted run never leaves a
    # partial file.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def load_saved(path: Path) -> object:
    """Return the saved JSON, or None if the file is absent or unreadable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_bytes())
    except json.JSONDecodeError:
        return None


def save_response(
    url: str,
    params: dict[str, str],
    path: Path,
    is_valid: Callable[[object], bool] = lambda saved: saved is not None,
) -> bool:
    """Download ``url`` to ``path`` unless ``load_saved(path)`` passes ``is_valid``."""
    if is_valid(load_saved(path)):
        return False
    write_atomic(path, fetch(url, params))
    return True


def page_path(query_id: str, page: int) -> Path:
    """Return where one ranking page is saved."""
    return RAW_DIR / "ranking" / query_id / f"page_{page:04d}.json"


def save_page(query_id: str, page: int) -> bool:
    """Download a ranking page if absent, and report whether it was fetched."""
    params = {**RANKING_QUERIES[query_id], "page": str(page)}
    return save_response(
        RANKING_URL,
        params,
        page_path(query_id, page),
        lambda saved: isinstance(saved, dict) and isinstance(saved.get("items"), list),
    )


def collect_ranking(pool: ThreadPoolExecutor, query_id: str) -> tuple[int, int]:
    """Download every page of one ranking query and return ``(totalPages, fetched)``."""
    # Read page 1 from disk so that a resumed run takes totalPages from the same body
    # as the first run did.
    first_fetched = save_page(query_id, 1)
    total_pages: int = json.loads(page_path(query_id, 1).read_bytes())["totalPages"]
    fetched = pool.map(partial(save_page, query_id), range(2, total_pages + 1))
    return total_pages, first_fetched + sum(fetched)


def single_year_age_ids(metadata: dict[str, Any]) -> list[int]:
    """Select the 101 single-year ``Idade`` categories from the aggregate's metadata."""
    (ages,) = (c for c in metadata["classificacoes"] if c["id"] == AGE_CLASSIFICATION)
    return [
        c["id"] for c in ages["categorias"] if age_from_label(c["nome"]) is not None
    ]


def census_classification(age_ids: str) -> str:
    """Build ``classificacao`` for each sex, given ages and all declaration forms."""
    return (
        f"{SEX_CLASSIFICATION}[{','.join(CENSUS_SEX)}]"
        f"|{AGE_CLASSIFICATION}[{age_ids}]"
        f"|{DECLARATION_CLASSIFICATION}[{DECLARATION_TOTAL}]"
    )


def collect_census() -> int:
    """Download the census metadata and single-year population; return files fetched."""
    metadata_path = RAW_DIR / "census" / "metadata.json"
    metadata_fetched = save_response(CENSUS_METADATA_URL, {}, metadata_path)
    age_ids = ",".join(
        map(str, single_year_age_ids(json.loads(metadata_path.read_bytes())))
    )
    # N1 is the national level; requests percent-encodes the brackets, commas and pipes.
    params = {"localidades": "N1[all]", "classificacao": census_classification(age_ids)}
    values_path = RAW_DIR / "census" / "values.json"
    return metadata_fetched + save_response(CENSUS_VALUES_URL, params, values_path)


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
    # Keep the start of any earlier run, whether it completed or was interrupted.
    if retrieval["start"] is None:
        retrieval["start"] = utc_now()
        write_retrieval(retrieval)

    pages_fetched = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for query_id in RANKING_QUERIES:
            query_start = time.perf_counter()
            total_pages, fetched = collect_ranking(pool, query_id)
            pages_fetched += fetched
            elapsed = time.perf_counter() - query_start
            print(
                f"{query_id:<12} {total_pages:>5} pages {fetched:>5} fetched "
                f"{elapsed:>7.1f} s"
            )

    files_fetched = pages_fetched + collect_census()
    for name in SPOT_CHECK_NAMES:
        path = RAW_DIR / "spot_checks" / f"{name}.json"
        files_fetched += save_response(NAME_URL.format(name=name), {}, path)

    # The end marks the last download. If all files are already saved, keep the
    # earlier end time. If this run downloads any files, its end supersedes that time.
    if files_fetched:
        retrieval["end"] = utc_now()
        write_retrieval(retrieval)
    elapsed = time.perf_counter() - run_start
    print(
        f"{pages_fetched} ranking pages fetched in {elapsed:.1f} s "
        f"({pages_fetched / elapsed:.1f} pages per second)"
    )


if __name__ == "__main__":
    main()
