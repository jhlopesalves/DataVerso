# DataVerso 002 · Brazilian first names in Censo 2022

## Reproduce locally

From the DataVerso root:

```bash
cd projects/002-nomes-brasil
uv sync
uv run python collect.py
uv run python build.py
```

The processed tables in [`data/processed/`](data/processed/) are committed, so the analysis
runs from them directly. Re-collecting downloads about 16,000 pages from IBGE's API and takes
about 12 minutes; an interrupted `collect.py` resumes where it stopped. `build.py` rebuilds
the tables from the downloaded responses, checks them and exits with an error when a check
fails.

## Data and provenance

Both sources come from IBGE's public data API and cover Brazil as a whole.

- **Nomes no Brasil**, from Censo Demográfico 2022: the national ranking of first names at
  `https://servicodados.ibge.gov.br/api/v3/nomes/2022/localidade/0/ranking/nome`, collected
  in full, for each birth-decade label and for each sex. The name endpoint
  `https://servicodados.ibge.gov.br/api/v3/nomes/2022/nome/{name}` cross-checks five names.
- **Tabela 9514 (SIDRA)**, from Censo Demográfico 2022: resident population (variable 93) by
  sex and single year of age, at `https://servicodados.ibge.gov.br/api/v3/agregados/9514`.

The data were retrieved on 26 November 2026, between 13:37 and 13:49 UTC.

| Ranking query | Names |
| --- | ---: |
| All names | 128,458 |
| `sexo=f` | 73,361 |
| `sexo=m` | 60,954 |
| `decada=1930` | 5,965 |
| `decada=1940` | 11,959 |
| `decada=1950` | 18,557 |
| `decada=1960` | 25,887 |
| `decada=1970` | 32,071 |
| `decada=1980` | 37,381 |
| `decada=1990` | 34,571 |
| `decada=2000` | 27,374 |
| `decada=2010` | 18,774 |
| `decada=2020` | 8,985 |

| Table | One row per | Rows |
| --- | --- | ---: |
| `names_total.parquet` | name | 128,458 |
| `names_decade.parquet` | decade label and name | 221,524 |
| `names_sex.parquet` | sex and name | 134,315 |
| `census_age_sex.parquet` | age (0 to 100, where 100 means 100 or more) and sex | 303 |

The raw API responses stay local in `data/raw/`. [`manifest.json`](data/processed/manifest.json)
records the endpoints, retrieval times, query sizes, check results and each table's SHA-256. It
also holds one SHA-256 over all raw responses, so a later download can be compared with this one.

## What the data record

These rules come from IBGE's
[technical note 01/2025](https://censo2022.ibge.gov.br/nomes/nota_tecnica/notatecnica.pdf).

- Only the first name is counted: IBGE splits the name field at its first space.
- Before the split, IBGE removes one-letter words, two-letter words made of one repeated letter,
  and the connectives da, das, de, des, do, dos, di, dis, du and dus.
- Diacritics are not recorded, so Antônio and Cauã appear as `antonio` and `caua`.
- Spelling variants are counted separately, so `luis` and `luiz` are different names.
- The national tables list only names with a frequency of at least 20, and the same threshold
  applies within each birth decade.
- The reference date is 1 August 2022.
- Sex is as declared at the interview.
- The note leaves the boundaries of the birth decades undefined.

Decade counts are people alive and resident in Brazil on the reference date, grouped by IBGE's
birth-decade labels (1930, 1940, …, 2020).

The national table accounts for 190,383,184 residents (93.7% of 203,080,756); names with fewer
than 20 bearers are not shown.

## Publication checks

- Confirm IBGE's terms for reusing these data and the citation it requests.
