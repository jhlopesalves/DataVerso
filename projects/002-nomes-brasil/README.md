# DataVerso 002 · Brazilian first names across generations

[Notebook](dataverso_002.ipynb) · [DataVerso](../../README.md)

How did Brazilian first names change from one generation to the next? This exploration uses IBGE's *Nomes no Brasil* tables from Censo Demográfico 2022, collected directly from IBGE's official API: 128,458 first names and the birth decades of 186 million residents. It looks at how concentrated names were in each decade, which names mark a generation, which life cycles names follow, and how their spelling changed. Figures are in English below; Portuguese versions are in [`figures/`](figures/).

## Findings

### 1. The ten most common names lost half their weight, then rose again

Within women's names, the ten most common names of each decade held 41% of people in the 1940 and 1950 labels, 19% in the 1980 and 1990 labels and 33% in the 2020 label; within men's names, the corresponding shares were 42–43%, about 20% and 30%. Maria alone held 31% of women's names in the 1950 label, fell to 4% in the 1990 label and reached 13% in the 2020 label. The number of names needed to cover half of each decade follows the same arc (women's names: 25, 80 and 37 in the 1950, 1980 and 2010 labels; men's: 20, 51 and 27), and the renewed concentration since the 1990s also appears after removing each decade's 20 most common names.

![Two panels of horizontal bars, women's names above and men's below, one bar per birth decade from 1930 to 2020. Each bar is made of the decade's ten most common names as blocks sized by their share. Maria's and José's blocks dominate until the 1960s; the bars shrink to about a fifth of people in the 1980s and 1990s and grow again to about a third, with João back at the top of men's names, followed by Davi and Miguel.](figures/01_top_names_en.png)

### 2. Every generation has its own names

Measured by the highest Log Ratio against the other nine decades combined among names that peak in each decade, the most characteristic names from 1930 to 2020 are Therezinha, Darci, Creusa, Amarildo, Valdirene, Éder, Jéssica, Kailane, Sophia and Gael. There are 450,000 Jéssicas in Brazil, 73% of them born in the 1990 label, and 87% of the people named Gael were born in the 2020 label.

![A ridgeline of ten names, one per birth decade from Therezinha (1930) to Gael (2020). Each wave shows the name's share across all decades, scaled to its own peak, and the peaks move diagonally from top left to bottom right. A table on the right gives each name's number of bearers and the share born in its peak decade.](figures/02_generations_en.png)

### 3. Names have life cycles, and one of them is the comeback

Clustering the log-scale trajectories of the 863 names held by at least 1 in 2,000 people in some decade with k-means (`k = 6`) finds six groups: five are phases of one rise-and-fall cycle, and one, of 57 names, lost ground until the 1970s and grew again. It includes João, Pedro and Miguel alongside older names such as Alice, Helena, Cecília, Joaquim and Aurora. The silhouette score falls steadily from three to ten groups, so the number of groups is a choice of resolution; a comeback group separates at every resolution from five to eight, and the same 57 names recur across random starts.

![Five small panels show the median trajectory and spread of five groups of names: falling since 1930, peaking in the 1960s, in the 1970s and 1980s, in the 1990s, and rising. A large teal panel shows the sixth group, 57 names that lost ground until the 1970s to 1990s and then grew again, with João, Pedro and Miguel highlighted.](figures/03_life_cycles_en.png)

### 4. First-name spelling has its fashions too

Names with k, w or y, th, a double consonant, or ending in -son or -ton went from 4% of the 1950 label to 20% of the 2000 label. The -son ending was a men's fashion that reached 9% of men's names in the 1980 label and fell to about 1%; double consonants, usually ll or nn, became mainly a women's fashion (9% in the 2010 label); th keeps rising among men's names (8% in the 2020 label). Brazil's 1943 spelling rules treated k, w and y as letters for special cases; the 1990 Orthographic Agreement, in force in Brazil since 2009, made them full letters of the alphabet.

![Five mirrored rows, one per spelling feature, labelled with large letters: k w y, th, ll nn, -son and -ton. In each row, the share of women's names with the feature is drawn above a central line and the share of men's names below it, by birth decade from 1930 to 2020, on one scale. k w y grows for both, th mostly for men, double consonants mostly for women, and -son and -ton appear almost only below the line, peaking around the 1980s.](figures/04_spelling_en.png)

## Methods

- **Denominators:** Every share uses the listed people of the relevant decade, or of the relevant decade and sex class, rather than census populations, because IBGE leaves the birth-decade boundaries undefined.
- **Concentration and robustness:** Concentration is measured with the half-coverage count (`n50`), the top-ten share and the Hill numbers of order 1 and 2, checked under one common share cut-off across decades, without Maria, Ana, João and José, within each sex class, and without each decade's 20 most common names.
- **Sex class:** Each name is classed nationally as women's (`f`) or men's (`m`) when at least 95% of its bearers belong to that sex, and as `other` otherwise.
- **Distinctiveness:** Characteristic names are ranked by Hardie's Log Ratio against the other nine decades pooled (adding 0.5 to the pooled counts), among names held by at least 1 in 2,000 listed people of the decade, with the added requirement of peaking in that decade for the generation figure.
- **Life cycles:** k-means (`k = 6`) is fitted on standardised base-10 log trajectories across the ten labels (using `1e-6` for unlisted decades) for the 863 names reaching 1 in 2,000 in some decade, with resolution and seed checks recorded in [`results/life_cycle_robustness.csv`](results/life_cycle_robustness.csv).
- **Spelling:** Five orthographic features (`[kwy]`, `th`, doubled consonants, `son$` and `ton$`) are matched with regular expressions on IBGE's lower-case, unaccented forms.

## Independent audit

A separate implementation, written from the raw API responses without reading this project's code, reproduced every results table cell by cell (the six core tables, the common cut-off table and the five tables behind the figures; 12,336 cells in the last five alone, with no disagreement). Its sensitivity checks found that the order-2 Hill number keeps its direction under extreme assumptions about unpublished names, and that concentration after the 1980s also rises without each decade's 20 most common names.

## Limitations

- IBGE records only the first word of the name field, so compound names such as Maria Eduarda or Davi Lucca collapse onto their first word, which can inflate concentration after 1990.
- Diacritics are stripped while spelling variants such as Sophia and Sofia are counted separately, which pulls measured concentration in the other direction.
- Names with fewer than 20 bearers nationally or within a birth decade are omitted from the published tables.
- Decade counts record people alive in Brazil on 1 August 2022 rather than births, so older decades are mostly women among survivors; within-sex measures address that composition.
- IBGE leaves the birth-decade boundaries undefined: the 1930 label includes earlier births and the 2020 label spans about four years.
- Each name receives a single national sex class across all ten decades.
- Regular-expression spelling patterns group distinct historical layers together, so older spellings such as Thereza and newer ones such as Thiago both count as `th`.

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

To run the analysis, select `.venv/bin/python` in this directory as the notebook kernel and run `dataverso_002.ipynb` from here.

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

## Credit and licence

Data: "IBGE, Censo Demográfico 2022, Nomes no Brasil" and "IBGE, Censo Demográfico 2022, tabela 9514 (SIDRA)", retrieved from IBGE's API on the date recorded in [`data/processed/manifest.json`](data/processed/manifest.json). Code under MIT ([`LICENSE`](../../LICENSE)); text and figures under CC BY 4.0 ([`LICENSE-CC-BY-4.0.txt`](../../LICENSE-CC-BY-4.0.txt)).

## Publication checks

- IBGE's API documentation states no licence. IBGE has stated that data published on its portal may be reused with credit to IBGE (as reported on the OpenStreetMap wiki, which cites an IBGE communication); confirm the current terms and the preferred citation with IBGE.

## Outputs

[`figures/`](figures/) holds the eight PNG figures (four compositions in Brazilian Portuguese and English); [`results/`](results/) holds the tables behind them.
