# CS5812 Predictive Data Analysis — EDA Pipeline

Steam Games Dataset analysis pipeline for the CS5812 coursework.

**Dataset:** FronkonGames/steam-games-dataset (~124k Steam games)
**Main task:** Classify `estimated_owners` bucket (classification)
**Secondary task:** Predict review sentiment ratio (regression)

## Setup

```bash
# From the project root
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place `games.csv` in `data/raw/`.

## Project layout

```
cs5812-predictive/
├── data/
│   ├── raw/              <- put games.csv here
│   ├── interim/          <- cleaned / featurised parquets (created by scripts)
│   └── processed/        <- modelling-ready train/test splits
├── src/
│   ├── 01_inspect.py         Stage 1  - initial inspection
│   ├── 02_clean.py           Stage 2  - missing values + sentinel zeros
│   ├── 03_target.py          Stage 3  - build classification & regression targets
│   ├── 04_univariate.py      Stage 4  - distributions, skew, log decisions
│   ├── 05_parse_lists.py     Stage 5  - parse genres/categories/languages
│   ├── 06_bivariate.py       Stage 6  - features vs. target
│   ├── 07_pca.py             Stage 7  - PCA (unsupervised #1)
│   ├── 08_cluster.py         Stage 8  - K-Means clustering (unsupervised #2)
│   ├── 09_features.py        Stage 9  - final features + stratified split
│   └── utils.py              shared helpers
├── reports/
│   └── figures/          <- all plots & tables for the PDF
└── requirements.txt
```

## Running the pipeline

Scripts use `# %%` cell markers so you can run them interactively in
Cursor / VS Code (click "Run Cell" above each marker), or end-to-end:

```bash
python src/01_inspect.py
python src/02_clean.py
python src/03_target.py
python src/04_univariate.py
python src/05_parse_lists.py
python src/06_bivariate.py
python src/07_pca.py
python src/08_cluster.py
python src/09_features.py
```

**Recommended workflow:** run stages 1–2 first, check the sentinel-zero
percentages and row count, then tune thresholds in stages 3+ to match
what the data actually looks like. Several thresholds in the scripts
(MIN_REVIEWS=50, top-15 genres, 4 collapsed classes) are starting
points — update them based on your EDA findings and justify the change
in the report.

## Report mapping

Each stage produces figures under `reports/figures/` that map directly
to sections of the coursework report:

| Stage | Report section                              |
|-------|---------------------------------------------|
| 1     | Data description and research question      |
| 2     | Data preparation and cleaning               |
| 3     | Data preparation and cleaning               |
| 4-6   | Exploratory data analysis                   |
| 7-8   | Exploratory data analysis (unsupervised)    |
| 9     | Machine Learning / Deep Learning prediction |
