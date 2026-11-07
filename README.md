# DataPilot-Spark

DataPilot-Spark turns a large raw dataset into automated analytics. Specialised
agents profile, clean, explore, model and explain the data. Apache Spark (PySpark)
does the heavy processing, and a Streamlit dashboard shows the results.

**Team:** Meera (data and Spark), Indrajith (machine learning), Ardra (supervisor, insights, dashboard)

## How it works

```
Raw dataset -> Data Agent -> Spark cleaning -> EDA Agent -> ML Agent -> Insight Agent -> Dashboard
               profile.json  Parquet          eda_results   model_results  insights.json
```

Each stage writes a JSON file that the next stage and the dashboard read. The exact
formats are in `docs/contracts.md`. Agents call tested Spark functions; they do not
run arbitrary code on the data.

## Datasets

| Name | Raw file | Target |
|---|---|---|
| taxi | `data/raw/yellow_tripdata_2026-01.parquet` | `fare_amount` (regression) |
| flights | `data/raw/flights.parquet` | `delayed` (1 = arrived more than 15 minutes late) |

Per-dataset settings (target, fill columns, leakage columns, chart columns) are in
`spark_jobs/datasets.py`. Files in `data/raw/` are never edited.

## Setup

Requirements: Python 3.13, Java (required by Spark), PowerShell or any shell.

```powershell
git clone https://github.com/meeraa77/datapilot-spark.git
cd datapilot-spark
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put the raw datasets in `data/raw/`. Spark runs in local mode, so no cluster is needed.
If the LLM stage needs an API key, copy `.env.example` to `.env` and fill it in.
Never commit `.env`.

## Running the project

Run everything from the project root with the virtual environment active.
Use `taxi` or `flights` as the dataset name.

```powershell
# 1. Profile -> outputs/<dataset>/profile.json
TODO: Data Agent command

# 2. Clean -> data/processed/<dataset>/ and outputs/<dataset>/cleaning_report.json
python -m spark_jobs.clean taxi
python -m spark_jobs.clean flights

# 3. EDA -> outputs/<dataset>/eda_results.json
TODO: EDA Agent command

# 4. Full pipeline (all stages, one call)
TODO: supervisor command, e.g. python -c "from agents.supervisor import run_pipeline; run_pipeline('taxi')"

# 5. Dashboard
streamlit run dashboard/app.py
```

## Outputs

| File | Written by |
|---|---|
| `outputs/<dataset>/profile.json` | Data Agent |
| `outputs/<dataset>/cleaning_report.json` | `spark_jobs/clean.py` (rows removed at each step, with reasons) |
| `data/processed/<dataset>/` | cleaned Parquet |
| `outputs/<dataset>/eda_results.json` | EDA Agent |
| `outputs/<dataset>/model_results.json` | ML Agent |
| `outputs/<dataset>/insights.json` | Insight Agent |
| `outputs/<dataset>/benchmark_*.csv`, `scalability.png`, `benchmark_machine.json` | benchmark script |

## Tests

```powershell
python -m pytest
```

Tests cover cleaning, profiling, EDA and the Spark session (`tests/`). The EDA
results on the taxi data were also checked against pandas: row count, mean,
standard deviation and correlation all match.

## Benchmarks

```powershell
python -m spark_jobs.benchmark taxi
python -m spark_jobs.benchmark flights
```

Each run builds 0.25x, 1x and 2x versions of the dataset, then times the profile,
clean and EDA stages in Spark: one uncounted warm-up run, then 5 timed runs per
stage and size. Cleaning is also timed in pandas for comparison (flights stops at
1x because of memory). Results are written to `outputs/<dataset>/`:
`benchmark_results.csv` (every run), `benchmark_summary.csv` (mean, median, std),
`scalability.png` and `benchmark_machine.json`.

**Machine:** TODO: CPU, RAM, Spark driver memory 4g, Python 3.13.

**Median seconds, flights** (1.5M, 5.8M and 11.6M rows; 37, 143 and 285 MB):

| Stage | 0.25x | 1x | 2x |
|---|---|---|---|
| profile (Spark) | 2.56 | 4.04 | 6.15 |
| eda (Spark) | 1.30 | 2.35 | 3.51 |
| clean (Spark) | TODO | TODO | TODO |
| clean (pandas) | TODO | TODO | not run |

**Taxi** (0.9M, 3.7M and 7.4M rows; 20, 78 and 156 MB): re-run with the current
script and paste the median table here. TODO

**Reading the results:** Spark's runtime grows more slowly than the data does, but
at these sizes pandas is faster for cleaning because Spark has a high fixed cost
(start-up, planning, writing output). Spark would pay off on data too large for
pandas to load, which this laptop cannot test.

## Limitations

- The largest tested size is about 285 MB, well below the 1 GB and 5 GB targets,
  because of laptop memory.
- Timings come from one machine and vary slightly between runs. Medians are more
  reliable than means.
- Flights: `DEPARTURE_DELAY` is not marked as leakage. The team has to decide
  whether predictions are made before or after departure, and the decision goes in
  `docs/`.
- Cleaning drops cancelled and diverted flights and rows with missing delay values;
  the counts are in `cleaning_report.json`.
- TODO: ML and LLM limitations (Indrajith and Ardra).

## Team workflow

- No direct commits to `main`. Work on a branch (`feature/<name>-<topic>`), open a
  pull request, and get at least one review before merging.
- Never commit `.env`, API keys, raw datasets or `data/benchmark/`.
- Shared files (`README.md`, `requirements.txt`, `docs/contracts.md`): tell the
  group chat and pull `main` first.
- Announce any change to a JSON format and log it in `docs/contracts.md`.

## Demo path

1. Explain the problem and the architecture.
2. Select a dataset and run the pipeline.
3. Open the dashboard: dataset info, EDA, model performance.
4. Read the generated insights.
5. Show the scalability chart and the manual-effort comparison.
6. Run the second dataset to show the workflow is general.
7. State the limitations.