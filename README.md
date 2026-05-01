# Adamsmith

A small research/code repository for simulating and analyzing a **“Smithian” financial market**: a price process with **fundamental drift + mean reversion to a “natural price” path**, and **time‐varying volatility** modeled with a **GJR‑GARCH(1,1)** specification and **Student‑t shocks**.

Most of the work lives in Jupyter notebooks (`papercode.ipynb`, `papercode2.ipynb`) and a Python simulator script in `SMITH MARKET SIM/`.

---

## What’s in this repo

- **Monte Carlo market simulator (Python)**
  - `SMITH MARKET SIM/Hypothetical Smithian Market Simulator.py`
  - Simulates prices/returns over many runs and generates summary plots + saved outputs.
- **Notebooks**
  - `papercode.ipynb`, `papercode2.ipynb` — exploratory / paper-style analysis (recommended entry points).
- **Data / outputs**
  - `data_cache/` — cached data (if present/used by notebooks)
  - `output_figures_and_data/`, `output_figures_and_data2/` — generated figures/data
  - Root-level images like `empirical_return_histograms_comparison.png`
- **Misc**
  - `eodhdnames.py` — helper related to EODHD naming (data source utility)
  - `export pkl.py` — exporting helpers (pickle)

---

## Model overview (high level)

The simulator generates a **natural (fundamental) price path** with a target total return over the horizon, and then simulates a market price that:

- drifts at a baseline rate (`mu_base`)
- mean reverts toward the natural price path (`gamma_price_reversion`)
- has conditional variance evolving via **GJR‑GARCH(1,1)**:
  - leverage/asymmetry via `gamma_leverage`
- uses **Student‑t innovations** with degrees of freedom `nu` (fat tails)

The script compares two example scenarios:
- **Scenario A**: “more regulated / Smithian ideal”
- **Scenario B**: “less regulated / behavioral impact”

---

## Requirements

This project is Python-based and relies on common scientific libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scipy
- statsmodels

Install (recommended via a virtual environment):

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install numpy pandas matplotlib seaborn scipy statsmodels
```

If you want to run notebooks:

```bash
pip install jupyter
```

---

## How to run

### 1) Run the simulator script

From the repo root:

```bash
python "SMITH MARKET SIM/Hypothetical Smithian Market Simulator.py"
```

The script (by default) runs Monte Carlo simulations and:
- produces plots (price paths, return distributions, QQ plots, ACF of absolute returns, etc.)
- saves figures (`plot_*.png`)
- saves output arrays / CSVs (`output_*.npy`, `output_df_run_stats_*.csv`)

Note: it uses multiprocessing (`cpu_count() - 1` cores by default).

### 2) Explore the notebooks

Open Jupyter:

```bash
jupyter notebook
```

Then open:
- `papercode.ipynb`
- `papercode2.ipynb`

---

## Outputs

You’ll typically see:
- **plot_01**: mean price paths and percentile bands vs. the natural price path
- **plot_02**: aggregated daily return distributions
- **plot_03**: distribution of final prices
- **plot_04/05**: kurtosis & skewness distributions
- **plot_06**: average conditional volatility distribution
- **plot_07**: maximum drawdown distribution
- **plot_08/09**: mean ACF of absolute returns
- **plot_10/11**: QQ plots vs normal

The repo also includes pre-generated figures and data under `output_figures_and_data*/`.

---

## Reproducibility notes

- Randomness comes from Student‑t draws; for exact reproducibility you can add a fixed seed near the top of the script, e.g. `np.random.seed(0)`.
- Simulation horizon and number of runs are controlled in the script (e.g., `T_periods`, `N_runs`).

---

## Repository structure (abridged)

- `SMITH MARKET SIM/`
  - `Hypothetical Smithian Market Simulator.py`
  - `plot_*.png` (generated)
  - `output.txt`
- `papercode.ipynb`
- `papercode2.ipynb`
- `data_cache/`
- `output_figures_and_data/`
- `output_figures_and_data2/`
- `eodhdnames.py`
- `export pkl.py`

---

## License

No license file is currently included. If you intend others to reuse this work, add a `LICENSE` (e.g., MIT, BSD-3, Apache-2.0) and clarify citation/attribution expectations.

---

## Disclaimer

This repository is for research/experimentation and does **not** constitute financial advice. The model is a simplified simulation and may not reflect real market dynamics.
