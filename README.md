# FCL Freight Rate Forecasting Model
**Antarctica Logistics — Internal Use Only (Private Repository)**

A hybrid SARIMAX + XGBoost stacked forecasting system for Full Container Load (FCL) freight rates across 15 global trade lanes, with a live Streamlit dashboard featuring X-factor scenario controls.

---

## Repository Structure

```
FCL-forecasting-model/
│
├── app_antarctica.py          # Main Streamlit dashboard (frontend)
├── requirements.txt           # Python dependencies
│
├── data/                      # Processed data files
│   ├── exog_features_antarctica.parquet   # Exogenous factor matrix (88 months)
│   ├── antarctica_monthly_panel.parquet   # Rate panel (wide format, 15 lanes)
│   ├── antarctica_monthly_long.parquet    # Rate panel (long format)
│   ├── features_monthly.parquet           # Extended feature pipeline output
│   ├── features_monthly.csv               # Same as above (CSV)
│   ├── qualifying_lanes.json              # Lane eligibility metadata
│   └── demo_lanes.json                    # Demo lane subset
│
├── models/
│   └── antarctica/
│       ├── path1_final_models.pkl         # Path 1 trained bundles (SARIMAX+XGB, 15 lanes)
│       ├── path2_final_models.pkl         # Path 2 trained bundles (SARIMAX+XGB, 15 lanes)
│       └── final_models_antarctica.pkl    # Legacy combined bundle
│
├── outputs/
│   └── antarctica/
│       ├── path1_forecasts.json           # Path 1 static forecast outputs
│       ├── path2_forecasts.json           # Path 2 static forecast outputs
│       ├── path1_val_table.csv            # Path 1 validation results
│       ├── path2_val_table.csv            # Path 2 validation results
│       ├── path1_backtest_*.csv           # Per-lane backtest CSVs (Path 1)
│       ├── path2_backtest_*.csv           # Per-lane backtest CSVs (Path 2)
│       └── path_comparison.csv            # Path 1 vs Path 2 comparison
│
├── training/                  # Model training scripts
│   ├── train_path1_AC.py      # Train Path 1 (SARIMAX + XGBoost, full data + shock dummies)
│   ├── train_path2_BC.py      # Train Path 2 (SARIMAX + XGBoost, post-COVID data only)
│   ├── train_antarctica.py    # Legacy training script
│   └── validate_2025.py       # Out-of-sample validation (Jan 2025–Apr 2026)
│
└── docs/
    └── FCL_Forecast_Spec_v2.md   # Model specification document
```

> **Note:** `.pkl` model files are excluded from git tracking due to size. They must be generated locally by running the training scripts, or transferred separately.

---

## Model Architecture

### Two Parallel Paths

| | Path 1 (A+C) | Path 2 (B+C) |
|---|---|---|
| **Training window** | Jul 2019 – Dec 2024 | Jan 2021 – Dec 2024 |
| **Includes shock dummies** | Yes (COVID, Supply Crunch, Ukraine, Red Sea) | No |
| **SARIMAX exog columns** | 12 | 10 |
| **XGBoost features** | 32 | 29 |
| **Best for** | Lanes with long history & structural breaks | Lanes with stable post-COVID patterns |

### Stacking Formula
```
Final Forecast = SARIMAX Prediction + XGBoost Residual Correction (capped at ±30%)
```

---

## Exogenous Factors

| Factor | Source | Frequency |
|--------|--------|-----------|
| Brent Crude Oil (USD/bbl) | FRED API — `DCOILBRENTEU` | Monthly |
| USD/CNY Exchange Rate | FRED API — `DEXCHUS` | Monthly |
| US Industrial Production Index | FRED API — `INDPRO` | Monthly |
| Chicago Fed National Activity Index (CFNAI) | FRED API — `CFNAI` | Monthly |
| China Total Exports (USD) | OECD SDMX REST API | Monthly |
| BDRY ETF — Dry Bulk Proxy | Yahoo Finance (yfinance) | Monthly |
| Event Dummies (COVID, Supply Crunch, Ukraine, Red Sea, Hormuz) | Hand-coded binary indicators | Monthly |

---

## Dashboard (Streamlit)

### Running Locally
```bash
pip install -r requirements.txt
streamlit run app_antarctica.py
```

### X-Factor Controls
- **Geopolitical toggles:** Red Sea, Hormuz, Ukraine, Panama Canal — each with lane-specific impact scales
- **Macro sliders:** Brent Crude ($40–$160), USD/CNY (6.0–9.0), BDRY ETF (3–50)
- **Developer mode:** Per-factor coefficient multipliers for sensitivity analysis

---

## Lanes Covered (15 Total)

| Region | Lane Codes |
|--------|-----------|
| East Asia | CNSHA, JPTYO, KRPUS |
| South/SE Asia | INNSA, THBKK |
| Europe | DEHAM, NLRTM, GBFXT |
| Americas | USNYC, USLAX, ARBUE |
| Middle East | AEJEA, QAHMD |
| Africa | NGLOS |
| Oceania | AUSYD |

---

## Data Date Range
- **Exogenous features:** January 2019 – April 2026 (88 monthly observations)
- **Training window:** July 2019 – December 2024
- **Validation window:** January 2025 – April 2026 (out-of-sample)

---

*Internal use only — Antarctica Logistics*
