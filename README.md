# AgriPrice-AI: Agricultural Commodity Price Prediction System 🌾

Production-grade AI/ML time-series forecasting and market intelligence platform customized for the **Indian Agricultural Ecosystem (APMC Mandis, AGMARKNET, FPOs, Traders, and Policymakers)**.

---

## 📌 1. Architectural Highlights

```
                                  DATA SOURCES
         [AGMARKNET / data.gov.in]    [NASA-POWER / IMD]    [MoA&FW / DES]
                     │                        │                    │
                     ▼                        ▼                    ▼
       ┌───────────────────────────────────────────────────────────────┐
       │                  INGESTION & CLEANING LAYER                   │
       │   • Outlier removal via Rolling MAD (Modified Z-Score)        │
       │   • Trading-day calendar imputation (Sundays/holidays)        │
       │   • Explicit Data Provenance Tagging (REAL vs SIMULATED)      │
       └───────────────────────────────┬───────────────────────────────┘
                                       │
                                       ▼
       ┌───────────────────────────────────────────────────────────────┐
       │                    FEATURE STORE & ENGINE                     │
       │   • Autoregressive Price Lags (t-1, t-7, t-14, t-30)          │
       │   • Arrival Elasticity & Glut Ratios (7d vs 30d)              │
       │   • Cyclical Fourier Transforms (Annual/Monsoonal)            │
       │   • Unseasonal Weather & Heatwave Shocks                      │
       └───────────────────────────────┬───────────────────────────────┘
                                       │
                                       ▼
       ┌───────────────────────────────────────────────────────────────┐
       │                        MODEL REGISTRY                         │
       │   • Baseline: SARIMAX (Statistical Benchmark)                 │
       │   • Regressor: Multi-Horizon Quantile XGBoost (P10/50/90)     │
       │   • Deep Learning: PyTorch Bi-LSTM + Temporal Attention       │
       └───────────────────────────────┬───────────────────────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
           ┌───────────────────────────┐   ┌───────────────────────────┐
           │      FASTAPI BACKEND      │   │   INTERACTIVE DASHBOARD   │
           │  REST API, Spatial Spread │   │  Streamlit + Plotly Cones │
           │  & Freight-Aware Arbitrage│   │  What-If Scenario Sim     │
           │  (Vercel Serverless Ready)│   │  Bilingual Localization   │
           └───────────────────────────┘   └───────────────────────────┘
```

---

## 📂 2. Repository Layout

```
.
├── src/
│   └── agriprice/
│       ├── api/              # FastAPI application, routes, and Pydantic schemas
│       ├── bot/              # Telegram advisory and alert bot
│       ├── config/           # Pydantic v2 settings and path resolutions
│       ├── dashboard/        # Streamlit interactive UI application
│       ├── data/             # Database session manager, loaders, and sample generators
│       ├── features/         # Time-series feature engineering and cyclical encoding
│       ├── i18n/             # Localization engine, locale JSONs, and Hindi names
│       ├── ingestion/        # Agmarknet and NASA POWER API connectors
│       ├── models/           # SARIMAX, XGBoost, and PyTorch Bi-LSTM architectures
│       ├── preprocessing/    # Anomaly detection, MAD outlier cleaning, calendar imputation
│       ├── services/         # Shared forecast, arbitrage, and alert services
│       └── utils/            # Structured logger and Haversine geospatial calculations
├── api/
│   └── index.py              # Vercel Serverless entrypoint
├── data/
│   ├── raw/                  # Raw ingestion caches (.gitignored)
│   ├── processed/            # Processed time-series tables (.gitignored)
│   └── reference/            # Authoritative reference master lists
├── models_store/             # Serialized model checkpoints (.gitignored)
├── reports/                  # Generated benchmark metrics and markdown summaries
├── requirements/
│   ├── base.txt              # Core runtime and serverless dependencies
│   └── dev.txt               # Testing, linting, and developer tooling
├── scripts/
│   ├── train.py              # Model training automation
│   └── evaluate.py           # Evaluation metric computation
├── tests/
│   ├── unit/                 # Unit tests for preprocessing, features, and models
│   └── integration/          # Integration tests for API endpoints
├── .github/workflows/ci.yml  # GitHub Actions linting and test workflow
├── Makefile                  # Developer workflow automation
├── pyproject.toml            # Packaging and tool configurations
├── vercel.json               # Vercel serverless deployment routing
└── .gitignore                # Production ignore rules
```

---

## 🚀 3. Quickstart Guide

### Prerequisites
- Python 3.10+ installed
- Node.js & npm (optional, for Vercel CLI)

### Local Setup
```bash
# 1. Clone repository
git clone https://github.com/rohit8882-code/AgriPrice-AI.git
cd AgriPrice-AI

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies and package in editable mode
make setup
```

### Running Tests and Linting
```bash
# Run pytest test suite
make test

# Run code linter
make lint

# Run code formatter
make format
```

### Starting the Applications
```bash
# Run FastAPI Backend (http://localhost:8000/docs)
make run-api

# Run Streamlit Dashboard (http://localhost:8501)
make run-ui
```

---

## ☁️ 4. Vercel Deployment

The FastAPI core is pre-configured for **Vercel Serverless Functions** (`@vercel/python`):

1. **Vercel Routing:** Handled via [vercel.json](file:///d:/PROJECT/Project/AI;ML%20%20agricultural/vercel.json) directing all requests to [api/index.py](file:///d:/PROJECT/Project/AI;ML%20%20agricultural/api/index.py).
2. **Deploy via Vercel CLI:**
   ```bash
   npx vercel
   ```
3. **Deploy via GitHub Integration:**
   - Push this repository to GitHub.
   - Go to [vercel.com/new](https://vercel.com/new) and import `AgriPrice-AI`.
   - Vercel will automatically detect `vercel.json` and deploy the FastAPI backend.

---

## 📊 5. Model Evaluation Benchmarks

*Generated automatically via `python scripts/evaluate.py` (30-day out-of-time test window on Onion @ Lasalgaon APMC):*

| Model Architecture | WMAPE (%) | MAE (₹/Qtl) | RMSE (₹/Qtl) | MDA (Directional Acc. %) |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Persistence** | 20.38% | ₹579.77 | ₹694.42 | 36.7% |
| **SARIMAX (Baseline)** | 29.07% | ₹827.14 | ₹867.28 | 43.3% |
| **XGBoost (Multi-Quantile)** | **6.32%** | **₹189.24** | **₹275.62** | **66.7%** |
| **Bi-LSTM + Temporal Attention** | **7.33%** | **₹208.59** | **₹267.02** | **86.7%** |

---

## 📜 License & Contribution

- **License:** [MIT License](file:///d:/PROJECT/Project/AI;ML%20%20agricultural/LICENSE)
- **Contributing:** Please refer to [CONTRIBUTING.md](file:///d:/PROJECT/Project/AI;ML%20%20agricultural/CONTRIBUTING.md)
