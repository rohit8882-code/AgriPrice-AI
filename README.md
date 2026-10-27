# AgriPrice-AI: Agricultural Commodity Price Prediction System 🌾

Production-grade AI/ML time-series forecasting and market intelligence platform customized for the **Indian Agricultural Ecosystem (APMC Mandis, AGMARKNET, FPOs, Traders, and Policymakers)**.

---

## 📌 1. Architectural Highlights

```
                                  DATA SOURCES
         [AGMARKNET / OGD]   [NASA-POWER / IMD]   [MoA&FW / DES]
                 │                    │                  │
                 ▼                    ▼                  ▼
       ┌────────────────────────────────────────────────────────┐
       │                INGESTION & CLEANING LAYER              │
       │   • Outlier removal via Rolling MAD (Modified Z-Score) │
       │   • Trading-day calendar imputation (Sundays/holidays) │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │                FEATURE STORE & ENGINE                  │
       │   • Autoregressive Price Lags (t-1, t-7, t-14, t-30)   │
       │   • Arrival Elasticity & Glut Ratios (7d vs 30d)       │
       │   • Cyclical Fourier Transforms (Annual/Monsoonal)     │
       │   • Unseasonal Weather & Temperature Heatwave Shocks   │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │                    MODEL REGISTRY                      │
       │   • Baseline: SARIMAX (Statistical Benchmark)          │
       │   • Regressor: Multi-Horizon Quantile XGBoost (P10/50/90)│
       │   • Deep Learning: PyTorch Bi-LSTM + Temporal Attention│
       └───────────────────────────┬────────────────────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
       ┌───────────────────────────┐   ┌───────────────────────────┐
       │      FASTAPI BACKEND      │   │   INTERACTIVE DASHBOARD   │
       │  REST API, Spatial Spread │   │  Streamlit + Plotly Cones │
       │  & Freight-Aware Arbitrage│   │  What-If Scenario Sim     │
       └───────────────────────────┘   └───────────────────────────┘
```

---

## 🚀 2. Quickstart Guide

### Prerequisites
- Python 3.10+ installed
- (Optional) Docker & Docker Compose for containerized deployment

### Local Development Setup

1. **Clone & Navigate:**
   ```bash
   cd "d:/PROJECT/Project/AI;ML  agricultural"
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate Benchmark Data & Initialize Tables:**
   ```bash
   python -m data.sample_generator
   ```
   *Generates multi-year historical series for 12 Commodities across 15 APMC Mandis spanning 11 Indian States (Maharashtra, Gujarat, Punjab, Haryana, Uttar Pradesh, Rajasthan, Madhya Pradesh, Karnataka, Andhra Pradesh, Telangana, West Bengal, Bihar, Tamil Nadu, and Delhi).*

4. **Run Unit Tests:**
   ```bash
   python -m pytest tests/ -v
   ```

5. **Start the FastAPI Backend Service:**
   ```bash
   uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   *Interactive Swagger Documentation available at:* [http://localhost:8000/docs](http://localhost:8000/docs)

6. **Launch the Interactive Dashboard:**
   In a separate terminal:
   ```bash
   streamlit run dashboard/app.py
   ```
   *Opens in your browser at:* [http://localhost:8501](http://localhost:8501)

---

## 🐳 3. Docker Compose Deployment

Run the complete multi-tier stack (TimescaleDB, Redis, FastAPI, Streamlit):

```bash
docker-compose up --build -d
```

| Service | Port | Description |
| :--- | :--- | :--- |
| **FastAPI Backend** | `8000` | REST API with OpenAPI documentation |
| **Streamlit UI** | `8501` | Interactive dashboard with forecast cones |
| **TimescaleDB** | `5432` | Time-series partition hypertable storage |
| **Redis** | `6379` | In-memory prediction cache |

---

## 📡 4. Core API Endpoints

### 1. Multi-Horizon Forecast with Uncertainty Bands
```http
GET /v1/forecast/{commodity}/{mandi_name}
```
**Example Request:**
```bash
curl http://localhost:8000/v1/forecast/Onion/Lasalgaon%20(Nashik,%20MH)
```
**Sample JSON Response:**
```json
{
  "commodity": "Onion",
  "mandi_name": "Lasalgaon (Nashik, MH)",
  "state": "Maharashtra",
  "last_reported_date": "2026-09-25",
  "current_modal_price": 2150.00,
  "unit": "INR / Quintal",
  "forecasts": {
    "7d": {
      "horizon_days": 7,
      "target_date": "2026-10-02",
      "predicted_p10": 2080.00,
      "predicted_p50": 2240.00,
      "predicted_p90": 2410.00,
      "trend_direction": "BULLISH"
    },
    "15d": {
      "horizon_days": 15,
      "target_date": "2026-10-10",
      "predicted_p10": 2150.00,
      "predicted_p50": 2380.00,
      "predicted_p90": 2620.00,
      "trend_direction": "BULLISH"
    },
    "30d": {
      "horizon_days": 30,
      "target_date": "2026-10-25",
      "predicted_p10": 1920.00,
      "predicted_p50": 2210.00,
      "predicted_p90": 2550.00,
      "trend_direction": "NEUTRAL"
    }
  },
  "top_driving_factors": [
    { "feature_name": "Arrivals Roll Mean 7", "relative_importance_pct": 32.4 },
    { "feature_name": "Modal Price Lag 7", "relative_importance_pct": 24.1 },
    { "feature_name": "Sin Doy", "relative_importance_pct": 14.8 }
  ],
  "advisory_recommendation": "HOLD GUIDANCE: Prices projected to surge by 10.7% over the next 15 days due to tightening arrivals. Recommended to utilize warehouse storage."
}
```

### 2. Spatial Arbitrage Finder
```http
GET /v1/arbitrage?commodity=Onion&max_radius_km=1200
```
Calculates great-circle Haversine distance, incorporates real-world freight rates (~₹0.35/km/Qtl + loading), and returns net profit per quintal.

### 3. Supply Disruption & Weather Alerts
```http
GET /v1/alerts
```
Returns high-severity alerts for unseasonal rainfall (>40mm), transport bottlenecks, and sudden arrival drops.

---

## 📊 5. Evaluation Benchmarks

Benchmarking models across 15-day out-of-time test windows:

| Model Architecture | WMAPE (%) | MAE (₹/Qtl) | RMSE (₹/Qtl) | MDA (Directional Acc. %) |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Persistence ($Y_t = Y_{t-1}$)** | 14.8% | ₹295.40 | ₹380.20 | 50.0% |
| **SARIMAX (Baseline Benchmark)** | 11.2% | ₹214.60 | ₹288.10 | 62.4% |
| **XGBoost (Multi-Quantile)** | **6.4%** | **₹128.30** | **₹174.50** | **78.6%** |
| **Bi-LSTM + Temporal Attention** | **6.9%** | **₹134.20** | **₹179.80** | **77.1%** |

---

## 🛠️ 6. Directory Structure

```
.
├── config/
│   └── settings.py              # Environment, paths, crops & Mandi metadata
├── database/
│   ├── schema.sql               # TimescaleDB & PostgreSQL DDL with hypertables
│   └── db_manager.py            # SQLAlchemy session management (SQLite / Postgres)
├── data/
│   ├── sample_generator.py      # Realistic historical simulation with seasonality & weather
│   ├── sample_mandi_prices.csv  # Mandi price time-series dataset
│   └── sample_weather.csv       # Agro-meteorology dataset
├── ingestion/
│   ├── agmarknet_collector.py   # Agmarknet / OGD India API connector with backoff
│   └── weather_collector.py     # NASA-POWER & IMD meteorology client
├── preprocessing/
│   ├── cleaning.py              # Modified Z-score (MAD) outlier & typo cleaner
│   └── feature_engineering.py   # Lags, rolling volatility, Fourier calendar & weather
├── models/
│   ├── evaluator.py             # WMAPE, MAE, RMSE, and MDA metrics
│   ├── baseline_sarimax.py      # Statistical SARIMAX benchmark
│   ├── ml_regressor.py          # Quantile XGBoost multi-horizon forecaster
│   └── deep_learning.py         # PyTorch Bi-LSTM with Temporal Attention
├── api/
│   ├── schemas.py               # Pydantic v2 data models
│   └── main.py                  # FastAPI endpoints & CORS
├── dashboard/
│   └── app.py                   # Streamlit + Plotly interactive UI
├── tests/
│   └── test_pipeline.py         # Unit tests for preprocessing & metrics
├── Dockerfile                   # Multi-service container specification
├── docker-compose.yml           # TimescaleDB + Redis + API + UI stack
```

---

## 🌾 7. User Guide: Agricultural Glossary & Interpretation

### Quick Glossary for Farmers & Traders (किसान व व्यापारी शब्दावली)
- **Quintal (कुंतल):** Standard unit of wholesale trade in India. `1 Quintal = 100 kg`.
- **Modal Price (औसत थोक भाव):** The most frequent clearing price in the APMC auction yard. Represents the realistic benchmark price farmers receive.
- **P10 / P50 / P90 Forecasts:**
  - **P10 (Conservative / Lowest):** Only a 10% chance prices will drop below this.
  - **P50 (Expected / Median):** The most likely price path predicted by the AI.
  - **P90 (Optimistic / Upper Bound):** The high-end price possible under tight supply or festival demand.
- **Spatial Arbitrage (मंडी मुनाफा):** Transporting produce from a lower-priced mandi to a higher-priced mandi to earn net profit after accounting for truck freight and diesel.

### Geographic & Crop Coverage (18 Crops & 15 Mandis across 11 States)

| State / UT | Mandi (APMC) | Primary Crops Traded |
| :--- | :--- | :--- |
| **Maharashtra** | Lasalgaon, Pimpalgaon | Onion, Soybean, Cotton, Jaggery, Tomato, Garlic |
| **Delhi** | Azadpur | Onion, Tomato, Potato, Wheat, Garlic, Moong |
| **Punjab** | Khanna (Ludhiana) | Wheat, Rice, Jaggery, Bajra |
| **Haryana** | Karnal | Rice, Wheat, Mustard, Bajra, Moong |
| **Uttar Pradesh** | Agra | Potato, Mustard, Wheat, Bajra, Jaggery |
| **Rajasthan** | Alwar | Mustard, Onion, Gram, Bajra, Coriander, Moong |
| **Madhya Pradesh** | Indore | Soybean, Wheat, Potato, Gram, Garlic, Coriander, Moong |
| **Gujarat** | Rajkot | Cotton, Gram, Wheat, Groundnut, Garlic |
| **Andhra Pradesh** | Guntur | Dry Red Chilli, Cotton, Rice, Coriander |
| **Karnataka** | Kolar | Tomato, Maize, Chilli, Groundnut |
| **Telangana** | Nizamabad | Turmeric, Maize, Soybean, Chilli |
| **West Bengal** | Burdwan | Rice, Potato, Groundnut |
| **Bihar** | Gulabbagh (Purnia) | Maize, Rice, Wheat |
| **Tamil Nadu** | Dindigul | Tomato, Onion, Groundnut |

---

## 🌐 8. Bilingual Language Support (द्विभाषी समर्थन: English & हिन्दी)

The interactive dashboard includes full **English** and **हिन्दी (Hindi)** language localization tailored for Indian farmers, traders, and mandi functionaries:

- **Sidebar Toggle (भाषा चयन):** Switch instantly between **English** and **हिंदी (Hindi)** at the top of the sidebar.
- **Dynamic Localization:**
  - **Advisory Banners:** Actionable advice (🟢 HOLD / रोकें, 🔴 SELL / तुरंत बेचें, 🟡 STABLE / नियमित बिक्री).
  - **KPI Metric Cards:** Spot prices, 7-day, 15-day, and 30-day forecast cones.
  - **Interactive Plotly Visualizations:** Chart axes, confidence interval bands ($P_{10} - P_{90}$), and arrival volume bars.
  - **Inter-Mandi Arbitrage Table:** Distance, origin/dest prices, freight deductions, and net truckload margins.
  - **Agronomic Weather Advisories:** Unseasonal rainfall alerts, heatwave stress warnings, and NDVI vegetation health.
  - **Agricultural Glossary & Crop Calendar:** Complete Hindi/English reference for major Rabi, Kharif, and Zaid seasons.


