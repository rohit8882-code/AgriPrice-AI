-- ==============================================================================
-- AgriPrice-AI: TimescaleDB & PostgreSQL Schema
-- High-throughput, partition-optimized database layout for Indian Mandis & Weather
-- ==============================================================================

-- 1. Enable TimescaleDB Extension if available
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- 2. Master Table: Commodities
CREATE TABLE IF NOT EXISTS commodities (
    commodity_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    category VARCHAR(50) NOT NULL, -- Perishable / Non-Perishable
    msp_price NUMERIC(10, 2),      -- Minimum Support Price in INR / Quintal
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Master Table: Mandis (APMCs)
CREATE TABLE IF NOT EXISTS mandis (
    mandi_id SERIAL PRIMARY KEY,
    name VARCHAR(150) UNIQUE NOT NULL,
    state VARCHAR(100) NOT NULL,
    district VARCHAR(100) NOT NULL,
    latitude NUMERIC(9, 6) NOT NULL,
    longitude NUMERIC(9, 6) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Hypertable: Daily Mandi Arrivals & Prices
CREATE TABLE IF NOT EXISTS mandi_prices_daily (
    record_date DATE NOT NULL,
    mandi_id INT REFERENCES mandis(mandi_id) ON DELETE CASCADE,
    commodity_id INT REFERENCES commodities(commodity_id) ON DELETE CASCADE,
    arrivals_qtl NUMERIC(12, 2) DEFAULT 0.0,
    min_price NUMERIC(10, 2) NOT NULL,
    max_price NUMERIC(10, 2) NOT NULL,
    modal_price NUMERIC(10, 2) NOT NULL,
    data_source VARCHAR(50) DEFAULT 'AGMARKNET',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (record_date, mandi_id, commodity_id)
);

-- Convert to TimescaleDB Hypertable partitioned by record_date (1-month chunks)
-- SELECT create_hypertable('mandi_prices_daily', 'record_date', chunk_time_interval => INTERVAL '1 month', if_not_exists => TRUE);

-- 5. Hypertable: Daily Microclimate & Agro-Meteorology (NASA-POWER / IMD)
CREATE TABLE IF NOT EXISTS weather_daily (
    record_date DATE NOT NULL,
    mandi_id INT REFERENCES mandis(mandi_id) ON DELETE CASCADE,
    rainfall_mm NUMERIC(8, 2) DEFAULT 0.0,
    temp_max_c NUMERIC(5, 2),
    temp_min_c NUMERIC(5, 2),
    soil_moisture_pct NUMERIC(5, 2),
    ndvi_index NUMERIC(4, 3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (record_date, mandi_id)
);

-- SELECT create_hypertable('weather_daily', 'record_date', chunk_time_interval => INTERVAL '1 month', if_not_exists => TRUE);

-- 6. Hypertable: Generated Predictions & Quantile Intervals
CREATE TABLE IF NOT EXISTS price_forecasts (
    forecast_date DATE NOT NULL,      -- Target future date
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    mandi_id INT REFERENCES mandis(mandi_id) ON DELETE CASCADE,
    commodity_id INT REFERENCES commodities(commodity_id) ON DELETE CASCADE,
    horizon_days INT NOT NULL,        -- 7, 15, or 30 days
    predicted_p10 NUMERIC(10, 2) NOT NULL,
    predicted_p50 NUMERIC(10, 2) NOT NULL, -- Point estimate
    predicted_p90 NUMERIC(10, 2) NOT NULL,
    model_name VARCHAR(50) NOT NULL, -- 'XGBoost', 'Bi-LSTM', 'SARIMAX', 'Ensemble'
    model_version VARCHAR(20) NOT NULL,
    PRIMARY KEY (forecast_date, mandi_id, commodity_id, model_name, generated_at)
);

-- 7. High-Performance Indices for Spatio-Temporal Queries
CREATE INDEX IF NOT EXISTS idx_mandi_prices_lookup 
ON mandi_prices_daily (commodity_id, mandi_id, record_date DESC);

CREATE INDEX IF NOT EXISTS idx_weather_lookup 
ON weather_daily (mandi_id, record_date DESC);

CREATE INDEX IF NOT EXISTS idx_forecast_lookup 
ON price_forecasts (commodity_id, mandi_id, forecast_date ASC);
