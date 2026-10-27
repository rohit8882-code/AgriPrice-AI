"""
Generates realistic multi-year Indian agricultural mandi datasets for benchmarking and development.
Simulates seasonality, weather shocks, arrival gluts, and data anomalies.
"""
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config.settings import settings, DATA_DIR
from database.db_manager import (
    init_db, SessionLocal, Commodity, Mandi, MandiPriceDaily, WeatherDaily, HAS_SQLALCHEMY
)

def generate_agricultural_dataset(start_date="2023-01-01", end_date="2026-09-25"):
    print("[*] Generating realistic Agmarknet & NASA-POWER agricultural dataset...")
    np.random.seed(42)
    random.seed(42)
    
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # Commodity specifications: base price (INR/Qtl), seasonality pattern, volatility
    commodity_specs = {
        "Onion": {
            "category": "Perishable",
            "base_price": 1800.0,
            "volatility": 0.35,
            "msp": None,
            # Kharif (Nov-Jan) and Rabi (Apr-Jun) harvest peaks
            "peak_months": [4, 5, 11, 12],
            "price_surge_months": [9, 10], # Pre-Kharif lean supply period
        },
        "Tomato": {
            "category": "Perishable",
            "base_price": 1500.0,
            "volatility": 0.45,
            "msp": None,
            "peak_months": [1, 2, 8, 9],
            "price_surge_months": [6, 7], # Summer extreme heat shock
        },
        "Potato": {
            "category": "Perishable",
            "base_price": 1200.0,
            "volatility": 0.20,
            "msp": None,
            "peak_months": [2, 3, 4],
            "price_surge_months": [10, 11],
        },
        "Wheat": {
            "category": "Non-Perishable",
            "base_price": 2275.0, # Near MSP
            "volatility": 0.10,
            "msp": 2275.0,
            "peak_months": [4, 5], # Post-Rabi harvest
            "price_surge_months": [1, 2],
        },
        "Soybean": {
            "category": "Non-Perishable",
            "base_price": 4600.0, # Near MSP
            "volatility": 0.15,
            "msp": 4600.0,
            "peak_months": [10, 11], # Kharif harvest
            "price_surge_months": [6, 7],
        },
        "Mustard": {
            "category": "Non-Perishable",
            "base_price": 5650.0, # Near MSP
            "volatility": 0.16,
            "msp": 5650.0,
            "peak_months": [3, 4], # Rabi harvest
            "price_surge_months": [9, 10],
        },
        "Rice": {
            "category": "Non-Perishable",
            "base_price": 2300.0, # Near MSP
            "volatility": 0.12,
            "msp": 2300.0,
            "peak_months": [10, 11, 12], # Post-Kharif harvest
            "price_surge_months": [6, 7],
        },
        "Cotton": {
            "category": "Commercial",
            "base_price": 7120.0, # Near MSP
            "volatility": 0.18,
            "msp": 7120.0,
            "peak_months": [11, 12, 1], # Winter harvest arrivals
            "price_surge_months": [5, 6],
        },
        "Maize": {
            "category": "Non-Perishable",
            "base_price": 2225.0, # Near MSP
            "volatility": 0.14,
            "msp": 2225.0,
            "peak_months": [10, 11, 3],
            "price_surge_months": [6, 7],
        },
        "Chilli": {
            "category": "Spice",
            "base_price": 16500.0,
            "volatility": 0.28,
            "msp": None,
            "peak_months": [2, 3, 4], # Guntur market peak
            "price_surge_months": [8, 9],
        },
        "Turmeric": {
            "category": "Spice",
            "base_price": 12800.0,
            "volatility": 0.25,
            "msp": None,
            "peak_months": [2, 3, 4], # Nizamabad harvest
            "price_surge_months": [9, 10],
        },
        "Gram": {
            "category": "Non-Perishable",
            "base_price": 5440.0, # Chana MSP
            "volatility": 0.14,
            "msp": 5440.0,
            "peak_months": [3, 4], # Rabi harvest
            "price_surge_months": [8, 9],
        },
        "Groundnut": {
            "category": "Commercial",
            "base_price": 6780.0, # Near MSP
            "volatility": 0.16,
            "msp": 6783.0,
            "peak_months": [10, 11, 12], # Kharif harvest
            "price_surge_months": [6, 7],
        },
        "Moong": {
            "category": "Non-Perishable",
            "base_price": 8680.0, # Green Gram MSP
            "volatility": 0.15,
            "msp": 8682.0,
            "peak_months": [9, 10], # Early Kharif harvest
            "price_surge_months": [4, 5],
        },
        "Garlic": {
            "category": "Perishable",
            "base_price": 9500.0,
            "volatility": 0.38,
            "msp": None,
            "peak_months": [2, 3, 4], # Spring harvest
            "price_surge_months": [10, 11],
        },
        "Coriander": {
            "category": "Spice",
            "base_price": 7600.0,
            "volatility": 0.22,
            "msp": None,
            "peak_months": [2, 3, 4], # Rabi harvest
            "price_surge_months": [9, 10],
        },
        "Bajra": {
            "category": "Non-Perishable",
            "base_price": 2625.0, # Pearl Millet MSP
            "volatility": 0.12,
            "msp": 2625.0,
            "peak_months": [10, 11], # Post-monsoon
            "price_surge_months": [6, 7],
        },
        "Jaggery": {
            "category": "Commercial",
            "base_price": 3900.0, # Gur / Jaggery
            "volatility": 0.15,
            "msp": None,
            "peak_months": [11, 12, 1, 2], # Winter cane season
            "price_surge_months": [7, 8],
        }
    }

    records = []
    weather_records = []
    
    # Pre-calculate weather per mandi per date
    for mandi_name, mandi_info in settings.SUPPORTED_MANDIS.items():
        lat = mandi_info["lat"]
        for current_date in dates:
            doy = current_date.dayofyear
            month = current_date.month
            
            # Monsoon rainfall (June - Sept)
            is_monsoon = 6 <= month <= 9
            rain_prob = 0.55 if is_monsoon else 0.08
            rainfall = np.random.exponential(18.0) if random.random() < rain_prob else 0.0
            
            # Extreme unseasonal event simulation (Western disturbance in March / late Oct)
            if (month == 3 and random.random() < 0.05) or (month == 10 and random.random() < 0.04):
                rainfall += np.random.uniform(35.0, 75.0)

            # Temperature cycles
            temp_max = 28.0 + 10.0 * np.sin(2 * np.pi * (doy - 100) / 365) + np.random.normal(0, 2)
            temp_min = temp_max - np.random.uniform(8.0, 15.0)
            soil_moisture = min(100.0, max(10.0, 20.0 + rainfall * 1.5 + np.random.normal(0, 3)))
            ndvi = 0.35 + 0.3 * np.sin(2 * np.pi * (doy - 200) / 365) + np.random.normal(0, 0.03)
            ndvi = max(0.1, min(0.9, ndvi))

            weather_records.append({
                "record_date": current_date.date(),
                "mandi_name": mandi_name,
                "rainfall_mm": round(rainfall, 2),
                "temp_max_c": round(temp_max, 1),
                "temp_min_c": round(temp_min, 1),
                "soil_moisture_pct": round(soil_moisture, 1),
                "ndvi_index": round(ndvi, 3)
            })

    weather_df = pd.DataFrame(weather_records)

    # Generate Mandi Prices & Arrivals
    for mandi_name, mandi_info in settings.SUPPORTED_MANDIS.items():
        for crop in mandi_info["primary_crops"]:
            if crop not in commodity_specs:
                continue
            spec = commodity_specs[crop]
            base_p = spec["base_price"]
            
            for current_date in dates:
                # Mandi is closed on Sundays (~14% of days)
                if current_date.weekday() == 6:
                    continue
                # 3% chance of mandi strike / local holiday
                if random.random() < 0.03:
                    continue

                doy = current_date.dayofyear
                month = current_date.month
                
                # Base seasonal arrival volume
                if spec["category"] == "Perishable":
                    base_arrival = 1200.0
                elif spec["category"] == "Spice":
                    base_arrival = 450.0
                elif spec["category"] == "Commercial":
                    base_arrival = 1800.0
                else:
                    base_arrival = 2500.0

                if month in spec["peak_months"]:
                    arrival_factor = np.random.uniform(2.2, 4.0) # Harvest Glut
                elif month in spec["price_surge_months"]:
                    arrival_factor = np.random.uniform(0.3, 0.6) # Lean Supply
                else:
                    arrival_factor = np.random.uniform(0.8, 1.4)
                
                arrivals = max(50.0, base_arrival * arrival_factor + np.random.normal(0, 100))
                
                # Price elasticity: Price is inversely related to arrivals + seasonal shifts
                elasticity = -0.35 if spec["category"] in ["Perishable", "Spice"] else -0.20
                arrival_effect = (arrivals / base_arrival) ** elasticity
                
                # Inflation trend (~5% per year)
                years_from_start = (current_date - dates[0]).days / 365.25
                inflation_drift = 1.0 + (0.05 * years_from_start)
                
                # Seasonal wave
                season_wave = 1.0 + (0.25 * np.sin(2 * np.pi * (doy - 240) / 365))
                noise = np.random.normal(1.0, spec["volatility"] * 0.15)
                
                # Unseasonal surge shock (e.g. Tomato July 2023 spike)
                shock = 1.0
                if crop == "Tomato" and current_date.year == 2023 and 6 <= month <= 8:
                    shock = 2.8
                elif crop == "Onion" and month in [9, 10] and random.random() < 0.15:
                    shock = 1.6

                modal_price = base_p * arrival_effect * season_wave * inflation_drift * noise * shock
                modal_price = max(200.0, modal_price)
                
                # Floor price if MSP exists (Government procurement support)
                if spec["msp"] and modal_price < spec["msp"] * 0.92:
                    modal_price = spec["msp"] * np.random.uniform(0.95, 1.05)

                min_price = modal_price * np.random.uniform(0.88, 0.96)
                max_price = modal_price * np.random.uniform(1.04, 1.15)

                # Inject 0.5% realistic data entry typo errors (e.g. ₹200 instead of ₹2000) to test cleaning
                if random.random() < 0.004:
                    modal_price = modal_price / 10.0 # Typo: missed zero
                elif random.random() < 0.003:
                    min_price = max_price * 1.2 # Inverted min/max entry error

                records.append({
                    "record_date": current_date.strftime("%Y-%m-%d"),
                    "mandi_name": mandi_name,
                    "commodity": crop,
                    "arrivals_qtl": round(arrivals, 1),
                    "min_price": round(min_price, 2),
                    "max_price": round(max_price, 2),
                    "modal_price": round(modal_price, 2)
                })

    df_prices = pd.DataFrame(records)
    
    # Save CSV files to disk
    prices_csv_path = DATA_DIR / "sample_mandi_prices.csv"
    weather_csv_path = DATA_DIR / "sample_weather.csv"
    df_prices.to_csv(prices_csv_path, index=False)
    weather_df.to_csv(weather_csv_path, index=False)
    print(f"[+] Saved {len(df_prices)} price records to {prices_csv_path}")
    print(f"[+] Saved {len(weather_df)} weather records to {weather_csv_path}")

    # Seed Database
    seed_database(df_prices, weather_df, commodity_specs)
    return df_prices, weather_df


def seed_database(df_prices, df_weather, commodity_specs, refresh_existing: bool = True):
    print("[*] Initializing SQLite / TimescaleDB and seeding tables...")
    init_db()
    if not HAS_SQLALCHEMY or SessionLocal is None:
        print("[!] Database engine not available. Skipping database seeding.")
        return

    session = SessionLocal()
    try:
        if refresh_existing:
            # Clear old records to allow full fresh seed with new crops & mandis
            session.query(MandiPriceDaily).delete()
            session.query(WeatherDaily).delete()
            session.commit()

        # 1. Commodities
        comm_map = {}
        for name, spec in commodity_specs.items():
            comm = session.query(Commodity).filter_by(name=name).first()
            if not comm:
                comm = Commodity(name=name, category=spec["category"], msp_price=spec["msp"])
                session.add(comm)
                session.flush()
            comm_map[name] = comm.commodity_id

        # 2. Mandis
        mandi_map = {}
        for name, info in settings.SUPPORTED_MANDIS.items():
            mandi = session.query(Mandi).filter_by(name=name).first()
            if not mandi:
                mandi = Mandi(
                    name=name, state=info["state"], district=info["district"],
                    latitude=info["lat"], longitude=info["lon"]
                )
                session.add(mandi)
                session.flush()
            mandi_map[name] = mandi.mandi_id

        session.commit()
        print(f"[+] Seeded {len(comm_map)} commodities and {len(mandi_map)} mandis.")

        # 3. Seed prices
        price_count = session.query(MandiPriceDaily).count()
        if price_count == 0:
            print("[*] Populating daily mandi price records...")
            price_mappings = []
            for _, row in df_prices.iterrows():
                m_id = mandi_map.get(row["mandi_name"])
                c_id = comm_map.get(row["commodity"])
                if m_id and c_id:
                    rec_date = (
                        datetime.strptime(str(row["record_date"]), "%Y-%m-%d").date()
                        if isinstance(row["record_date"], str)
                        else row["record_date"]
                    )
                    if hasattr(rec_date, "date"):
                        rec_date = rec_date.date()
                    price_mappings.append({
                        "record_date": rec_date,
                        "mandi_id": m_id,
                        "commodity_id": c_id,
                        "arrivals_qtl": float(row["arrivals_qtl"]),
                        "min_price": float(row["min_price"]),
                        "max_price": float(row["max_price"]),
                        "modal_price": float(row["modal_price"]),
                        "data_source": "AGMARKNET"
                    })
            if price_mappings:
                session.bulk_insert_mappings(MandiPriceDaily, price_mappings)
                session.commit()
                print(f"[+] Populated {len(price_mappings)} mandi price records.")

        # 4. Seed weather if table is empty
        weather_count = session.query(WeatherDaily).count()
        if weather_count == 0:
            print("[*] Populating daily weather records...")
            weather_mappings = []
            for _, row in df_weather.iterrows():
                m_id = mandi_map.get(row["mandi_name"])
                if m_id:
                    rec_date = (
                        datetime.strptime(str(row["record_date"]), "%Y-%m-%d").date()
                        if isinstance(row["record_date"], str)
                        else row["record_date"]
                    )
                    if hasattr(rec_date, "date"):
                        rec_date = rec_date.date()
                    weather_mappings.append({
                        "record_date": rec_date,
                        "mandi_id": m_id,
                        "rainfall_mm": float(row["rainfall_mm"]),
                        "temp_max_c": float(row["temp_max_c"]),
                        "temp_min_c": float(row["temp_min_c"]),
                        "soil_moisture_pct": float(row["soil_moisture_pct"]),
                        "ndvi_index": float(row["ndvi_index"])
                    })
            if weather_mappings:
                session.bulk_insert_mappings(WeatherDaily, weather_mappings)
                session.commit()
                print(f"[+] Populated {len(weather_mappings)} weather records.")

    except Exception as e:
        session.rollback()
        print(f"[-] Database seed error: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    generate_agricultural_dataset()
