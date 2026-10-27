"""
Database connection and Session management supporting SQLite & PostgreSQL/TimescaleDB.
Provides graceful fallback if SQLAlchemy is not yet installed in the environment.
"""
from datetime import date
from config.settings import settings

try:
    from sqlalchemy import (
        create_engine, Column, Integer, String, Float, Date as SqlDate, DateTime,
        ForeignKey, Numeric, PrimaryKeyConstraint, func
    )
    from sqlalchemy.orm import declarative_base, sessionmaker, relationship

    Base = declarative_base()

    class Commodity(Base):
        __tablename__ = "commodities"
        commodity_id = Column(Integer, primary_key=True, autoincrement=True)
        name = Column(String(100), unique=True, nullable=False)
        category = Column(String(50), nullable=False)
        msp_price = Column(Numeric(10, 2), nullable=True)

    class Mandi(Base):
        __tablename__ = "mandis"
        mandi_id = Column(Integer, primary_key=True, autoincrement=True)
        name = Column(String(150), unique=True, nullable=False)
        state = Column(String(100), nullable=False)
        district = Column(String(100), nullable=False)
        latitude = Column(Float, nullable=False)
        longitude = Column(Float, nullable=False)

    class MandiPriceDaily(Base):
        __tablename__ = "mandi_prices_daily"
        record_date = Column(SqlDate, nullable=False)
        mandi_id = Column(Integer, ForeignKey("mandis.mandi_id"), nullable=False)
        commodity_id = Column(Integer, ForeignKey("commodities.commodity_id"), nullable=False)
        arrivals_qtl = Column(Float, default=0.0)
        min_price = Column(Float, nullable=False)
        max_price = Column(Float, nullable=False)
        modal_price = Column(Float, nullable=False)
        data_source = Column(String(50), default="AGMARKNET")

        __table_args__ = (
            PrimaryKeyConstraint("record_date", "mandi_id", "commodity_id"),
        )

    class WeatherDaily(Base):
        __tablename__ = "weather_daily"
        record_date = Column(SqlDate, nullable=False)
        mandi_id = Column(Integer, ForeignKey("mandis.mandi_id"), nullable=False)
        rainfall_mm = Column(Float, default=0.0)
        temp_max_c = Column(Float, nullable=True)
        temp_min_c = Column(Float, nullable=True)
        soil_moisture_pct = Column(Float, nullable=True)
        ndvi_index = Column(Float, nullable=True)

        __table_args__ = (
            PrimaryKeyConstraint("record_date", "mandi_id"),
        )

    class PriceForecast(Base):
        __tablename__ = "price_forecasts"
        forecast_date = Column(SqlDate, nullable=False)
        mandi_id = Column(Integer, ForeignKey("mandis.mandi_id"), nullable=False)
        commodity_id = Column(Integer, ForeignKey("commodities.commodity_id"), nullable=False)
        horizon_days = Column(Integer, nullable=False)
        predicted_p10 = Column(Float, nullable=False)
        predicted_p50 = Column(Float, nullable=False)
        predicted_p90 = Column(Float, nullable=False)
        model_name = Column(String(50), nullable=False)
        model_version = Column(String(20), default="v1.0")

        __table_args__ = (
            PrimaryKeyConstraint("forecast_date", "mandi_id", "commodity_id", "horizon_days", "model_name"),
        )

    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    HAS_SQLALCHEMY = True

except ImportError:
    HAS_SQLALCHEMY = False
    Base = None
    engine = None
    SessionLocal = None
    Commodity = None
    Mandi = None
    MandiPriceDaily = None
    WeatherDaily = None
    PriceForecast = None


def init_db():
    """Initializes tables in database if SQLAlchemy is installed."""
    if HAS_SQLALCHEMY and engine is not None:
        Base.metadata.create_all(bind=engine)
        print("[+] Initialized database schema.")
    else:
        print("[!] SQLAlchemy not installed in current environment. Using Parquet/CSV file storage.")

def get_db():
    """Dependency for API sessions."""
    if not HAS_SQLALCHEMY:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
