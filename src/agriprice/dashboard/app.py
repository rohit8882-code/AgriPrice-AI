"""
AgriPrice-AI: Interactive AgTech Commodity Intelligence & Price Prediction Dashboard.
Streamlit application featuring forecast cones, arrival elasticity, spatial arbitrage, and simulation.
Supports full bilingual English and Hindi (द्विभाषी - अंग्रेजी एवं हिंदी) localization.
Designed for Farmers (किसान), Traders (व्यापारी), FPOs, and AgTech Analysts.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="AgriPrice-AI | Indian Mandi Price Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling with Rich Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #0b1120;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #475569;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 700;
        margin-top: 4px;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 6px;
    }
    .bullish { color: #10b981; }
    .bearish { color: #f43f5e; }
    .neutral { color: #38bdf8; }
    
    .advisory-card {
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 24px;
        border: 1px solid;
    }
    .advisory-hold {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.2) 100%);
        border-color: #10b981;
    }
    .advisory-sell {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.15) 0%, rgba(136, 19, 55, 0.2) 100%);
        border-color: #f43f5e;
    }
    .advisory-stable {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(12, 74, 110, 0.2) 100%);
        border-color: #38bdf8;
    }
    
    .glossary-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .tag-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(148, 163, 184, 0.15);
        color: #cbd5e1;
        margin-right: 6px;
    }
    .lang-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid #0284c7;
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .alert-banner {
        border-left: 4px solid #ef4444;
        background: rgba(239, 68, 68, 0.12);
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Import backend modules
from agriprice.config.settings import settings, DATA_DIR
from agriprice.preprocessing.cleaning import detect_and_clean_outliers, impute_missing_trading_days
from agriprice.features.feature_engineering import create_agricultural_features
from agriprice.models.ml_regressor import AgriculturalXGBoostForecaster
from agriprice.api.main import find_spatial_arbitrage, get_market_alerts

# ==========================================
# BILINGUAL LOCALIZATION & CROP METADATA
# ==========================================
TRANSLATIONS = {
    "en": {
        "lang_name": "English",
        "app_title": "AgriPrice-AI",
        "app_subtitle": "Indian Mandi Commodity Price Prediction & Market Intelligence",
        "language_box_title": "🌐 Language Selection",
        "choose_lang": "Choose Language / भाषा चुनें",
        "select_state_mandi": "📍 1. Select State & Mandi",
        "filter_state": "Filter by State",
        "all_states": "All States",
        "select_mandi": "Select Mandi / APMC",
        "select_crop": "🌱 2. Select Crop",
        "commodity": "Commodity / Crop",
        "quick_glossary": "💡 Quick Glossary (Understand Terms)",
        "glossary_content": """
        - **1 Quintal** = 100 Kilograms (kg)
        - **Modal Price** = The rate at which the highest volume was traded today.
        - **Arrivals** = Total crop volume brought to mandi by farmers today.
        - **MSP** = Minimum Support Price guaranteed by Government of India.
        - **P10 / P50 / P90** = Conservative / Most Likely / Optimistic price forecasts.
        """,
        "market_sync": "Last Market Sync",
        "msp_label": "MSP",
        "msp_na": "MSP: Not Applicable",
        "hold_title": "🟢 HOLD RECOMMENDATION",
        "hold_text": "Prices for **{commodity}** at **{mandi_name}** are projected to rise by **+{pct_15d:.1f}% (+₹{diff:,.0f}/Quintal)** over the next 15 days due to tightening market arrivals. If you have safe storage or warehouse access, holding your produce can yield substantially higher earnings.",
        "sell_title": "🔴 SELL PROMPTLY",
        "sell_text": "Prices for **{commodity}** at **{mandi_name}** are expected to decline by **{pct_15d:.1f}% (-₹{diff:,.0f}/Quintal)** as heavy harvest arrivals enter the market. Liquidating ready harvest immediately is recommended to avoid price drop losses.",
        "stable_title": "🟡 STABLE MARKET",
        "stable_text": "Prices for **{commodity}** at **{mandi_name}** are expected to remain steady within **±{pct_15d:.1f}%** over the next fortnight. Routine selling at the prevailing spot rate is recommended.",
        "kpi_spot_rate": "Current Spot Rate",
        "kpi_per_qtl": "/ Qtl",
        "kpi_rate_for_100kg": "Rate for 100 kg at {mandi}",
        "kpi_7d_forecast": "7-Day Forecast",
        "kpi_15d_forecast": "15-Day Harvest Window",
        "kpi_30d_forecast": "30-Day Outlook",
        "range_label": "Range",
        "tab1_title": "📈 1. Price Forecast & Trends",
        "tab2_title": "🚛 2. Inter-Mandi Transport Profit",
        "tab3_title": "🌦️ 3. Weather Impact",
        "tab4_title": "🧪 4. What-If Simulator",
        "tab5_title": "📖 5. User Guide & Glossary",
        "chart_title": "📈 30-Day Price Forecast Cone for {commodity}",
        "chart_help_title": "ℹ️ How to read this chart?",
        "chart_help_body": """
        - **Solid Blue Line (Actual Price):** The real market rates recorded over the past 4 months.
        - **Dashed Green Line (Predicted Median P50):** The most likely price path computed by the AI model.
        - **Shaded Green Area (P10 to P90 Range):** The confidence interval. In 80% of historical conditions, the actual price falls within this band.
        - **Gray Vertical Bars (Daily Arrivals):** The volume of crop brought to the market in Quintals. When arrival bars are very tall, prices usually face downward pressure.
        """,
        "trace_arrivals": "Market Arrivals (Quintals)",
        "trace_actual": "Actual Modal Price (₹/Qtl)",
        "trace_p90": "Optimistic Upper Price (P90)",
        "trace_band": "Expected Price Range (P10 - P90)",
        "trace_p50": "Predicted Expected Price (P50)",
        "yaxis_price": "Price (₹ / Quintal)",
        "yaxis_arrivals": "Daily Arrivals (Quintals)",
        "why_moving": "🧠 Why is the price moving?",
        "feature_weight_title": "Relative Influence Weight (%)",
        "top_factors_title": "Top Factors for {commodity}:",
        "top_factor_1": "**Arrival Momentum:** Sudden surges in mandi arrivals immediately soften wholesale bids.",
        "top_factor_2": "**Calendar & Seasonality:** Annual sowing and harvest cycles create predictable supply surges and lean periods.",
        "top_factor_3": "**Weather Shocks:** Unseasonal rainfall disrupts transport and damages field crops, triggering short-term price spikes.",
        "table_forecast_expander": "📋 View Day-by-Day Forecast Data Table (30 Days)",
        "col_day": "Day",
        "col_date": "Date",
        "col_p10": "Conservative Price (₹/Qtl)",
        "col_p50": "Expected Price (₹/Qtl)",
        "col_p90": "Optimistic Price (₹/Qtl)",
        "col_change": "Change from Today",
        "arbitrage_title": "🚛 Inter-Mandi Arbitrage Opportunities for {commodity}",
        "arbitrage_desc": """
        **What is Inter-Mandi Arbitrage?**
        If the wholesale price of your crop in a distant city is much higher than your local Mandi, you or your FPO can transport produce by truck. If the price difference is higher than the truck freight & diesel cost, you earn direct additional profit.
        """,
        "col_origin_mandi": "Origin Mandi",
        "col_dest_mandi": "Destination Mandi",
        "col_distance": "Distance",
        "col_origin_price": "Origin Price",
        "col_dest_price": "Dest. Price",
        "col_spread": "Gross Spread",
        "col_freight": "Estimated Freight",
        "col_net_profit": "Net Profit / Qtl",
        "col_truckload": "Profit per 10-Tonne Truck",
        "top_route_title": "🏆 Top Route Recommendation:",
        "transport_from_to": "Transport **{commodity}** from **{origin}** ({origin_price}/Qtl) to **{dest}** ({dest_price}/Qtl).",
        "freight_toll": "Freight & Toll Cost:",
        "net_profit_banner": "Net Clear Profit:",
        "per_quintal": "per quintal",
        "per_truck": "per 10-tonne truck!",
        "no_arbitrage": "No inter-mandi price spreads currently exceed freight costs for {commodity}. Local selling is currently most cost-effective.",
        "weather_title": "🌦️ Microclimate & Weather Feed for {mandi_name}",
        "weather_sub": "Synchronized daily weather indicators and their direct agronomic impact on crop trade.",
        "w_rainfall": "24-Hour Rainfall",
        "w_temp": "Max Temperature",
        "w_moisture": "Soil Moisture",
        "w_ndvi": "Vegetation Health (NDVI)",
        "w_rain_alert": "🌧️ Heavy Rain Alert",
        "w_rain_ok": "☀️ Favorable Harvest Weather",
        "w_heat_alert": "🔥 High Heat Shock",
        "w_heat_ok": "🌤️ Normal Thermal Range",
        "w_moisture_sub": "Adequate root-zone moisture",
        "w_ndvi_sub": "Healthy canopy vigor",
        "agronomic_advisory": "🌾 Agronomic Advisory for Current Weather",
        "adv_rain": "⚠️ **Precipitation Alert:** Heavy rainfall ({val:.1f} mm) recorded around {mandi}. Mandi arrivals may drop over the next 2-3 days as farm transport halts. Temporary price firming expected.",
        "adv_heat": "🌡️ **Heat Stress Advisory:** High temperature ({val:.1f} °C) accelerates moisture loss for perishables (Tomato, Onion). Ensure proper ventilation during transport to prevent rotting.",
        "adv_clear": "✅ **Clear Weather:** Microclimate around {mandi} is stable. Harvesting, drying, and open-truck transportation can proceed smoothly.",
        "sim_title": "🧪 What-If Market Scenario Simulator",
        "sim_desc": "Explore how sudden climate events, bumper harvests, or government export bans would alter your crop's prices over the next 15 days.",
        "sim_rain_label": "Unseasonal Rainfall Shock (+mm)",
        "sim_rain_help": "Simulate sudden downpours disrupting harvest and mandi arrivals",
        "sim_glut_label": "Arrival Supply Shock (%)",
        "sim_glut_help": "Negative for shortage/drought, Positive for bumper harvest glut",
        "sim_export_label": "Government Export Ban / Stock Limit Imposed",
        "sim_export_help": "Restricts border movement and forces produce into domestic market",
        "sim_fuel_label": "Diesel Price Shift (%)",
        "sim_fuel_help": "Alters long-distance truck freight costs",
        "sim_baseline_title": "Standard Baseline 15d Price",
        "sim_shock_title": "Simulated Shock 15d Price",
        "sim_export_alert": "📉 **Export Ban Impact:** Imposing export bans creates immediate domestic gluts. Prices drop by ~15%, squeezing farmer margins while cooling consumer retail inflation.",
        "sim_rain_alert": "📈 **Rainfall Shock Impact:** Heavy unseasonal rains damage produce on fields and block mandi approach roads, leading to rapid price appreciation of 12-18%.",
        "guide_title": "📖 Farmer & Trader Guide: Understanding Agricultural Markets",
        "guide_terms_title": "🌾 Key Agricultural Terms",
        "guide_terms_content": """
        - **Quintal:** Standard unit of agricultural trade in India. 1 Quintal = 100 kg = 0.1 Metric Tonne.
        - **Modal Price:** The most frequent clearing price in the APMC auction yard. This is the realistic benchmark rate farmers receive.
        - **Min & Max Price:** The lowest and highest prices paid on a day, depending on crop moisture, grade, and quality.
        - **MSP (Minimum Support Price):** Floor price guaranteed by the Government of India for major crops (Wheat, Paddy, Mustard, Soybean, Cotton, Gram, etc.).
        - **Arrivals:** The total quantity of produce unloaded in the mandi during trading hours.
        """,
        "guide_ai_title": "🤖 How Do the AI Forecasts Work?",
        "guide_ai_content": """
        AgriPrice-AI uses state-of-the-art machine learning models trained on multi-year Indian market records:
        - **Multi-Quantile XGBoost:** Predicts conservative (P10), expected (P50), and optimistic (P90) prices simultaneously.
        - **PyTorch Bi-LSTM with Temporal Attention:** Captures seasonal time dependencies and weather shock memory.
        - **Arrival Elasticity Modeling:** Inversely relates sudden arrival gluts to price drops.
        - **Trading Day Calendar:** Automatically accounts for Sundays, APMC holidays, and strike disruptions.
        """,
        "season_table_title": "🗓️ Indian Crop Seasons Quick Reference",
        "footer_text": "AgriPrice-AI Platform • Empowering Indian Farmers, Mandis & AgTech Enterprises with Open Market Intelligence",
        "status_quo": "Status Quo Outlook",
        "sim_shift": "Simulated Shift",
        "net_change": "Net change"
    },
    "hi": {
        "lang_name": "हिंदी (Hindi)",
        "app_title": "एग्रीप्राइस-एआई (AgriPrice-AI)",
        "app_subtitle": "भारतीय कृषि उपज मंडी मूल्य पूर्वानुमान एवं बाजार विश्लेषण",
        "language_box_title": "🌐 भाषा चयन (Language)",
        "choose_lang": "भाषा चुनें / Choose Language",
        "select_state_mandi": "📍 1. राज्य एवं मंडी चुनें",
        "filter_state": "राज्य अनुसार फ़िल्टर करें",
        "all_states": "सभी राज्य (All States)",
        "select_mandi": "कृषि उपज मंडी (APMC) चुनें",
        "select_crop": "🌱 2. फसल चुनें",
        "commodity": "फसल (कृषि उत्पाद)",
        "quick_glossary": "💡 त्वरित शब्दावली (शब्दावली समझें)",
        "glossary_content": """
        - **1 कुंतल (Quintal)** = 100 किलोग्राम (किग्रा)
        - **औसत भाव (Modal Price)** = वह भाव जिस पर आज मंडी में सबसे अधिक मात्रा में व्यापार हुआ।
        - **दैनिक आवक (Arrivals)** = आज किसानों द्वारा मंडी में लाई गई फसल की कुल मात्रा (कुंतल में)।
        - **न्यूनतम समर्थन मूल्य (MSP)** = सरकार द्वारा फसलों के लिए निर्धारित न्यूनतम गारंटीकृत मूल्य।
        - **P10 / P50 / P90** = न्यूनतम / संभावित औसत / उच्चतम भाव का AI पूर्वानुमान।
        """,
        "market_sync": "अंतिम मंडी अपडेट",
        "msp_label": "समर्थन मूल्य (MSP)",
        "msp_na": "MSP: लागू नहीं (बाजार आधारित)",
        "hold_title": "🟢 माल रोककर रखने की सलाह (HOLD)",
        "hold_text": "**{mandi_name}** में **{commodity}** के भाव अगले 15 दिनों में आवक कम होने के कारण **+{pct_15d:.1f}% (+₹{diff:,.0f}/कुंतल)** बढ़ने का अनुमान है। यदि आपके पास सुरक्षित गोदाम या भंडारण की सुविधा है, तो माल रोककर रखने से आपको काफी अधिक लाभ मिल सकता है।",
        "sell_title": "🔴 तुरंत बेचने की सलाह (SELL PROMPTLY)",
        "sell_text": "**{mandi_name}** में **{commodity}** की भारी आवक के कारण भाव में **{pct_15d:.1f}% (-₹{diff:,.0f}/कुंतल)** की गिरावट आने की संभावना है। कीमतों में गिरावट के नुकसान से बचने के लिए तैयार फसल को तुरंत मंडी में बेचना उचित रहेगा।",
        "stable_title": "🟡 सामान्य स्थिति - नियमित बिक्री (STABLE MARKET)",
        "stable_text": "**{mandi_name}** में **{commodity}** के भाव अगले पखवाड़े में **±{pct_15d:.1f}%** के दायरे में स्थिर रहने का अनुमान है। मौजूदा बाजार भाव पर सामान्य रूप से बिक्री जारी रखें।",
        "kpi_spot_rate": "आज का थोक भाव",
        "kpi_per_qtl": "/ कुंतल",
        "kpi_rate_for_100kg": "{mandi} में 100 किग्रा का भाव",
        "kpi_7d_forecast": "7 दिन का AI अनुमान",
        "kpi_15d_forecast": "15 दिन का अनुमान (कटाई विंडो)",
        "kpi_30d_forecast": "30 दिन का रुझान (मासिक आउटलुक)",
        "range_label": "अनुमानित दायरा",
        "tab1_title": "📈 1. मूल्य पूर्वानुमान एवं रुझान",
        "tab2_title": "🚛 2. मंडी परिवहन मुनाफा (आर्बिट्राज)",
        "tab3_title": "🌦️ 3. मौसम व फसल स्थिति",
        "tab4_title": "🧪 4. क्या होगा अगर... सिम्युलेटर",
        "tab5_title": "📖 5. किसान एवं व्यापारी गाइड",
        "chart_title": "📈 {commodity} के लिए 30-दिवसीय मूल्य पूर्वानुमान कोन",
        "chart_help_title": "ℹ️ यह चार्ट कैसे समझें?",
        "chart_help_body": """
        - **ठोस नीली रेखा (वास्तविक भाव):** पिछले 4 महीनों में दर्ज किए गए वास्तविक मंडी भाव।
        - **डैश हरी रेखा (संभावित औसत P50):** AI मॉडल द्वारा आकलित सबसे संभावित भाव दिशा।
        - **हल्का हरा क्षेत्र (P10 से P90 दायरा):** संभाव्यता बैंड। 80% परिस्थितियों में वास्तविक भाव इसी दायरे में रहता है।
        - **ग्रे वर्टिकल बार (दैनिक आवक):** मंडी में आई फसल की कुल मात्रा (कुंतल में)। आवक बढ़ने पर भाव पर सामान्यतः दबाव आता है।
        """,
        "trace_arrivals": "मंडी आवक (कुंतल)",
        "trace_actual": "वास्तविक औसत भाव (₹/कुंतल)",
        "trace_p90": "उच्चतम संभावित भाव (P90)",
        "trace_band": "संभावित मूल्य दायरा (P10 - P90)",
        "trace_p50": "अनुमानित औसत भाव (P50)",
        "yaxis_price": "भाव (₹ / कुंतल)",
        "yaxis_arrivals": "दैनिक आवक (कुंतल)",
        "why_moving": "🧠 भाव में उतार-चढ़ाव के मुख्य कारण",
        "feature_weight_title": "सापेक्ष प्रभाव भार (%)",
        "top_factors_title": "{commodity} के लिए मुख्य कारक:",
        "top_factor_1": "**आवक की गति (Arrival Momentum):** मंडी में आवक अचानक बढ़ने से थोक बोलियां तुरंत नरम पड़ जाती हैं।",
        "top_factor_2": "**त्यौहारी व फसल चक्र (Seasonality):** वार्षिक बुवाई और कटाई चक्र से आपूर्ति में अनुमानित उतार-चढ़ाव आता है।",
        "top_factor_3": "**मौसम के झटके (Weather Shocks):** बेमौसम बारिश से परिवहन बाधित होता है और खेतों में खड़ी फसल को नुकसान पहुंचता है।",
        "table_forecast_expander": "📋 30 दिनों का दैनिक पूर्वानुमान विवरण देखें (तालिका)",
        "col_day": "दिन",
        "col_date": "दिनांक",
        "col_p10": "न्यूनतम अनुमान (₹/कुंतल)",
        "col_p50": "संभावित भाव (₹/कुंतल)",
        "col_p90": "अधिकतम अनुमान (₹/कुंतल)",
        "col_change": "आज से बदलाव",
        "arbitrage_title": "🚛 {commodity} के लिए अन्य मंडियों में व्यापार व परिवहन मुनाफा",
        "arbitrage_desc": """
        **अंतर-मंडी परिवहन मुनाफा (Arbitrage) क्या है?**
        यदि किसी अन्य शहर या मंडी में आपकी फसल का थोक भाव आपकी स्थानीय मंडी से काफी अधिक है, तो आप या आपका किसान संगठन (FPO) ट्रक द्वारा माल वहां भेज सकते हैं। यदि भाव का अंतर ट्रक भाड़े व डीजल खर्च से अधिक है, तो आपको सीधा अतिरिक्त शुद्ध लाभ मिलता है।
        """,
        "col_origin_mandi": "खरीद मंडी (Origin)",
        "col_dest_mandi": "बिक्री मंडी (Destination)",
        "col_distance": "दूरी (Distance)",
        "col_origin_price": "खरीद भाव",
        "col_dest_price": "बिक्री भाव",
        "col_spread": "सकल अंतर",
        "col_freight": "अनुमानित भाड़ा",
        "col_net_profit": "शुद्ध लाभ / कुंतल",
        "col_truckload": "10 टन ट्रक पर शुद्ध बचत",
        "top_route_title": "🏆 सर्वश्रेष्ठ परिवहन मार्ग अनुशंसा:",
        "transport_from_to": "**{commodity}** को **{origin}** ({origin_price}/कुंतल) से **{dest}** ({dest_price}/कुंतल) ले जाएं।",
        "freight_toll": "भाड़ा एवं टोल खर्च:",
        "net_profit_banner": "स्पष्ट शुद्ध लाभ:",
        "per_quintal": "प्रति कुंतल",
        "per_truck": "प्रति 10-टन ट्रक!",
        "no_arbitrage": "{commodity} के लिए वर्तमान में कोई अंतर-मंडी मूल्य अंतर भाड़ा लागत से अधिक नहीं है। स्थानीय मंडी में बेचना ही सर्वाधिक लाभकारी है।",
        "weather_title": "🌦️ {mandi_name} के लिए स्थानीय मौसम एवं फसल स्थिति",
        "weather_sub": "दैनिक मौसम संकेतक और फसल व्यापार पर उनका सीधा कृषि-आर्थिक प्रभाव।",
        "w_rainfall": "24 घंटे की वर्षा",
        "w_temp": "अधिकतम तापमान",
        "w_moisture": "मिट्टी में नमी",
        "w_ndvi": "फसल हरियाली (NDVI)",
        "w_rain_alert": "🌧️ भारी बारिश की चेतावनी",
        "w_rain_ok": "☀️ कटाई हेतु अनुकूल मौसम",
        "w_heat_alert": "🔥 अत्यधिक गर्मी (हीट शॉक)",
        "w_heat_ok": "🌤️ सामान्य तापमान दायरा",
        "w_moisture_sub": "जड़ों के लिए पर्याप्त नमी उपलब्ध",
        "w_ndvi_sub": "फसल वृद्धि एवं हरियाली उत्तम",
        "agronomic_advisory": "🌾 वर्तमान मौसम के लिए कृषि सलाह",
        "adv_rain": "⚠️ **वर्षा चेतावनी:** {mandi} के आसपास भारी बारिश ({val:.1f} मिमी) दर्ज की गई है। खेतों से परिवहन रुकने से अगले 2-3 दिनों में मंडी आवक घट सकती है। अल्पकालिक मूल्य वृद्धि की संभावना।",
        "adv_heat": "🌡️ **लू व गर्मी की सलाह:** उच्च तापमान ({val:.1f} °C) से जल्द खराब होने वाली फसलों में नमी का नुकसान तेजी से होता है। परिवहन के दौरान उचित हवादार व्यवस्था रखें।",
        "adv_clear": "✅ **साफ मौसम:** {mandi} के आसपास स्थानीय मौसम स्थिर है। कटाई, सुखाने और खुले ट्रकों में माल परिवहन का कार्य सुचारू रूप से किया जा सकता है।",
        "sim_title": "🧪 क्या होगा अगर... बाजार परिदृश्य सिम्युलेटर (What-If Lab)",
        "sim_desc": "देखें कि अचानक बेमौसम बारिश, बंपर आवक, या सरकारी निर्यात प्रतिबंध अगले 15 दिनों में आपकी फसल के भाव कैसे बदल सकते हैं।",
        "sim_rain_label": "बेमौसम बारिश का झटका (+मिमी)",
        "sim_rain_help": "अचानक तेज बारिश से कटाई और आवक प्रभावित होने का परिदृश्य",
        "sim_glut_label": "मंडी आवक में आपूर्ति झटका (%)",
        "sim_glut_help": "माइनस (-) यानी फसल की कमी, प्लस (+) यानी बंपर आवक की भरमार",
        "sim_export_label": "सरकारी निर्यात प्रतिबंध / स्टॉक सीमा लागू",
        "sim_export_help": "सीमा पार व्यापार रोककर सारा माल घरेलू बाजार में लाने का प्रभाव",
        "sim_fuel_label": "डीजल के दामों में बदलाव (%)",
        "sim_fuel_help": "लंबी दूरी के ट्रक भाड़े पर प्रभाव डालता है",
        "sim_baseline_title": "सामान्य आधारभूत 15-दिवसीय भाव",
        "sim_shock_title": "परिदृश्य के बाद अनुमानित 15-दिवसीय भाव",
        "sim_export_alert": "📉 **निर्यात प्रतिबंध प्रभाव:** निर्यात पर रोक लगने से घरेलू बाजार में तुरंत माल की अधिकता हो जाती है। भाव ~15% तक गिर जाते हैं, जिससे किसानों के लाभ मार्जिन पर दबाव आता है।",
        "sim_rain_alert": "📈 **बारिश के झटके का प्रभाव:** बेमौसम भारी बारिश से खेतों में खड़ी फसल खराब होती है और मंडी के रास्ते अवरुद्ध होते हैं, जिससे 12-18% तक त्वरित मूल्य उछाल आता है।",
        "guide_title": "📖 किसान एवं व्यापारी गाइड: कृषि मंडी को समझें",
        "guide_terms_title": "🌾 मुख्य कृषि शब्दावली (Key Terms)",
        "guide_terms_content": """
        - **कुंतल (Quintal):** भारत में कृषि व्यापार की मानक इकाई। 1 कुंतल = 100 किलोग्राम = 0.1 मीट्रिक टन।
        - **औसत थोक भाव (Modal Price):** कृषि उपज मंडी नीलामी यार्ड में सर्वाधिक सौदों का निष्कासन मूल्य। यही किसानों को मिलने वाला वास्तविक मानक भाव है।
        - **न्यूनतम व अधिकतम भाव (Min & Max):** दिन में माल की गुणवत्ता, नमी और ग्रेडिंग के आधार पर लगने वाले सबसे कम व सबसे अधिक भाव।
        - **न्यूनतम समर्थन मूल्य (MSP):** प्रमुख फसलों के लिए भारत सरकार द्वारा घोषित गारंटीकृत न्यूनतम मूल्य।
        - **दैनिक आवक (Arrivals):** मंडी कार्य समय के दौरान अनलोड किए गए कुल कृषि उत्पाद की मात्रा।
        """,
        "guide_ai_title": "🤖 AI पूर्वानुमान मॉडल कैसे कार्य करते हैं?",
        "guide_ai_content": """
        एग्रीप्राइस-एआई भारतीय कृषि मंडियों के बहु-वर्षीय ऐतिहासिक आंकड़ों पर प्रशिक्षित उन्नत मशीन लर्निंग मॉडल का उपयोग करता है:
        - **मल्टी-क्वांटाइल XGBoost:** एक साथ न्यूनतम (P10), औसत (P50) और अधिकतम (P90) भावों का सटीक अनुमान लगाता है।
        - **PyTorch Bi-LSTM (अटेंशन लेयर सहित):** मौसमी चक्र, त्योहारी मांग और मौसम के झटकों के प्रभाव को समझता है।
        - **आवक संवेदनशीलता (Arrival Elasticity):** आवक के अचानक बढ़ने और घटने से कीमतों पर पड़ने वाले दबाव का आकलन करता है।
        - **मंडी कैलेंडर अनुकूलन:** रविवार, एपीएमसी अवकाश और हड़तालों के प्रभाव को स्वतः समायोजित करता है।
        """,
        "season_table_title": "🗓️ भारतीय प्रमुख फसल चक्र संदर्भ तालिका",
        "footer_text": "एग्रीप्राइस-एआई (AgriPrice-AI) प्लेटफॉर्म • भारतीय किसानों, व्यापारियों एवं एफपीओ (FPOs) के लिए समर्पित मुक्त बाजार गुप्तचर प्रणाली",
        "status_quo": "सामान्य स्थिति रुझान",
        "sim_shift": "सिम्युलेटेड बदलाव",
        "net_change": "कुल अंतर"
    }
}

# Crop Dictionary with English/Hindi Names, Categories, and Domain Context
CROP_METADATA = {
    "Onion": {
        "display_en": "🧅 Onion",
        "display_hi": "🧅 प्याज़ (Onion)",
        "category_en": "Perishable Crop",
        "category_hi": "जल्द खराब होने वाली फसल",
        "msp": None,
        "desc_en": "High price volatility driven by unseasonal rains and storage losses in Nashik/Alwar.",
        "desc_hi": "नाशिक व अलवर मंडियों में बेमौसम बारिश एवं भंडारण नुकसान से अत्यधिक मूल्य अस्थिरता।"
    },
    "Tomato": {
        "display_en": "🍅 Tomato",
        "display_hi": "🍅 टमाटर (Tomato)",
        "category_en": "Perishable Crop",
        "category_hi": "जल्द खराब होने वाली फसल",
        "msp": None,
        "desc_en": "Extreme short-term price elasticity; summer heatwaves severely reduce production.",
        "desc_hi": "अत्यधिक अल्पकालिक मूल्य संवेदनशीलता; गर्मी की लू से उत्पादन में भारी गिरावट।"
    },
    "Potato": {
        "display_en": "🥔 Potato",
        "display_hi": "🥔 आलू (Potato)",
        "category_en": "Perishable / Cold Storage",
        "category_hi": "शीतगृह फसल (Cold Storage)",
        "msp": None,
        "desc_en": "Rabi harvest peaks Feb-April. Heavy cold-storage releases dictate summer rates.",
        "desc_hi": "रबी फसल की आवक फरवरी-अप्रैल में चरम पर होती है। शीतगृहों से निकासी गर्मियों के भाव तय करती है।"
    },
    "Wheat": {
        "display_en": "🌾 Wheat",
        "display_hi": "🌾 गेहूं (Wheat)",
        "category_en": "Non-Perishable Food Grain",
        "category_hi": "मुख्य खाद्य अनाज (Non-Perishable)",
        "msp": 2275.0,
        "desc_en": "Primary Rabi food grain supported by Government procurement at MSP.",
        "desc_hi": "प्रमुख रबी खाद्य अनाज, सरकारी खरीद एवं न्यूनतम समर्थन मूल्य (MSP) द्वारा समर्थित।"
    },
    "Soybean": {
        "display_en": "🟡 Soybean",
        "display_hi": "🟡 सोयाबीन (Soybean)",
        "category_en": "Commercial Oilseed",
        "category_hi": "प्रमुख खरीफ तिलहन फसल",
        "msp": 4600.0,
        "desc_en": "Major Kharif oilseed in MP & Maharashtra; influenced by global edible oil demand.",
        "desc_hi": "मध्य प्रदेश एवं महाराष्ट्र की प्रमुख खरीफ तिलहन फसल; वैश्विक खाद्य तेल मांग से प्रभावित।"
    },
    "Mustard": {
        "display_en": "🌼 Mustard",
        "display_hi": "🌼 सरसों (Mustard)",
        "category_en": "Rabi Oilseed",
        "category_hi": "रबी तिलहन फसल",
        "msp": 5650.0,
        "desc_en": "Key winter oilseed in Rajasthan & Haryana; sensitive to post-harvest weather.",
        "desc_hi": "राजस्थान व हरियाणा की प्रमुख शीतकालीन तिलहन फसल; कटाई बाद मौसम के प्रति संवेदनशील।"
    },
    "Rice": {
        "display_en": "🌾 Rice / Paddy",
        "display_hi": "🌾 धान / चावल (Paddy)",
        "category_en": "Staple Food Grain",
        "category_hi": "मुख्य खाद्य अनाज",
        "msp": 2300.0,
        "desc_en": "India's staple crop. Heavy Kharif arrivals across Punjab, Haryana, and West Bengal.",
        "desc_hi": "भारत की मुख्य फसल। पंजाब, हरियाणा और पश्चिम बंगाल में भारी खरीफ आवक।"
    },
    "Cotton": {
        "display_en": "☁️ Cotton",
        "display_hi": "☁️ कपास / रुई (Cotton)",
        "category_en": "Commercial Cash Crop",
        "category_hi": "व्यापारिक नकदी फसल",
        "msp": 7120.0,
        "desc_en": "White gold of Gujarat & Maharashtra. Major driver of textile export earnings.",
        "desc_hi": "गुजरात एवं महाराष्ट्र का सफेद सोना। वस्त्र निर्यात का मुख्य आधार।"
    },
    "Maize": {
        "display_en": "🌽 Maize / Corn",
        "display_hi": "🌽 मक्का (Maize)",
        "category_en": "Coarse Grain / Feed",
        "category_hi": "मोटा अनाज / पोल्ट्री फीड",
        "msp": 2225.0,
        "desc_en": "Dual harvest crop with soaring industrial demand from poultry feed and starch units.",
        "desc_hi": "पोल्ट्री फीड और स्टार्च इकाइयों से बढ़ती औद्योगिक मांग वाली दोहरी फसल।"
    },
    "Chilli": {
        "display_en": "🌶️ Dry Red Chilli",
        "display_hi": "🌶️ सूखी लाल मिर्च (Red Chilli)",
        "category_en": "High-Value Spice",
        "category_hi": "उच्च मूल्य मसाला फसल",
        "msp": None,
        "desc_en": "Guntur (Andhra Pradesh) is Asia's largest trading hub. Driven by international export demand.",
        "desc_hi": "गुंटूर (आंध्र प्रदेश) एशिया का सबसे बड़ा व्यापार केंद्र। अंतर्राष्ट्रीय निर्यात मांग पर आधारित।"
    },
    "Turmeric": {
        "display_en": "🟡 Turmeric",
        "display_hi": "🟡 हल्दी (Turmeric)",
        "category_en": "Medicinal & Spice Crop",
        "category_hi": "मसाला व औषधीय फसल",
        "msp": None,
        "desc_en": "Nizamabad & Sangli market hubs. Curcumin quality and festival demand drive price spikes.",
        "desc_hi": "निजामाबाद व सांगली मुख्य मंडी केंद्र। करक्यूमिन गुणवत्ता एवं त्योहारी मांग से भाव उछलते हैं।"
    },
    "Gram": {
        "display_en": "🌰 Gram / Chana",
        "display_hi": "🌰 चना / दाल (Gram)",
        "category_en": "Major Pulse / Legume",
        "category_hi": "मुख्य रबी दलहन फसल",
        "msp": 5440.0,
        "desc_en": "Most consumed pulse in India. NAFED buffer procurement anchors market prices.",
        "desc_hi": "भारत की सर्वाधिक उपभोग्य दाल। नैफेड (NAFED) बफर खरीद से बाजार भाव स्थिर रहता है।"
    },
    "Groundnut": {
        "display_en": "🥜 Groundnut",
        "display_hi": "🥜 मूंगफली (Groundnut)",
        "category_en": "Commercial Oilseed",
        "category_hi": "व्यापारिक तिलहन फसल",
        "msp": 6783.0,
        "desc_en": "Key Kharif/Summer oilseed across Gujarat, AP, and Karnataka. Edible oil export driver.",
        "desc_hi": "गुजरात, आंध्र प्रदेश व कर्नाटक की प्रमुख तिलहन फसल। खाद्य तेल निर्यात पर आधारित।"
    },
    "Moong": {
        "display_en": "🟢 Moong Dal",
        "display_hi": "🟢 मूँग दाल (Moong)",
        "category_en": "High-Value Pulse",
        "category_hi": "उच्च मूल्य दलहन फसल",
        "msp": 8682.0,
        "desc_en": "Short duration high-value pulse. Early arrivals fetch premium prices in MP and Rajasthan.",
        "desc_hi": "कम अवधि वाली उच्च मूल्य दलहन। शुरुआती आवक पर मध्य प्रदेश और राजस्थान में अधिक भाव मिलता है।"
    },
    "Garlic": {
        "display_en": "🧄 Garlic",
        "display_hi": "🧄 लहसुन (Garlic)",
        "category_en": "High-Value Cash Crop",
        "category_hi": "उच्च मूल्य नकदी फसल",
        "msp": None,
        "desc_en": "High price volatility. MP (Mandsaur/Indore) and Rajasthan hubs dictate national rates.",
        "desc_hi": "अत्यधिक मूल्य उतार-चढ़ाव। मंदसौर, इंदौर और राजस्थान की मंडियां राष्ट्रीय भाव तय करती हैं।"
    },
    "Coriander": {
        "display_en": "🌿 Coriander Seeds",
        "display_hi": "🌿 धनिया (Coriander)",
        "category_en": "Rabi Spice Crop",
        "category_hi": "रबी मसाला फसल",
        "msp": None,
        "desc_en": "Major Rabi spice crop in Rajasthan and MP. Export demand and festival cycles guide prices.",
        "desc_hi": "राजस्थान व मध्य प्रदेश की प्रमुख मसाला फसल। निर्यात मांग एवं त्योहारी चक्र से भाव निर्धारित होते हैं।"
    },
    "Bajra": {
        "display_en": "🌾 Bajra / Pearl Millet",
        "display_hi": "🌾 बाजरा (Bajra)",
        "category_en": "Nutri-Cereal / Millet",
        "category_hi": "पोषक मोटा अनाज (Millets)",
        "msp": 2625.0,
        "desc_en": "Hardy drought-resistant millet. Winter consumption and feed industry support market bids.",
        "desc_hi": "सूखा सहनशील मोटा अनाज। शीतकालीन उपभोग एवं फीड उद्योग से मजबूत मांग समर्थन।"
    },
    "Jaggery": {
        "display_en": "🍯 Jaggery / Gur",
        "display_hi": "🍯 गुड़ (Jaggery)",
        "category_en": "Agro-Processed Commodity",
        "category_hi": "कृषि प्रसंस्कृत गन्ना उत्पाद",
        "msp": None,
        "desc_en": "Winter seasonal cane crushing product. High winter & festive demand across UP, Punjab, and MH.",
        "desc_hi": "गन्ना पेराई का मौसमी उत्पाद। उत्तर प्रदेश, पंजाब और महाराष्ट्र में सर्दियों में भारी मांग।"
    }
}

@st.cache_data(ttl=3600)
def load_data():
    prices_path = DATA_DIR / "sample_mandi_prices.csv"
    weather_path = DATA_DIR / "sample_weather.csv"
    
    if not prices_path.exists() or not weather_path.exists():
        from data.sample_generator import generate_agricultural_dataset
        generate_agricultural_dataset()
        
    df_p = pd.read_csv(prices_path)
    df_w = pd.read_csv(weather_path)
    df_p["record_date"] = pd.to_datetime(df_p["record_date"])
    df_w["record_date"] = pd.to_datetime(df_w["record_date"])
    return df_p, df_w

df_prices, df_weather = load_data()

# ==========================================
# SIDEBAR CONTROLS & SELECTION
# ==========================================
st.sidebar.title("🌾 AgriPrice-AI")

# 1. LANGUAGE SELECTOR (English / हिंदी)
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155; border-radius: 10px; padding: 10px 14px; margin-bottom: 12px;">
    <div style="color: #94a3b8; font-size: 0.76rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
        🌐 Language / भाषा चुनें
    </div>
</div>
""", unsafe_allow_html=True)

lang_choice = st.sidebar.radio(
    "Choose Language / भाषा",
    options=["English", "हिंदी (Hindi)"],
    index=0 if st.session_state.get("lang", "en") == "en" else 1,
    horizontal=True,
    label_visibility="collapsed"
)
is_hindi = "हिंदी" in lang_choice
lang = "hi" if is_hindi else "en"
st.session_state["lang"] = lang

def t(key: str) -> str:
    """Helper function to fetch localized strings."""
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

st.sidebar.caption(t("app_subtitle"))

# Extract unique states
all_states = sorted(list({meta["state"] for meta in settings.SUPPORTED_MANDIS.values()}))

st.sidebar.subheader(t("select_state_mandi"))
state_options = [t("all_states")] + all_states

# Persist state filter across language switch
current_state = st.session_state.get("selected_state", t("all_states"))
state_idx = 0
if current_state in state_options:
    state_idx = state_options.index(current_state)

state_choice = st.sidebar.selectbox(
    t("filter_state"),
    state_options,
    index=state_idx
)
st.session_state["selected_state"] = state_choice

# Filter mandis based on state
if state_choice != t("all_states") and state_choice in all_states:
    mandis_in_state = [
        m for m, meta in settings.SUPPORTED_MANDIS.items()
        if meta["state"] == state_choice
    ]
else:
    mandis_in_state = list(settings.SUPPORTED_MANDIS.keys())

# Persist Mandi across language switch
current_mandi = st.session_state.get("selected_mandi", mandis_in_state[0])
mandi_idx = 0
if current_mandi in mandis_in_state:
    mandi_idx = mandis_in_state.index(current_mandi)

mandi_name = st.sidebar.selectbox(t("select_mandi"), mandis_in_state, index=mandi_idx)
st.session_state["selected_mandi"] = mandi_name
mandi_meta = settings.SUPPORTED_MANDIS[mandi_name]

# Crop Selector based on Mandi's traded crops
st.sidebar.subheader(t("select_crop"))
primary_crops = mandi_meta.get("primary_crops", settings.SUPPORTED_COMMODITIES)
available_crops = [c for c in settings.SUPPORTED_COMMODITIES if c in primary_crops]
if not available_crops:
    available_crops = settings.SUPPORTED_COMMODITIES

def get_crop_label(crop_key):
    meta = CROP_METADATA.get(crop_key, {})
    return meta.get("display_hi" if is_hindi else "display_en", crop_key)

crop_labels = [get_crop_label(c) for c in available_crops]

# Persist crop across language switch
current_crop = st.session_state.get("selected_commodity", available_crops[0])
crop_idx = 0
if current_crop in available_crops:
    crop_idx = available_crops.index(current_crop)

selected_crop_label = st.sidebar.selectbox(
    t("commodity"),
    crop_labels,
    index=crop_idx
)

# Reverse lookup commodity key
commodity = available_crops[crop_labels.index(selected_crop_label)]
st.session_state["selected_commodity"] = commodity

crop_info = CROP_METADATA.get(commodity, {
    "display_en": commodity, "display_hi": commodity,
    "category_en": "General", "category_hi": "सामान्य",
    "msp": None, "desc_en": "", "desc_hi": ""
})

# Quick glossary expander in sidebar
with st.sidebar.expander(t("quick_glossary")):
    st.markdown(t("glossary_content"))

# Filter historical dataset for selected pair
crop_df = df_prices[
    (df_prices["commodity"] == commodity) & 
    (df_prices["mandi_name"] == mandi_name)
].sort_values("record_date").copy()

if len(crop_df) < 30:
    st.warning("⚠️ Trade records are limited for this commodity in the selected Mandi.")
    st.stop()

# Preprocess & Feature Extraction
cleaned_df, _ = detect_and_clean_outliers(crop_df)
daily_df = impute_missing_trading_days(cleaned_df)
feat_df = create_agricultural_features(daily_df, df_weather, horizons=[7, 15, 30])

# Model Inference
@st.cache_resource
def get_trained_model(crop, mandi):
    m = AgriculturalXGBoostForecaster(horizons=[7, 15, 30])
    m.train(feat_df, test_days=30)
    return m

forecaster = get_trained_model(commodity, mandi_name)
latest_feat = feat_df.iloc[[-1]].copy()
preds = forecaster.predict_latest(latest_feat)
importances = forecaster.get_feature_importances(horizon=7, top_n=5)

curr_price = float(latest_feat["modal_price"].iloc[0])
p50_7d = preds["7d"]["p50"]
p50_15d = preds["15d"]["p50"]
p50_30d = preds["30d"]["p50"]

pct_7d = ((p50_7d - curr_price) / curr_price) * 100
pct_15d = ((p50_15d - curr_price) / curr_price) * 100
pct_30d = ((p50_30d - curr_price) / curr_price) * 100

# ==========================================
# MAIN INTERFACE HEADER
# ==========================================
col_h1, col_h2 = st.columns([3, 1])
crop_display_current = crop_info["display_hi" if is_hindi else "display_en"]
crop_category_current = crop_info["category_hi" if is_hindi else "category_en"]

with col_h1:
    st.title(f"{crop_display_current} - {t('app_title')}")
    msp_badge = f"⚖️ {t('msp_label')}: ₹{int(crop_info['msp']):,}/Qtl" if crop_info['msp'] else f"⚖️ {t('msp_na')}"
    st.markdown(f"""
    <div style="margin-bottom: 12px;">
        <span class="lang-pill">🌐 {'हिन्दी संस्करण' if is_hindi else 'English Mode'}</span>
        <span class="tag-badge">📍 {mandi_name}</span>
        <span class="tag-badge">🏛️ {mandi_meta['state']} ({mandi_meta['district']})</span>
        <span class="tag-badge">🏷️ {crop_category_current}</span>
        <span class="tag-badge">{msp_badge}</span>
    </div>
    """, unsafe_allow_html=True)
with col_h2:
    last_trade_date = df_prices["record_date"].max().strftime("%d %b %Y")
    st.markdown(f"""
    <div style="text-align: right; padding-top: 10px;">
        <span style="color:#94a3b8; font-size:0.85rem;">{t('market_sync')}</span><br/>
        <strong style="color:#f8fafc; font-size:1.1rem;">📅 {last_trade_date}</strong>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# ACTIONABLE FARMER / TRADER RECOMMENDATION
# ==========================================
diff_15d_inr = p50_15d - curr_price

if pct_15d > 5.0:
    adv_class = "advisory-hold"
    adv_title = t("hold_title")
    adv_text = t("hold_text").format(
        commodity=crop_display_current,
        mandi_name=mandi_name,
        pct_15d=pct_15d,
        diff=diff_15d_inr
    )
elif pct_15d < -5.0:
    adv_class = "advisory-sell"
    adv_title = t("sell_title")
    adv_text = t("sell_text").format(
        commodity=crop_display_current,
        mandi_name=mandi_name,
        pct_15d=pct_15d,
        diff=abs(diff_15d_inr)
    )
else:
    adv_class = "advisory-stable"
    adv_title = t("stable_title")
    adv_text = t("stable_text").format(
        commodity=crop_display_current,
        mandi_name=mandi_name,
        pct_15d=abs(pct_15d)
    )

st.markdown(f"""
<div class="advisory-card {adv_class}">
    <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 6px;">{adv_title}</div>
    <div style="font-size: 0.95rem; line-height: 1.5; color: #f1f5f9;">{adv_text}</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# TOP 4 KPI CARDS
# ==========================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{t('kpi_spot_rate')}</div>
        <div class="metric-value">₹{curr_price:,.0f} <span style="font-size:0.95rem;color:#94a3b8">{t('kpi_per_qtl')}</span></div>
        <div class="metric-sub" style="color:#94a3b8">{t('kpi_rate_for_100kg').format(mandi=mandi_name.split('(')[0].strip())}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    color_class = "bullish" if pct_7d >= 0 else "bearish"
    symbol = "▲" if pct_7d >= 0 else "▼"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{t('kpi_7d_forecast')}</div>
        <div class="metric-value">₹{p50_7d:,.0f}</div>
        <div class="metric-sub {color_class}">{symbol} {abs(pct_7d):.1f}% ({'+₹' if pct_7d >= 0 else '-₹'}{abs(p50_7d - curr_price):,.0f})</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    color_class = "bullish" if pct_15d >= 0 else "bearish"
    symbol = "▲" if pct_15d >= 0 else "▼"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{t('kpi_15d_forecast')}</div>
        <div class="metric-value">₹{p50_15d:,.0f}</div>
        <div class="metric-sub {color_class}">{symbol} {abs(pct_15d):.1f}% ({'+₹' if pct_15d >= 0 else '-₹'}{abs(diff_15d_inr):,.0f})</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    color_class = "bullish" if pct_30d >= 0 else "bearish"
    symbol = "▲" if pct_30d >= 0 else "▼"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{t('kpi_30d_forecast')}</div>
        <div class="metric-value">₹{p50_30d:,.0f}</div>
        <div class="metric-sub {color_class}">{symbol} {abs(pct_30d):.1f}% ({t('range_label')}: ₹{preds['30d']['p10']:,.0f} - ₹{preds['30d']['p90']:,.0f})</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ==========================================
# 5 DEDICATED USER-FRIENDLY TABS
# ==========================================
tab_forecast, tab_arbitrage, tab_weather, tab_simulator, tab_guide = st.tabs([
    t("tab1_title"),
    t("tab2_title"),
    t("tab3_title"),
    t("tab4_title"),
    t("tab5_title")
])

# ------------------------------------------
# TAB 1: PRICE FORECAST & TRENDS
# ------------------------------------------
with tab_forecast:
    st.subheader(t("chart_title").format(commodity=crop_display_current))
    
    with st.expander(t("chart_help_title")):
        st.markdown(t("chart_help_body"))

    chart_history = daily_df.tail(120).copy()
    last_hist_date = chart_history["record_date"].max()

    future_dates = [last_hist_date + timedelta(days=i) for i in range(1, 31)]
    cone_dates = [last_hist_date] + future_dates
    cone_p50 = np.interp(range(len(cone_dates)), [0, 7, 15, 30], [curr_price, p50_7d, p50_15d, p50_30d])
    cone_p10 = np.interp(range(len(cone_dates)), [0, 7, 15, 30], [curr_price, preds['7d']['p10'], preds['15d']['p10'], preds['30d']['p10']])
    cone_p90 = np.interp(range(len(cone_dates)), [0, 7, 15, 30], [curr_price, preds['7d']['p90'], preds['15d']['p90'], preds['30d']['p90']])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=chart_history["record_date"],
            y=chart_history["arrivals_qtl"],
            name=t("trace_arrivals"),
            marker_color="rgba(148, 163, 184, 0.25)",
            opacity=0.6
        ),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=chart_history["record_date"],
            y=chart_history["modal_price"],
            name=t("trace_actual"),
            line=dict(color="#38bdf8", width=2.5)
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=cone_dates,
            y=cone_p90,
            name=t("trace_p90"),
            line=dict(color="rgba(16, 185, 129, 0)", width=0),
            showlegend=False
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=cone_dates,
            y=cone_p10,
            name=t("trace_band"),
            fill="tonexty",
            fillcolor="rgba(16, 185, 129, 0.16)",
            line=dict(color="rgba(16, 185, 129, 0)", width=0),
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=cone_dates,
            y=cone_p50,
            name=t("trace_p50"),
            line=dict(color="#10b981", width=3, dash="dash")
        ),
        secondary_y=False
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        height=480,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=True, gridcolor="#334155")
    fig.update_yaxes(title_text=t("yaxis_price"), secondary_y=False, showgrid=True, gridcolor="#334155")
    fig.update_yaxes(title_text=t("yaxis_arrivals"), secondary_y=True, showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    # Key Factor Explainability
    st.write("---")
    st.subheader(t("why_moving"))
    c_f1, c_f2 = st.columns([3, 2])
    with c_f1:
        if importances:
            imp_df = pd.DataFrame(importances, columns=["Feature", "Relative Importance (%)"])
            imp_df["Feature"] = imp_df["Feature"].str.replace("_", " ").str.title()
            bar_fig = go.Figure(go.Bar(
                x=imp_df["Relative Importance (%)"],
                y=imp_df["Feature"],
                orientation="h",
                marker_color="#10b981"
            ))
            bar_fig.update_layout(
                template="plotly_dark",
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title=t("feature_weight_title"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(bar_fig, use_container_width=True)
    with c_f2:
        st.markdown(f"""
        **{t('top_factors_title').format(commodity=crop_display_current)}**
        1. {t('top_factor_1')}
        2. {t('top_factor_2')}
        3. {t('top_factor_3')}
        """)

    # Tabular 30-Day Outlook
    with st.expander(t("table_forecast_expander")):
        daily_forecast_records = []
        for i, (f_date, p_low, p_exp, p_high) in enumerate(zip(future_dates, cone_p10[1:], cone_p50[1:], cone_p90[1:]), start=1):
            daily_forecast_records.append({
                t("col_day"): f"+{i}",
                t("col_date"): f_date.strftime("%d %b %Y"),
                t("col_p10"): round(p_low, 2),
                t("col_p50"): round(p_exp, 2),
                t("col_p90"): round(p_high, 2),
                t("col_change"): f"{((p_exp - curr_price)/curr_price)*100:+.1f}%"
            })
        st.dataframe(pd.DataFrame(daily_forecast_records), use_container_width=True)

# ------------------------------------------
# TAB 2: INTER-MANDI TRANSPORT PROFIT (ARBITRAGE)
# ------------------------------------------
with tab_arbitrage:
    st.subheader(t("arbitrage_title").format(commodity=crop_display_current))
    st.markdown(t("arbitrage_desc"))

    try:
        arbitrage_ops = find_spatial_arbitrage(commodity=commodity, max_radius_km=3000.0)
        if arbitrage_ops:
            arb_rows = []
            for op in arbitrage_ops:
                net_profit = op.net_profit_potential_inr_qtl
                truckload_profit = net_profit * 100 # Standard 10-Tonne Truckload = 100 Quintals
                arb_rows.append({
                    t("col_origin_mandi"): op.origin_mandi,
                    t("col_dest_mandi"): op.destination_mandi,
                    t("col_distance"): f"{op.distance_km:,.0f} km",
                    t("col_origin_price"): f"₹{op.origin_price:,.0f}/Qtl",
                    t("col_dest_price"): f"₹{op.destination_price:,.0f}/Qtl",
                    t("col_spread"): f"₹{op.gross_spread_inr:,.0f}/Qtl",
                    t("col_freight"): f"₹{op.estimated_freight_inr:,.0f}/Qtl",
                    t("col_net_profit"): f"₹{net_profit:,.0f}/Qtl",
                    t("col_truckload"): f"₹{truckload_profit:,.0f}"
                })
            
            st.dataframe(pd.DataFrame(arb_rows), use_container_width=True)
            
            # Highlight Best Route
            best_op = arbitrage_ops[0]
            st.success(f"""
            {t('top_route_title')} 
            {t('transport_from_to').format(commodity=crop_display_current, origin=best_op.origin_mandi, origin_price=f'₹{best_op.origin_price:,.0f}', dest=best_op.destination_mandi, dest_price=f'₹{best_op.destination_price:,.0f}')}
            - **{t('col_distance')}:** {best_op.distance_km:,.0f} km
            - **{t('freight_toll')}** ₹{best_op.estimated_freight_inr:,.0f} {t('per_quintal')}
            - **{t('net_profit_banner')}** **₹{best_op.net_profit_potential_inr_qtl:,.0f} {t('per_quintal')}** (approx. **₹{best_op.net_profit_potential_inr_qtl * 100:,.0f} {t('per_truck')}**)
            """)
        else:
            st.info(t("no_arbitrage").format(commodity=crop_display_current))
    except Exception as e:
        st.error(f"Error computing spatial arbitrage: {e}")

# ------------------------------------------
# TAB 3: WEATHER IMPACT
# ------------------------------------------
with tab_weather:
    st.subheader(t("weather_title").format(mandi_name=mandi_name))
    st.caption(t("weather_sub"))

    recent_weather = df_weather[df_weather["mandi_name"] == mandi_name].sort_values("record_date").tail(7)
    if len(recent_weather) > 0:
        latest_w = recent_weather.iloc[-1]
        
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{t('w_rainfall')}</div>
                <div class="metric-value">{latest_w['rainfall_mm']:.1f} <span style="font-size:1rem;color:#94a3b8">mm</span></div>
                <div class="metric-sub {'bearish' if latest_w['rainfall_mm'] > 30 else 'bullish'}">{t('w_rain_alert') if latest_w['rainfall_mm'] > 30 else t('w_rain_ok')}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with w2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{t('w_temp')}</div>
                <div class="metric-value">{latest_w['temp_max_c']:.1f} <span style="font-size:1rem;color:#94a3b8">°C</span></div>
                <div class="metric-sub {'bearish' if latest_w['temp_max_c'] > 38 else 'neutral'}">{t('w_heat_alert') if latest_w['temp_max_c'] > 38 else t('w_heat_ok')}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with w3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{t('w_moisture')}</div>
                <div class="metric-value">{latest_w['soil_moisture_pct']:.1f} <span style="font-size:1rem;color:#94a3b8">%</span></div>
                <div class="metric-sub neutral">{t('w_moisture_sub')}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with w4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{t('w_ndvi')}</div>
                <div class="metric-value">{latest_w['ndvi_index']:.2f}</div>
                <div class="metric-sub bullish">{t('w_ndvi_sub')}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.subheader(t("agronomic_advisory"))
        if latest_w["rainfall_mm"] > 25.0:
            st.warning(t("adv_rain").format(val=latest_w['rainfall_mm'], mandi=mandi_name))
        elif latest_w["temp_max_c"] > 38.0:
            st.warning(t("adv_heat").format(val=latest_w['temp_max_c']))
        else:
            st.info(t("adv_clear").format(mandi=mandi_name))

# ------------------------------------------
# TAB 4: WHAT-IF SCENARIO LAB
# ------------------------------------------
with tab_simulator:
    st.subheader(t("sim_title"))
    st.markdown(t("sim_desc"))

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_rain = st.slider(t("sim_rain_label"), 0.0, 100.0, 0.0, step=10.0, help=t("sim_rain_help"))
        sim_glut = st.slider(t("sim_glut_label"), -50, 100, 0, step=10, help=t("sim_glut_help"))
    with sim_col2:
        sim_export = st.toggle(t("sim_export_label"), value=False, help=t("sim_export_help"))
        sim_fuel = st.slider(t("sim_fuel_label"), -20, 40, 0, step=5, help=t("sim_fuel_help"))

    # Calculate simulated 15d price
    sim_multiplier = 1.0
    if sim_rain > 0:
        sim_multiplier += (sim_rain / 100.0) * 0.18 # Rain disrupts supply, raises price
    if sim_glut != 0:
        sim_multiplier -= (sim_glut / 100.0) * 0.22 # Glut depresses price, shortage raises it
    if sim_export:
        sim_multiplier *= 0.85 # Export ban locks supply inside, causing price drop

    sim_p15d = p50_15d * sim_multiplier
    sim_diff_pct = ((sim_p15d - curr_price) / curr_price) * 100

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{t('sim_baseline_title')}</div>
            <div class="metric-value">₹{p50_15d:,.0f} {t('kpi_per_qtl')}</div>
            <div class="metric-sub {'bullish' if pct_15d >= 0 else 'bearish'}">{t('status_quo')}: {pct_15d:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
        <div class="metric-card" style="border-color: #38bdf8;">
            <div class="metric-title">{t('sim_shock_title')}</div>
            <div class="metric-value" style="color: #38bdf8;">₹{sim_p15d:,.0f} {t('kpi_per_qtl')}</div>
            <div class="metric-sub {'bullish' if sim_diff_pct >= 0 else 'bearish'}">{t('sim_shift')}: {sim_diff_pct:+.1f}% ({t('net_change')}: {sim_p15d - p50_15d:+,.0f} ₹/Qtl)</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    if sim_export:
        st.error(t("sim_export_alert"))
    elif sim_rain > 40:
        st.success(t("sim_rain_alert"))

# ------------------------------------------
# TAB 5: BEGINNER GUIDE & GLOSSARY
# ------------------------------------------
with tab_guide:
    st.subheader(t("guide_title"))
    
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown(f"### {t('guide_terms_title')}")
        st.markdown(t("guide_terms_content"))
        
    with g_col2:
        st.markdown(f"### {t('guide_ai_title')}")
        st.markdown(t("guide_ai_content"))

    st.write("---")
    st.subheader(t("season_table_title"))
    
    if is_hindi:
        season_data = [
            {"फसल (Crop)": "गेहूं (Wheat)", "फसल मौसम": "रबी (Rabi)", "बुवाई का समय": "अक्टूबर - दिसंबर", "कटाई व मुख्य आवक": "मार्च - मई", "MSP सुरक्षा": "हाँ (₹2,275/कुंतल)"},
            {"फसल (Crop)": "धान / चावल (Rice)", "फसल मौसम": "खरीफ (Kharif)", "बुवाई का समय": "जून - जुलाई", "कटाई व मुख्य आवक": "अक्टूबर - दिसंबर", "MSP सुरक्षा": "हाँ (₹2,300/कुंतल)"},
            {"फसल (Crop)": "सरसों (Mustard)", "फसल मौसम": "रबी (Rabi)", "बुवाई का समय": "अक्टूबर - नवंबर", "कटाई व मुख्य आवक": "फरवरी - अप्रैल", "MSP सुरक्षा": "हाँ (₹5,650/कुंतल)"},
            {"फसल (Crop)": "सोयाबीन (Soybean)", "फसल मौसम": "खरीफ (Kharif)", "बुवाई का समय": "जून - जुलाई", "कटाई व मुख्य आवक": "सितंबर - नवंबर", "MSP सुरक्षा": "हाँ (₹4,600/कुंतल)"},
            {"फसल (Crop)": "कपास (Cotton)", "फसल मौसम": "खरीफ (Kharif)", "बुवाई का समय": "मई - जुलाई", "कटाई व मुख्य आवक": "अक्टूबर - जनवरी", "MSP सुरक्षा": "हाँ (₹7,120/कुंतल)"},
            {"फसल (Crop)": "चना (Gram)", "फसल मौसम": "रबी (Rabi)", "बुवाई का समय": "अक्टूबर - नवंबर", "कटाई व मुख्य आवक": "फरवरी - अप्रैल", "MSP सुरक्षा": "हाँ (₹5,440/कुंतल)"},
            {"फसल (Crop)": "मूंगफली (Groundnut)", "फसल मौसम": "खरीफ व जायद", "बुवाई का समय": "मई - जुलाई", "कटाई व मुख्य आवक": "अक्टूबर - दिसंबर", "MSP सुरक्षा": "हाँ (₹6,783/कुंतल)"},
            {"फसल (Crop)": "मूँग दाल (Moong)", "फसल मौसम": "खरीफ व जायद", "बुवाई का समय": "जून - जुलाई एवं मार्च", "कटाई व मुख्य आवक": "सितंबर - अक्टूबर एवं मई", "MSP सुरक्षा": "हाँ (₹8,682/कुंतल)"},
            {"फसल (Crop)": "लहसुन (Garlic)", "फसल मौसम": "रबी (Rabi)", "बुवाई का समय": "सितंबर - नवंबर", "कटाई व मुख्य आवक": "फरवरी - अप्रैल", "MSP सुरक्षा": "नहीं (बाजार आधारित)"},
            {"फसल (Crop)": "धनिया (Coriander)", "फसल मौसम": "रबी (Rabi)", "बुवाई का समय": "अक्टूबर - नवंबर", "कटाई व मुख्य आवक": "फरवरी - अप्रैल", "MSP सुरक्षा": "नहीं (बाजार आधारित)"},
            {"फसल (Crop)": "बाजरा (Bajra)", "फसल मौसम": "खरीफ (Kharif)", "बुवाई का समय": "जून - जुलाई", "कटाई व मुख्य आवक": "अक्टूबर - नवंबर", "MSP सुरक्षा": "हाँ (₹2,625/कुंतल)"},
            {"फसल (Crop)": "गुड़ (Jaggery)", "फसल मौसम": "गन्ना पेराई मौसम", "बुवाई का समय": "वर्ष भर", "कटाई व मुख्य आवक": "नवंबर - फरवरी", "MSP सुरक्षा": "नहीं (प्रसंस्कृत)"},
            {"फसल (Crop)": "प्याज़ (Onion)", "फसल मौसम": "खरीफ व रबी", "बुवाई का समय": "जुलाई एवं नवंबर", "कटाई व मुख्य आवक": "नवंबर-जनवरी एवं अप्रैल-जून", "MSP सुरक्षा": "नहीं (बाजार आधारित)"},
            {"फसल (Crop)": "टमाटर (Tomato)", "फसल मौसम": "बहु-मौसमी (Year-round)", "बुवाई का समय": "वर्ष भर", "कटाई व मुख्य आवक": "दिसंबर-मार्च एवं जुलाई-सितंबर", "MSP सुरक्षा": "नहीं (बाजार आधारित)"}
        ]
    else:
        season_data = [
            {"Crop": "Wheat", "Season": "Rabi", "Sowing Period": "Oct - Dec", "Harvest / Peak Arrivals": "Mar - May", "MSP Protected": "Yes (₹2,275/Qtl)"},
            {"Crop": "Rice / Paddy", "Season": "Kharif", "Sowing Period": "Jun - Jul", "Harvest / Peak Arrivals": "Oct - Dec", "MSP Protected": "Yes (₹2,300/Qtl)"},
            {"Crop": "Mustard", "Season": "Rabi", "Sowing Period": "Oct - Nov", "Harvest / Peak Arrivals": "Feb - Apr", "MSP Protected": "Yes (₹5,650/Qtl)"},
            {"Crop": "Soybean", "Season": "Kharif", "Sowing Period": "Jun - Jul", "Harvest / Peak Arrivals": "Sep - Nov", "MSP Protected": "Yes (₹4,600/Qtl)"},
            {"Crop": "Cotton", "Season": "Kharif", "Sowing Period": "May - Jul", "Harvest / Peak Arrivals": "Oct - Jan", "MSP Protected": "Yes (₹7,120/Qtl)"},
            {"Crop": "Gram / Chana", "Season": "Rabi", "Sowing Period": "Oct - Nov", "Harvest / Peak Arrivals": "Feb - Apr", "MSP Protected": "Yes (₹5,440/Qtl)"},
            {"Crop": "Groundnut", "Season": "Kharif & Summer", "Sowing Period": "May - Jul", "Harvest / Peak Arrivals": "Oct - Dec", "MSP Protected": "Yes (₹6,783/Qtl)"},
            {"Crop": "Moong Dal", "Season": "Kharif & Zaid", "Sowing Period": "Jun - Jul & Mar", "Harvest / Peak Arrivals": "Sep - Oct & May", "MSP Protected": "Yes (₹8,682/Qtl)"},
            {"Crop": "Garlic", "Season": "Rabi", "Sowing Period": "Sep - Nov", "Harvest / Peak Arrivals": "Feb - Apr", "MSP Protected": "No (Market Driven)"},
            {"Crop": "Coriander", "Season": "Rabi", "Sowing Period": "Oct - Nov", "Harvest / Peak Arrivals": "Feb - Apr", "MSP Protected": "No (Market Driven)"},
            {"Crop": "Bajra", "Season": "Kharif", "Sowing Period": "Jun - Jul", "Harvest / Peak Arrivals": "Oct - Nov", "MSP Protected": "Yes (₹2,625/Qtl)"},
            {"Crop": "Jaggery / Gur", "Season": "Post-Harvest Cane", "Sowing Period": "Year-round", "Harvest / Peak Arrivals": "Nov - Feb", "MSP Protected": "No (Agro-processed)"},
            {"Crop": "Onion", "Season": "Kharif & Rabi", "Sowing Period": "Jul & Nov", "Harvest / Peak Arrivals": "Nov-Jan & Apr-Jun", "MSP Protected": "No (Market Driven)"},
            {"Crop": "Tomato", "Season": "Multi-season", "Sowing Period": "Year-round", "Harvest / Peak Arrivals": "Dec-Mar & Jul-Sep", "MSP Protected": "No (Market Driven)"}
        ]
    st.dataframe(pd.DataFrame(season_data), use_container_width=True)

# Footer
st.markdown(f"""
<div style="text-align: center; margin-top: 40px; padding: 20px; color: #64748b; font-size: 0.85rem; border-top: 1px solid #1e293b;">
    {t('footer_text')}
</div>
""", unsafe_allow_html=True)
