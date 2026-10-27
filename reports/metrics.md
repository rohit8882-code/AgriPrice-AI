# AgriPrice-AI Model Performance Benchmarks

**Target Commodity:** Onion | **Benchmark Mandi:** Lasalgaon (Nashik, MH)
**Test Window:** 30 days out-of-time split

| Model Architecture | WMAPE (%) | MAE (₹/Qtl) | RMSE (₹/Qtl) | MDA (Directional Acc. %) |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Persistence** | 20.38% | ₹579.77 | ₹694.42 | 36.7% |
| **SARIMAX (Baseline)** | 29.07% | ₹827.14 | ₹867.28 | 43.3% |
| **XGBoost (Multi-Quantile)** | 6.32% | ₹189.24 | ₹275.62 | 66.7% |
| **Bi-LSTM + Temporal Attention** | 7.33% | ₹208.59 | ₹267.02 | 86.7% |
