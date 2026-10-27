"""
Evaluation script: benchmarks model performance across architectures and writes reports/metrics.csv & reports/metrics.md.
"""

import sys
from pathlib import Path
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agriprice.config.settings import settings, REPORTS_DIR
from agriprice.models.benchmark_runner import run_benchmark
from agriprice.utils.logger import get_logger

logger = get_logger("evaluate_script")


def run_full_evaluation():
    logger.info("Executing comprehensive model evaluation...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_data = run_benchmark(
        commodity="Onion",
        mandi_name="Lasalgaon (Nashik, MH)",
        test_days=30
    )

    rows = []
    for model_name, metrics in benchmark_data.get("benchmarks", {}).items():
        rows.append({
            "model_name": model_name,
            "wmape_pct": metrics.get("wmape_pct", 0.0),
            "mae_inr_qtl": metrics.get("mae_inr_qtl", 0.0),
            "rmse_inr_qtl": metrics.get("rmse_inr_qtl", 0.0),
            "mda_directional_acc_pct": metrics.get("mda_directional_acc_pct", 0.0),
        })

    metrics_df = pd.DataFrame(rows)

    # 1. Write reports/metrics.csv
    csv_path = REPORTS_DIR / "metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    logger.info("Saved metrics to %s", csv_path)

    # 2. Write reports/metrics.md
    md_path = REPORTS_DIR / "metrics.md"
    md_content = [
        "# AgriPrice-AI Model Performance Benchmarks",
        "",
        f"**Target Commodity:** {benchmark_data.get('commodity')} | **Benchmark Mandi:** {benchmark_data.get('mandi_name')}",
        f"**Test Window:** {benchmark_data.get('test_days_window', 30)} days out-of-time split",
        "",
        "| Model Architecture | WMAPE (%) | MAE (₹/Qtl) | RMSE (₹/Qtl) | MDA (Directional Acc. %) |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]
    for _, r in metrics_df.iterrows():
        mda_val = f"{r['mda_directional_acc_pct']:.1f}%" if pd.notnull(r['mda_directional_acc_pct']) else "N/A"
        md_content.append(
            f"| **{r['model_name']}** | {r['wmape_pct']:.2f}% | ₹{r['mae_inr_qtl']:.2f} | ₹{r['rmse_inr_qtl']:.2f} | {mda_val} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content) + "\n")

    logger.info("Saved markdown summary to %s", md_path)


if __name__ == "__main__":
    run_full_evaluation()
