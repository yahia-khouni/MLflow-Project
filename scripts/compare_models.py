"""
compare_models.py — Compare two model versions from the registry
================================================================
Given two model versions, loads their training metrics from MLflow
and prints a side-by-side comparison. Used for deciding whether to
promote a new model or keep the existing one.

Think of it as a "diff" between two deployments:
  - Like comparing lighthouse scores between two builds
  - Or A/B test results between two feature flags

Usage:
    python scripts/compare_models.py --v1 1 --v2 2
    python scripts/compare_models.py --v1 1 --v2 2 --model-name churn-predictor

Requires:
    pip install mlflow boto3
"""

import os
import sys
import argparse
import mlflow
from mlflow.tracking import MlflowClient

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin_secret_2024")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

# Metrics to compare (in display order)
METRICS = [
    ("auc_roc", "AUC-ROC", True),        # (key, display_name, higher_is_better)
    ("accuracy", "Accuracy", True),
    ("precision", "Precision", True),
    ("recall", "Recall", True),
    ("f1_score", "F1-Score", True),
    ("training_duration_seconds", "Training Time (s)", False),  # lower is better
]


def get_version_info(client: MlflowClient, model_name: str, version: str) -> dict:
    """Fetch model version metadata and training metrics."""
    mv = client.get_model_version(model_name, version)
    run = client.get_run(mv.run_id)

    return {
        "version": version,
        "stage": mv.current_stage,
        "run_id": mv.run_id,
        "description": mv.description or "(no description)",
        "metrics": run.data.metrics,
        "params": run.data.params,
        "tags": {t.key: t.value for t in mv.tags} if hasattr(mv, 'tags') else {},
    }


def compare(v1_info: dict, v2_info: dict):
    """Print a side-by-side comparison table."""
    print(f"\n  {'Metric':<25} {'v' + v1_info['version']:>12} {'v' + v2_info['version']:>12} {'Winner':>10}")
    print("  " + "-" * 62)

    v1_wins = 0
    v2_wins = 0

    for metric_key, display_name, higher_is_better in METRICS:
        val1 = v1_info["metrics"].get(metric_key, 0.0)
        val2 = v2_info["metrics"].get(metric_key, 0.0)

        if metric_key == "training_duration_seconds":
            fmt1 = f"{val1:>10.1f}s"
            fmt2 = f"{val2:>10.1f}s"
        else:
            fmt1 = f"{val1:>12.4f}"
            fmt2 = f"{val2:>12.4f}"

        # Determine winner
        if higher_is_better:
            if val1 > val2:
                winner = f"v{v1_info['version']} ✅"
                v1_wins += 1
            elif val2 > val1:
                winner = f"v{v2_info['version']} ✅"
                v2_wins += 1
            else:
                winner = "tie"
        else:
            if val1 < val2:
                winner = f"v{v1_info['version']} ✅"
                v1_wins += 1
            elif val2 < val1:
                winner = f"v{v2_info['version']} ✅"
                v2_wins += 1
            else:
                winner = "tie"

        print(f"  {display_name:<25} {fmt1} {fmt2} {winner:>10}")

    # Overall verdict
    print("  " + "-" * 62)
    print(f"  {'Score':<25} {'':>12} {'':>12} {v1_wins}-{v2_wins}")

    if v1_wins > v2_wins:
        overall_winner = v1_info["version"]
    elif v2_wins > v1_wins:
        overall_winner = v2_info["version"]
    else:
        overall_winner = None

    return overall_winner


def main():
    parser = argparse.ArgumentParser(description="Compare two model versions")
    parser.add_argument("--model-name", default="churn-predictor", help="Registered model name")
    parser.add_argument("--v1", required=True, help="First model version number")
    parser.add_argument("--v2", required=True, help="Second model version number")
    args = parser.parse_args()

    print("=" * 60)
    print("Model Version Comparison")
    print("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print(f"\n  Model: {args.model_name}")
    print(f"  Comparing: v{args.v1} vs v{args.v2}")

    # Fetch info for both versions
    try:
        v1_info = get_version_info(client, args.model_name, args.v1)
        v2_info = get_version_info(client, args.model_name, args.v2)
    except Exception as e:
        print(f"\n  ❌ Error fetching model versions: {e}")
        print("  Make sure both versions exist in the registry.")
        sys.exit(1)

    # Print version details
    for info in [v1_info, v2_info]:
        print(f"\n  Version {info['version']}:")
        print(f"    Stage:     {info['stage']}")
        print(f"    Algorithm: {info['params'].get('algorithm', 'unknown')}")
        print(f"    Run ID:    {info['run_id'][:12]}...")

    # Compare metrics
    winner = compare(v1_info, v2_info)

    if winner:
        print(f"\n  🏆 RECOMMENDATION: Version {winner} has better overall metrics.")
        auc_winner = v1_info if winner == v1_info["version"] else v2_info
        print(f"     AUC-ROC: {auc_winner['metrics'].get('auc_roc', 0):.4f}")

        if auc_winner["stage"] != "Production":
            print(f"\n  To promote v{winner} to Production, run:")
            print(f"    python scripts/promote_model.py --version {winner}")
    else:
        print(f"\n  🤝 TIE — Both versions have equal metrics.")
        print("  Keep the current Production model to avoid unnecessary changes.")


if __name__ == "__main__":
    main()
