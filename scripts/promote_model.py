"""
promote_model.py — Automate Staging → Production transition
============================================================
This script validates a model version in Staging and promotes
it to Production if it meets the quality threshold (AUC > 0.80).

Think of this as a CI/CD deployment script:
  - Like `vercel promote staging` or `kubectl set image deployment/...`
  - But for ML models instead of web apps

Usage:
    python scripts/promote_model.py
    python scripts/promote_model.py --model-name churn-predictor --version 1
    python scripts/promote_model.py --threshold 0.85

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


def get_staging_version(client: MlflowClient, model_name: str, version: str = None):
    """Get the model version in Staging, or a specific version."""
    if version:
        mv = client.get_model_version(model_name, version)
        return mv

    # Find the latest version in Staging
    versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not versions:
        print(f"  ❌ No model version found in 'Staging' for '{model_name}'")
        print("  Run the training pipeline first, then check MLflow UI.")
        sys.exit(1)
    return versions[0]


def validate_model(client: MlflowClient, model_version, threshold: float) -> bool:
    """
    Validate that the model meets quality thresholds.
    Checks the AUC-ROC metric from the training run.
    """
    run_id = model_version.run_id
    run = client.get_run(run_id)
    metrics = run.data.metrics

    auc_roc = metrics.get("auc_roc", 0.0)
    accuracy = metrics.get("accuracy", 0.0)
    f1_score = metrics.get("f1_score", 0.0)

    print(f"\n  📊 Validation Results (from training run {run_id[:8]}...):")
    print(f"     AUC-ROC:   {auc_roc:.4f}  {'✅' if auc_roc >= threshold else '❌'} (threshold: {threshold})")
    print(f"     Accuracy:  {accuracy:.4f}")
    print(f"     F1-Score:  {f1_score:.4f}")

    return auc_roc >= threshold


def promote(client: MlflowClient, model_name: str, version: str):
    """Transition model from Staging → Production."""
    # Archive any existing Production models first
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for pv in prod_versions:
        print(f"  📦 Archiving current Production model (v{pv.version})...")
        client.transition_model_version_stage(
            model_name, pv.version, "Archived"
        )

    # Promote new version
    client.transition_model_version_stage(
        model_name, version, "Production"
    )
    print(f"\n  🎉 Model '{model_name}' v{version} promoted to Production!")


def main():
    parser = argparse.ArgumentParser(description="Promote a model from Staging to Production")
    parser.add_argument("--model-name", default="churn-predictor", help="Registered model name")
    parser.add_argument("--version", default=None, help="Specific version to promote (default: latest Staging)")
    parser.add_argument("--threshold", type=float, default=0.80, help="Minimum AUC-ROC threshold")
    parser.add_argument("--force", action="store_true", help="Skip validation and force promotion")
    args = parser.parse_args()

    print("=" * 60)
    print("Model Promotion Script")
    print("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print(f"\n  Model:     {args.model_name}")
    print(f"  MLflow:    {MLFLOW_TRACKING_URI}")
    print(f"  Threshold: {args.threshold}")

    # Get the staging version
    model_version = get_staging_version(client, args.model_name, args.version)
    version_num = model_version.version
    print(f"  Version:   {version_num} (stage: {model_version.current_stage})")

    if args.force:
        print("\n  ⚠️  Force mode — skipping validation")
        promote(client, args.model_name, version_num)
        return

    # Validate
    if validate_model(client, model_version, args.threshold):
        print(f"\n  ✅ Model passes validation!")
        promote(client, args.model_name, version_num)
    else:
        print(f"\n  ❌ Model FAILED validation (AUC below {args.threshold})")
        print("  Model NOT promoted. Retrain with better hyperparameters.")
        sys.exit(1)


if __name__ == "__main__":
    main()
