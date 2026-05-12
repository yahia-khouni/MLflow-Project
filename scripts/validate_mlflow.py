"""
MLflow Validation Script — Phase 1 Smoke Test
==============================================
This script verifies that the MLflow Tracking Server is properly
connected to PostgreSQL (metadata) and MinIO (artifacts).

It creates a test experiment, logs some fake parameters/metrics,
saves a test artifact, and then reads everything back.

Run after docker-compose up:
    pip install mlflow boto3
    python scripts/validate_mlflow.py

If you see "ALL CHECKS PASSED" at the end, Phase 1 is complete!
"""

import os
import sys
import time
import json
import tempfile

# --- Configuration ---
# When running from your host machine (not inside Docker),
# we connect to MLflow via localhost. Inside Docker, services
# use container names (e.g., http://mlflow:5000).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# MinIO credentials — MLflow needs these to store artifacts.
# These match what's in .env / .env.example.
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin_secret_2024")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

import mlflow

def main():
    print("=" * 60)
    print("MLflow Infrastructure Validation Script")
    print("=" * 60)
    print()

    # --- Step 1: Connect to MLflow ---
    print(f"[1/5] Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        # This call hits the MLflow API — if it fails, the server is down
        experiments = mlflow.search_experiments()
        print(f"      ✅ Connected! Found {len(experiments)} existing experiment(s).")
    except Exception as e:
        print(f"      ❌ Failed to connect to MLflow: {e}")
        print("      Make sure docker-compose is running: docker-compose up -d")
        sys.exit(1)

    # --- Step 2: Create a test experiment ---
    print()
    print("[2/5] Creating test experiment 'infrastructure-validation'...")
    experiment_name = "infrastructure-validation"
    mlflow.set_experiment(experiment_name)
    print(f"      ✅ Experiment '{experiment_name}' created/found.")

    # --- Step 3: Log parameters, metrics, and an artifact ---
    print()
    print("[3/5] Starting a test run with parameters, metrics, and an artifact...")

    with mlflow.start_run(run_name="validation-run") as run:
        # Log parameters (like hyperparameters in ML, or config in web dev)
        mlflow.log_param("test_param", "hello_mlflow")
        mlflow.log_param("algorithm", "logistic_regression")
        mlflow.log_param("learning_rate", 0.01)
        print("      ✅ Parameters logged.")

        # Log metrics (like performance scores)
        mlflow.log_metric("accuracy", 0.95)
        mlflow.log_metric("auc_roc", 0.89)
        mlflow.log_metric("f1_score", 0.91)
        print("      ✅ Metrics logged.")

        # Log an artifact (a file stored in MinIO)
        # This tests the MinIO connection!
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                {
                    "validation": True,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "message": "If you can read this, MinIO artifact storage is working!",
                },
                f,
                indent=2,
            )
            temp_path = f.name

        mlflow.log_artifact(temp_path, artifact_path="validation")
        os.unlink(temp_path)
        print("      ✅ Artifact logged (stored in MinIO).")

        run_id = run.info.run_id
        print(f"      Run ID: {run_id}")

    # --- Step 4: Read back the logged data ---
    print()
    print("[4/5] Reading back logged data to verify persistence...")

    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id)

    params = run_data.data.params
    metrics = run_data.data.metrics

    assert params["test_param"] == "hello_mlflow", "Parameter mismatch!"
    assert float(metrics["accuracy"]) == 0.95, "Metric mismatch!"
    print("      ✅ Parameters and metrics verified in PostgreSQL.")

    artifacts = client.list_artifacts(run_id, path="validation")
    assert len(artifacts) > 0, "No artifacts found in MinIO!"
    print("      ✅ Artifact verified in MinIO.")

    # --- Step 5: Summary ---
    print()
    print("[5/5] Cleaning up...")
    # We keep the test experiment for reference — you can delete it
    # from the MLflow UI if you want.
    print("      ℹ️  Test experiment kept for reference (visible in MLflow UI).")

    print()
    print("=" * 60)
    print("🎉 ALL CHECKS PASSED — Phase 1 Infrastructure is working!")
    print("=" * 60)
    print()
    print("What was verified:")
    print("  ✅ MLflow Tracking Server is running and accessible")
    print("  ✅ PostgreSQL backend store is working (params/metrics persisted)")
    print("  ✅ MinIO artifact store is working (file uploaded and retrieved)")
    print("  ✅ Experiment creation and run logging work end-to-end")
    print()
    print("Next steps:")
    print("  1. Open MLflow UI: http://localhost:5000")
    print("  2. Click on 'infrastructure-validation' experiment")
    print("  3. You should see the 'validation-run' with metrics and artifacts")
    print("  4. If everything looks good, you're ready for Phase 2! 🚀")
    print()


if __name__ == "__main__":
    main()
