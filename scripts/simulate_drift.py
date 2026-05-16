"""
simulate_drift.py — Data Drift Simulation Script
==================================================
This script demonstrates what happens when the input data
distribution shifts after the model was trained.

Think of it like this:
  - You trained a spam filter on English emails
  - Suddenly your users start sending Spanish emails
  - The model hasn't seen this pattern → its confidence drops

We simulate this by:
  1. Sending 500 "normal" requests (realistic customer data)
  2. Sending 500 "drifted" requests (extreme values)
  3. Watching the probability distribution shift in Grafana

Usage:
    python scripts/simulate_drift.py
    python scripts/simulate_drift.py --api http://localhost:8000

Watch the Grafana dashboard at http://localhost:3000 while running!
"""

import os
import sys
import time
import random
import argparse
import json

try:
    import httpx
except ImportError:
    print("Please install httpx: pip install httpx")
    sys.exit(1)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ============================================================
# Normal Customer Templates (realistic data)
# ============================================================
# These represent typical customers from the training dataset.
# The model should make confident predictions on these.
# ============================================================
NORMAL_TEMPLATES = [
    {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 24, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "Yes",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "One year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 56.95, "TotalCharges": 1367.8,
    },
    {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "Yes", "tenure": 48, "PhoneService": "Yes",
        "MultipleLines": "Yes", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "Yes", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Two year", "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 99.65, "TotalCharges": 4783.2,
    },
    {
        "gender": "Male", "SeniorCitizen": 1, "Partner": "No",
        "Dependents": "No", "tenure": 6, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.1, "TotalCharges": 420.6,
    },
]


def create_normal_request():
    """Create a realistic customer request with slight randomization."""
    template = random.choice(NORMAL_TEMPLATES).copy()
    # Add small random variation to numeric fields
    template["tenure"] = max(0, min(72, template["tenure"] + random.randint(-3, 3)))
    template["MonthlyCharges"] = round(template["MonthlyCharges"] * random.uniform(0.9, 1.1), 2)
    template["TotalCharges"] = round(template["MonthlyCharges"] * template["tenure"], 2)
    return template


def create_drifted_request():
    """
    Create an extreme/unusual customer request.

    Drift scenario: new high-risk customer cohort arrives.
    - MonthlyCharges multiplied by 3 (way above training range)
    - tenure set to 1 (brand new customers)
    - All month-to-month contracts with electronic check

    The model hasn't seen this pattern during training, so its
    predictions should be less confident / more extreme.
    """
    template = random.choice(NORMAL_TEMPLATES).copy()
    # Apply drift
    template["tenure"] = 1
    template["MonthlyCharges"] = round(template["MonthlyCharges"] * 3, 2)
    template["TotalCharges"] = round(template["MonthlyCharges"] * 1, 2)
    template["Contract"] = "Month-to-month"
    template["PaymentMethod"] = "Electronic check"
    template["PaperlessBilling"] = "Yes"
    template["SeniorCitizen"] = random.choice([0, 1])
    return template


def send_requests(client, api_url, requests_data, label, delay=0.05):
    """Send requests to the API and track results."""
    success = 0
    errors = 0
    churn_count = 0
    total_prob = 0.0

    for i, data in enumerate(requests_data):
        try:
            resp = client.post(f"{api_url}/predict", json=data, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                success += 1
                if result["churn_prediction"]:
                    churn_count += 1
                total_prob += result["churn_probability"]
            else:
                errors += 1
        except Exception:
            errors += 1

        # Progress indicator
        if (i + 1) % 50 == 0:
            avg_prob = total_prob / max(success, 1)
            churn_pct = churn_count / max(success, 1) * 100
            print(
                f"    [{label}] {i+1}/{len(requests_data)} sent | "
                f"Churn rate: {churn_pct:.1f}% | "
                f"Avg prob: {avg_prob:.3f} | "
                f"Errors: {errors}"
            )

        time.sleep(delay)

    return success, errors, churn_count, total_prob


def main():
    parser = argparse.ArgumentParser(description="Simulate data drift for monitoring")
    parser.add_argument("--api", default=API_URL, help="API base URL")
    parser.add_argument("--normal", type=int, default=500, help="Number of normal requests")
    parser.add_argument("--drifted", type=int, default=500, help="Number of drifted requests")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between requests (seconds)")
    args = parser.parse_args()

    print("=" * 60)
    print("Data Drift Simulation")
    print("=" * 60)
    print(f"\n  API: {args.api}")
    print(f"  Normal requests: {args.normal}")
    print(f"  Drifted requests: {args.drifted}")
    print(f"  Delay: {args.delay}s between requests")
    print(f"\n  Open Grafana to see the drift: http://localhost:3000")

    client = httpx.Client()

    # --- Phase 1: Normal requests (baseline) ---
    print(f"\n{'=' * 60}")
    print("PHASE 1: Sending NORMAL requests (baseline distribution)")
    print(f"{'=' * 60}")

    normal_data = [create_normal_request() for _ in range(args.normal)]
    n_success, n_errors, n_churn, n_prob = send_requests(
        client, args.api, normal_data, "NORMAL", args.delay
    )

    avg_normal_prob = n_prob / max(n_success, 1)
    print(f"\n  Normal results:")
    print(f"    Success: {n_success} | Errors: {n_errors}")
    print(f"    Churn rate: {n_churn/max(n_success,1)*100:.1f}%")
    print(f"    Avg probability: {avg_normal_prob:.3f}")

    # Pause to let the baseline settle in Grafana
    print(f"\n  [PAUSE] Pausing 5 seconds to establish baseline in Grafana...")
    time.sleep(5)

    # --- Phase 2: Drifted requests ---
    print(f"\n{'=' * 60}")
    print("PHASE 2: Sending DRIFTED requests (shifted distribution)")
    print(f"{'=' * 60}")
    print("  MonthlyCharges × 3, tenure = 1, month-to-month contracts")

    drifted_data = [create_drifted_request() for _ in range(args.drifted)]
    d_success, d_errors, d_churn, d_prob = send_requests(
        client, args.api, drifted_data, "DRIFT", args.delay
    )

    avg_drift_prob = d_prob / max(d_success, 1)
    print(f"\n  Drifted results:")
    print(f"    Success: {d_success} | Errors: {d_errors}")
    print(f"    Churn rate: {d_churn/max(d_success,1)*100:.1f}%")
    print(f"    Avg probability: {avg_drift_prob:.3f}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("DRIFT COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25} {'Normal':>10} {'Drifted':>10} {'Change':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Churn rate':<25} {n_churn/max(n_success,1)*100:>9.1f}% {d_churn/max(d_success,1)*100:>9.1f}%")
    print(f"  {'Avg probability':<25} {avg_normal_prob:>10.3f} {avg_drift_prob:>10.3f} {avg_drift_prob-avg_normal_prob:>+10.3f}")
    print()
    print("  >> Check Grafana now! You should see:")
    print("     - Churn probability distribution shifted right")
    print("     - Higher churn prediction rate in the second half")
    print("     - Possible change in confidence level distribution")

    client.close()


if __name__ == "__main__":
    main()
