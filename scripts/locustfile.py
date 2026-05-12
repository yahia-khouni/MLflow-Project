"""
locustfile.py — Load Testing with Locust
==========================================
Locust simulates many users hitting your API at once.
Think of it like Apache JMeter or k6 but in Python.

It answers: "Can our API handle 200 concurrent users?"

Usage:
    pip install locust
    locust -f scripts/locustfile.py --host http://localhost:8000

Then open http://localhost:8089 to configure and start the test.
Or run headless:
    locust -f scripts/locustfile.py --host http://localhost:8000 \
           --users 200 --spawn-rate 10 --run-time 5m --headless
"""

import random
from locust import HttpUser, task, between


# Valid customer data for generating random requests
GENDERS = ["Male", "Female"]
YES_NO = ["Yes", "No"]
INTERNET = ["DSL", "Fiber optic", "No"]
INTERNET_DEPS = ["Yes", "No", "No internet service"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENTS = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)",
]


def random_customer():
    """Generate a random but valid customer feature set."""
    internet = random.choice(INTERNET)
    inet_dep = "No internet service" if internet == "No" else random.choice(["Yes", "No"])
    phone = random.choice(YES_NO)
    multi = "No phone service" if phone == "No" else random.choice(["Yes", "No"])
    tenure = random.randint(0, 72)
    monthly = round(random.uniform(18.0, 120.0), 2)

    return {
        "gender": random.choice(GENDERS),
        "SeniorCitizen": random.choice([0, 1]),
        "Partner": random.choice(YES_NO),
        "Dependents": random.choice(YES_NO),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": inet_dep,
        "OnlineBackup": inet_dep,
        "DeviceProtection": inet_dep,
        "TechSupport": inet_dep,
        "StreamingTV": inet_dep,
        "StreamingMovies": inet_dep,
        "Contract": random.choice(CONTRACTS),
        "PaperlessBilling": random.choice(YES_NO),
        "PaymentMethod": random.choice(PAYMENTS),
        "MonthlyCharges": monthly,
        "TotalCharges": round(monthly * tenure, 2),
    }


class ChurnPredictionUser(HttpUser):
    """
    Simulates a user of the Churn Prediction API.

    wait_time = between(0.5, 2) means each user waits
    0.5–2 seconds between requests (realistic pacing).
    """
    wait_time = between(0.5, 2)

    @task(7)
    def predict_single(self):
        """Single prediction — the most common request type."""
        self.client.post(
            "/predict",
            json=random_customer(),
            name="/predict [single]",
        )

    @task(2)
    def predict_batch(self):
        """Batch prediction with 50 records."""
        batch = {"customers": [random_customer() for _ in range(50)]}
        self.client.post(
            "/predict/batch",
            json=batch,
            name="/predict/batch [50 records]",
        )

    @task(1)
    def check_health(self):
        """Health check — occasional monitoring request."""
        self.client.get("/health", name="/health")
