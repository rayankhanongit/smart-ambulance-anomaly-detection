import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score

# Load predictions
df = pd.read_csv("data/processed/final_predictions.csv")

# -----------------------------
# Create ground truth labels
# -----------------------------
# Based on how we generated the data,
# deterioration happened between 900–1200

df["ground_truth"] = 0
df.loc[900:1200, "ground_truth"] = 1

# -----------------------------
# Prediction column
# -----------------------------

df["predicted"] = df["alert"].astype(int)

# -----------------------------
# Calculate metrics
# -----------------------------

precision = precision_score(df["ground_truth"], df["predicted"])
recall = recall_score(df["ground_truth"], df["predicted"])

false_alerts = ((df["predicted"] == 1) & (df["ground_truth"] == 0)).sum()
total_alerts = (df["predicted"] == 1).sum()

false_alert_rate = false_alerts / total_alerts if total_alerts > 0 else 0

print("Evaluation Results")
print("-------------------")
print("Precision:", precision)
print("Recall:", recall)
print("False Alert Rate:", false_alert_rate)

# -----------------------------
# Alert latency calculation
# -----------------------------

true_event_start = 900

alerts = df[df["predicted"] == 1].index

latency = None

for a in alerts:
    if a >= true_event_start:
        latency = a - true_event_start
        break

print("Alert Latency:", latency, "seconds")

# Save evaluation results

results = pd.DataFrame({
    "precision":[precision],
    "recall":[recall],
    "false_alert_rate":[false_alert_rate],
    "latency_seconds":[latency]
})

results.to_csv("data/processed/evaluation_metrics.csv", index=False)

print("Metrics saved!")