import pandas as pd
from sklearn.ensemble import IsolationForest

# Load features
df = pd.read_csv("data/processed/features.csv")

# Train anomaly detection model
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(df)

# Predict anomalies
df["anomaly"] = model.predict(df)

# Convert labels (-1 = anomaly)
df["anomaly_flag"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)

df.to_csv("data/processed/anomaly_predictions.csv", index=False)

print("Anomaly detection completed!")
print("Total anomalies detected:", df["anomaly_flag"].sum())

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.plot(df["hr_mean"], label="HR mean")

anomalies = df[df["anomaly_flag"] == 1]

plt.scatter(anomalies.index, anomalies["hr_mean"], color="red", label="Anomaly")

plt.legend()
plt.title("Detected Anomalies")
plt.savefig("plots/anomaly_detection.png")