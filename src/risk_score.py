import pandas as pd
import numpy as np

# Load anomaly predictions
df = pd.read_csv("data/processed/anomaly_predictions.csv")

# -----------------------------
# Risk component functions
# -----------------------------

def hr_risk(hr):

    if hr > 130:
        return 1.0
    elif hr > 110:
        return 0.7
    elif hr > 95:
        return 0.4
    else:
        return 0.1


def spo2_risk(spo2):

    if spo2 < 85:
        return 1.0
    elif spo2 < 90:
        return 0.8
    elif spo2 < 94:
        return 0.5
    else:
        return 0.1


def bp_risk(bp):

    if bp > 160:
        return 0.9
    elif bp > 140:
        return 0.6
    elif bp > 130:
        return 0.3
    else:
        return 0.1


# -----------------------------
# Calculate risk score
# -----------------------------

risk_scores = []

for _, row in df.iterrows():

    hr = row["hr_mean"]
    spo2 = row["spo2_mean"]
    bp = row["bp_sys_mean"]

    hr_score = hr_risk(hr)
    spo2_score = spo2_risk(spo2)
    bp_score = bp_risk(bp)

    # Weighted risk score
    risk = (
        0.4 * hr_score +
        0.4 * spo2_score +
        0.2 * bp_score
    )

    risk_scores.append(risk)

df["risk_score"] = risk_scores

# -----------------------------
# Confidence Score
# -----------------------------

df["confidence"] = 1 - df["motion_mean"]

# clamp values between 0 and 1
df["confidence"] = df["confidence"].clip(0,1)

# -----------------------------
# Alert Decision
# -----------------------------

df["alert"] = (
    (df["risk_score"] > 0.6) |
    (df["anomaly_flag"] == 1)
)

# -----------------------------
# Save results
# -----------------------------

df.to_csv("data/processed/final_predictions.csv", index=False)

print("Risk scoring completed!")
print("Total alerts:", df["alert"].sum())