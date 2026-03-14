import pandas as pd
import matplotlib.pyplot as plt

# Load final predictions
df = pd.read_csv("data/processed/final_predictions.csv")

# Take a sample alert
alert_df = df[df["alert"] == True].head(1)

if alert_df.empty:
    print("No alerts found for explanation.")
    exit()

row = alert_df.iloc[0]

# Risk contributions
hr = row["hr_mean"]
spo2 = row["spo2_mean"]
bp = row["bp_sys_mean"]

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

hr_score = 0.4 * hr_risk(hr)
spo2_score = 0.4 * spo2_risk(spo2)
bp_score = 0.2 * bp_risk(bp)

labels = ["Heart Rate", "SpO2", "Blood Pressure"]
values = [hr_score, spo2_score, bp_score]

plt.figure()
plt.bar(labels, values)
plt.title("Risk Contribution for Alert")
plt.ylabel("Contribution")
plt.savefig("plots/risk_explainability.png")

print("Explainability plot saved!")