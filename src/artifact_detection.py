import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/raw/vitals_patient1.csv")

# Create a copy for cleaning
clean_df = df.copy()

# -----------------------------
# 1. Detect Motion Artifacts
# -----------------------------
# If motion is very high and SpO2 drops suddenly,
# it's likely a sensor artifact.

motion_threshold = 0.8

artifact_mask = (clean_df["motion"] > motion_threshold) & (clean_df["spo2"] < 90)

clean_df.loc[artifact_mask, "spo2"] = np.nan
clean_df.loc[artifact_mask, "heart_rate"] = np.nan

# -----------------------------
# 2. Remove Heart Rate Spikes
# -----------------------------
# HR above physiological limit likely artifact

hr_upper_limit = 180
clean_df.loc[clean_df["heart_rate"] > hr_upper_limit, "heart_rate"] = np.nan

# -----------------------------
# 3. Remove SpO2 Outliers
# -----------------------------
# SpO2 normally stays between 85 and 100

clean_df.loc[clean_df["spo2"] < 85, "spo2"] = np.nan

# -----------------------------
# 4. Handle Missing Data
# -----------------------------

clean_df["heart_rate"] = clean_df["heart_rate"].interpolate()
clean_df["spo2"] = clean_df["spo2"].interpolate()

clean_df["bp_sys"] = clean_df["bp_sys"].interpolate()
clean_df["bp_dia"] = clean_df["bp_dia"].interpolate()

# -----------------------------
# 5. Smooth Signals
# -----------------------------

clean_df["heart_rate"] = clean_df["heart_rate"].rolling(5, min_periods=1).mean()
clean_df["spo2"] = clean_df["spo2"].rolling(5, min_periods=1).mean()

# -----------------------------
# 6. Save Clean Data
# -----------------------------

clean_df.to_csv("data/processed/vitals_clean.csv", index=False)

print("Artifact cleaning completed!")

# -----------------------------
# 7. Plot Before vs After
# -----------------------------

plt.figure(figsize=(12,4))
plt.plot(df["heart_rate"], label="Before Cleaning")
plt.plot(clean_df["heart_rate"], label="After Cleaning")
plt.legend()
plt.title("Heart Rate Cleaning")
plt.savefig("plots/hr_cleaning.png")

plt.figure(figsize=(12,4))
plt.plot(df["spo2"], label="Before Cleaning")
plt.plot(clean_df["spo2"], label="After Cleaning")
plt.legend()
plt.title("SpO2 Cleaning")
plt.savefig("plots/spo2_cleaning.png")

print("Cleaning plots saved!")