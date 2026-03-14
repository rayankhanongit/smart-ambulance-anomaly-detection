import pandas as pd
import numpy as np

# Load cleaned data
df = pd.read_csv("data/processed/vitals_clean.csv")

window_size = 10
features = []

for i in range(len(df) - window_size):

    window = df.iloc[i:i+window_size]

    feature = {
        "hr_mean": window["heart_rate"].mean(),
        "hr_std": window["heart_rate"].std(),
        "spo2_mean": window["spo2"].mean(),
        "spo2_std": window["spo2"].std(),
        "bp_sys_mean": window["bp_sys"].mean(),
        "motion_mean": window["motion"].mean()
    }

    features.append(feature)

feature_df = pd.DataFrame(features)

feature_df.to_csv("data/processed/features.csv", index=False)

print("Feature extraction completed!")