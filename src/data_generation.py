import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# simulation length
duration = 1800  # 30 minutes (1 sample per second)
time = np.arange(duration)

# -----------------------------
# 1. NORMAL BASELINE SIGNALS
# -----------------------------

np.random.seed(42)

hr = np.random.normal(80, 3, duration)       # heart rate
spo2 = np.random.normal(97, 0.5, duration)   # oxygen saturation
bp_sys = np.random.normal(120, 4, duration)  # systolic
bp_dia = np.random.normal(80, 3, duration)   # diastolic
motion = np.random.normal(0.2, 0.05, duration)

# -----------------------------
# 2. AMBULANCE MOTION EVENTS
# -----------------------------

for i in range(300, 340):
    motion[i] = 1.0
    hr[i] += np.random.normal(20,5)
    spo2[i] -= np.random.normal(10,2)

# -----------------------------
# 3. PATIENT DETERIORATION
# -----------------------------

for i in range(900,1200):

    hr[i] += (i-900)*0.03
    spo2[i] -= (i-900)*0.01
    bp_sys[i] += np.random.normal(5,2)

# -----------------------------
# 4. SENSOR ARTIFACTS
# -----------------------------

artifact_indices = np.random.choice(duration,20)

for idx in artifact_indices:

    spo2[idx] = np.random.uniform(60,80)
    hr[idx] = np.random.uniform(140,170)

# -----------------------------
# 5. MISSING DATA
# -----------------------------

missing = np.random.choice(duration,15)

for idx in missing:

    hr[idx] = np.nan
    spo2[idx] = np.nan

# -----------------------------
# 6. CREATE DATAFRAME
# -----------------------------

df = pd.DataFrame({
    "time": time,
    "heart_rate": hr,
    "spo2": spo2,
    "bp_sys": bp_sys,
    "bp_dia": bp_dia,
    "motion": motion
})

# -----------------------------
# 7. SAVE DATA
# -----------------------------

df.to_csv("data/raw/vitals_patient1.csv", index=False)

print("Dataset generated successfully!")

# -----------------------------
# 8. PLOT SIGNALS
# -----------------------------

plt.figure(figsize=(12,4))
plt.plot(df["heart_rate"])
plt.title("Heart Rate Signal")
plt.xlabel("Time")
plt.ylabel("HR")
plt.savefig("plots/hr_signal.png")

plt.figure(figsize=(12,4))
plt.plot(df["spo2"])
plt.title("SpO2 Signal")
plt.xlabel("Time")
plt.ylabel("SpO2")
plt.savefig("plots/spo2_signal.png")

plt.figure(figsize=(12,4))
plt.plot(df["motion"])
plt.title("Ambulance Motion")
plt.xlabel("Time")
plt.ylabel("Motion")
plt.savefig("plots/motion_signal.png")

print("Plots saved in plots folder")