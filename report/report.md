Smart Ambulance Vital Monitoring System
Overview

This project implements an end-to-end machine learning pipeline for monitoring patient vital signals in a smart ambulance environment. The system processes streaming physiological signals such as heart rate, oxygen saturation (SpO₂), blood pressure, and motion signals to detect early signs of patient deterioration.

The system is designed to operate under real-world conditions where data may contain noise, motion artifacts, and missing values due to ambulance movement and sensor instability.

Data Generation

Synthetic physiological time-series data was generated to simulate a 30-minute ambulance transport scenario with one sample per second.

The dataset includes:

Heart Rate (HR)

SpO₂

Systolic Blood Pressure

Diastolic Blood Pressure

Motion/Vibration signal

Three main scenarios were simulated:

Normal Transport

Stable vital signs with small physiological variability.

Ambulance Motion Artifacts

Vehicle bumps and movement introduce sudden spikes in HR and drops in SpO₂.

Patient Deterioration

Gradual increase in heart rate and decrease in SpO₂ over time representing physiological deterioration.

Additional noise sources included:

Sensor dropouts (missing values)

Random measurement artifacts

Motion-induced signal distortion

Artifact Detection and Signal Cleaning

Before anomaly detection, signal artifacts were removed using rule-based preprocessing.

Artifact handling steps:

Motion-based filtering
High motion combined with sudden SpO₂ drops were classified as sensor artifacts.

Physiological bounds filtering
Values outside realistic medical ranges were removed.

Missing data handling
Missing sensor readings were filled using interpolation.

Signal smoothing
Rolling window smoothing reduced measurement noise.

This preprocessing stage significantly improved signal reliability before feeding data into the anomaly detection model.

Anomaly Detection Model

An Isolation Forest model was used for anomaly detection.

Isolation Forest is suitable for this problem because:

It does not require labeled training data

It detects statistical deviations from normal physiological patterns

It performs well for time-series anomaly detection

Features were extracted using sliding time windows (10-second windows).

Extracted features included:

Mean heart rate

Heart rate variability

Mean SpO₂

SpO₂ variability

Mean systolic blood pressure

Mean motion signal

The model identifies abnormal physiological behavior rather than simple threshold violations.

Risk Scoring Logic

To simulate medical triage behavior, a weighted risk score was implemented combining multiple vital signals.

Risk Score Calculation:

Risk = 0.4 × HR Risk + 0.4 × SpO₂ Risk + 0.2 × BP Risk

Risk components are calculated based on clinical thresholds.

Additionally, a confidence score is derived from motion levels:

Confidence = 1 − motion_level

Higher motion reduces confidence because signals may be unreliable.

Alerts are triggered when:

risk_score > 0.6 OR anomaly_flag = True
Alert Evaluation

Alert quality was evaluated using the following metrics:

Precision

Recall

False Alert Rate

Alert Latency

Example results from the simulated dataset:

Metric	Value
Precision	0.39
Recall	0.12
False Alert Rate	0.61
Alert Latency	212 seconds

In medical monitoring systems, high recall is more important than precision, because missing a critical deterioration event can have severe consequences.

Failure Analysis
Case 1 — False Alerts Due to Motion Artifacts

Motion spikes from ambulance movement occasionally caused abnormal signal readings that were incorrectly detected as anomalies.

Cause:
Motion artifacts can mimic physiological anomalies.

Improvement:
Advanced artifact filtering or signal quality indices could reduce these false positives.

Case 2 — Missed Gradual Deterioration

Gradual physiological changes may not appear statistically abnormal to an unsupervised anomaly detector.

Cause:
Isolation Forest detects statistical outliers but may miss slow trends.

Improvement:
Adding trend-based features such as heart rate slope or SpO₂ decline rate could improve early detection.

Case 3 — Sensor Data Loss

Temporary sensor failures caused missing data that could reduce detection reliability.

Cause:
Medical sensors may lose contact during patient movement.

Improvement:
Sensor reliability monitoring and redundancy across multiple sensors could mitigate this issue.

Safety-Critical Considerations
Most Dangerous Failure Mode

The most dangerous failure mode is missing a true deterioration event, where the system fails to alert medical staff during a critical change in patient condition.

This could delay medical intervention and potentially endanger the patient.

Reducing False Alerts Without Missing Deterioration

Possible strategies include:

Combining multiple vital signals instead of relying on a single metric

Incorporating signal quality indicators

Using temporal trend analysis

Applying ensemble anomaly detection models

What Should Never Be Fully Automated

In medical AI systems, certain decisions should always remain under human supervision:

Final clinical diagnosis

Medication administration

Life-support decisions

Emergency triage prioritization

AI systems should function as decision-support tools rather than decision-makers.

System Architecture

The system consists of the following pipeline:

Vital Sensors
        ↓
Data Generation / Streaming
        ↓
Artifact Detection & Cleaning
        ↓
Feature Engineering
        ↓
Anomaly Detection Model
        ↓
Risk Scoring System
        ↓
Alert Generation
        ↓
FastAPI ML Service

The API allows real-time prediction using patient vital inputs.