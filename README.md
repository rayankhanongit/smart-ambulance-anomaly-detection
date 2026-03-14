🚑 Smart Ambulance Vital Monitoring System

AI/ML system for detecting early patient deterioration in ambulance transport using physiological time-series data.

This project simulates a real-world smart ambulance monitoring pipeline where patient vitals are continuously analyzed to detect anomalies and generate early warning alerts for medical staff.

📌 Project Overview

Ambulances operate in highly noisy environments where motion, sensor instability, and patient movement can distort physiological signals.

This project builds a robust ML pipeline capable of:

Handling noisy physiological signals

Detecting abnormal patient conditions

Generating risk-based alerts

Providing an API for real-time monitoring

The system processes the following vitals:

Signal	Description
Heart Rate (HR)	Patient heart beats per minute
SpO₂	Oxygen saturation level
Blood Pressure	Systolic and diastolic pressure
Motion	Ambulance vibration and patient movement
⚙️ System Pipeline
Vital Sensors
      ↓
Data Generation
      ↓
Artifact Detection
      ↓
Signal Cleaning
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
📊 Features
Synthetic Medical Data Simulation

Generates 30 minutes of patient vital signals including:

Normal transport conditions

Patient deterioration

Ambulance motion artifacts

Sensor noise

Missing data

Artifact Detection

Handles real-world signal issues:

Motion-induced SpO₂ drops

Heart rate spikes from bumps

Missing sensor data

Signal noise

Feature Engineering

Sliding window features extracted from time-series data:

HR mean and variability

SpO₂ mean and variability

Blood pressure trends

Motion levels

Anomaly Detection

Isolation Forest detects abnormal physiological behavior without labeled training data.

Risk Scoring Logic

Weighted triage-style risk score:

Risk = 0.4 × HR Risk + 0.4 × SpO₂ Risk + 0.2 × BP Risk

Alerts are triggered based on:

anomaly detection

physiological risk score

signal confidence

Alert Evaluation

System performance evaluated using:

Precision

Recall

False Alert Rate

Alert Latency

Example results from simulated dataset:

Metric	Value
Precision	0.39
Recall	0.12
False Alert Rate	0.61
Alert Latency	212 sec
🧠 Machine Learning Model

The system uses Isolation Forest, a robust unsupervised anomaly detection algorithm.

Advantages:

Works without labeled data

Effective for anomaly detection

Handles high-dimensional feature space

🌐 API Service

The trained model is exposed using FastAPI.

Start the API
uvicorn api.main:app --reload

API will run at:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs
Example API Request
POST /predict

Input:

{
  "heart_rate": 120,
  "spo2": 90,
  "bp_sys": 150,
  "bp_dia": 90,
  "motion": 0.2
}

Response:

{
  "risk_score": 0.72,
  "confidence": 0.8,
  "anomaly_flag": true
}
📁 Project Structure
smart-ambulance-anomaly-detection

api/
    main.py

src/
    data_generation.py
    artifact_detection.py
    feature_engineering.py
    anomaly_model.py
    risk_score.py
    evaluation.py

data/
    raw/
    processed/

plots/

report/
    report.md

requirements.txt
README.md
🛠️ Installation

Clone the repository:

git clone <repo-url>
cd smart-ambulance-anomaly-detection

Create virtual environment:

python -m venv venv

Activate environment:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
▶️ Running the Pipeline

Generate synthetic dataset:

python src/data_generation.py

Clean artifacts:

python src/artifact_detection.py

Extract features:

python src/feature_engineering.py

Train anomaly model:

python src/anomaly_model.py

Generate risk scores:

python src/risk_score.py

Evaluate alerts:

python src/evaluation.py
⚠️ Safety Considerations

AI-based medical monitoring systems must be used as decision-support tools, not replacements for medical professionals.

Critical decisions such as:

diagnosis

treatment

medication

life-support actions

must always remain under human supervision.

🚀 Future Improvements

Possible extensions for production systems:

Real patient datasets (PhysioNet)

Deep learning models for time-series prediction

Signal quality indices

Multi-patient monitoring

Real-time streaming pipeline

Explainable AI visualizations

👨‍💻 Author

AI/ML Engineering Assignment – Smart Ambulance Monitoring System