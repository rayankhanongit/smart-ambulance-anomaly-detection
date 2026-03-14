from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Smart Ambulance Monitoring API")

# -----------------------------
# Input Schema
# -----------------------------

class Vitals(BaseModel):

    heart_rate: float
    spo2: float
    bp_sys: float
    bp_dia: float
    motion: float


# -----------------------------
# Risk Functions
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
# Prediction Endpoint
# -----------------------------

@app.post("/predict")

def predict(vitals: Vitals):

    hr_score = hr_risk(vitals.heart_rate)
    spo2_score = spo2_risk(vitals.spo2)
    bp_score = bp_risk(vitals.bp_sys)

    risk_score = (
        0.4 * hr_score +
        0.4 * spo2_score +
        0.2 * bp_score
    )

    confidence = max(0, 1 - vitals.motion)

    anomaly = risk_score > 0.6

    return {
        "risk_score": round(risk_score,3),
        "confidence": round(confidence,3),
        "anomaly_flag": anomaly
    }