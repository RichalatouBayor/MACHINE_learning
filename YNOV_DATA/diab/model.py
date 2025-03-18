from pydantic import BaseModel
from typing import List

class PredictionInput(BaseModel):
  age: float
  location: float
  race_african_american: float
  race_asian: float
  race_caucasian: float
  race_hispanic: float
  race_other: float
  hypertension: float
  heart_disease: float
  smoking_history: float
  bmi: float
  hbA1c_level: float
  blood_glucose_level: float
  Female: float
  Male: float