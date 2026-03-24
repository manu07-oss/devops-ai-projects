import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)
print("Model loaded successfully!")

app = FastAPI(title="Salary Prediction API")

Instrumentator().instrument(app).expose(app)

class SalaryInput(BaseModel):
    years_experience: float
    education_level: int

class SalaryOutput(BaseModel):
    predicted_salary: float
    model_used: str

@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForest"}

@app.post("/predict", response_model=SalaryOutput)
def predict(data: SalaryInput):
    input_df = pd.DataFrame([{
        "years_experience": data.years_experience,
        "education_level": data.education_level
    }])
    prediction = model.predict(input_df)[0]
    return SalaryOutput(
        predicted_salary=round(prediction, 2),
        model_used="RandomForest"
    )
