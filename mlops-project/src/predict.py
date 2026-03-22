import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize app
app = FastAPI(title="Salary Prediction API")

# Load best model from MLflow (RandomForest run)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load the latest RandomForest model
model = mlflow.sklearn.load_model("models:/RandomForest/latest") if False else None

# Fallback — load directly from runs
import mlflow.tracking
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("salary-prediction")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="params.model_type = 'RandomForest'",
    order_by=["metrics.r2_score DESC"],
    max_results=1
)
best_run_id = runs[0].info.run_id
model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
print(f"Loaded RandomForest model from run: {best_run_id}")

# Define input schema
class SalaryInput(BaseModel):
    years_experience: float
    education_level: int

# Define output schema
class SalaryOutput(BaseModel):
    predicted_salary: float
    model_used: str
    run_id: str

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForest"}

# Prediction endpoint
@app.post("/predict", response_model=SalaryOutput)
def predict(data: SalaryInput):
    input_df = pd.DataFrame([{
        "years_experience": data.years_experience,
        "education_level":  data.education_level
    }])
    prediction = model.predict(input_df)[0]
    return SalaryOutput(
        predicted_salary=round(prediction, 2),
        model_used="RandomForest",
        run_id=best_run_id
    )