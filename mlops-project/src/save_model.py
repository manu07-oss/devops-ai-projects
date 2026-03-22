import mlflow
import mlflow.sklearn
import pickle
import os

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("salary-prediction")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="params.model_type = 'RandomForest'",
    order_by=["metrics.r2_score DESC"],
    max_results=1
)
best_run_id = runs[0].info.run_id
print(f"Best run: {best_run_id}")

# Load model from MLflow
model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

# Save to fixed path
os.makedirs("models", exist_ok=True)
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to models/best_model.pkl")