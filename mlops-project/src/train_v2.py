import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# Load data
df = pd.read_csv("data/salary_data.csv")
X = df[["years_experience", "education_level"]]
y = df["salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define 3 models to compare
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

mlflow.set_experiment("salary-prediction")

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):

        # Train
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, predictions)
        r2  = r2_score(y_test, predictions)

        # Log to MLflow
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(model, name="model")

        print(f"{model_name:25s} → MAE: {mae:8.2f} | R2: {r2:.4f}")

print("\nAll runs logged! Open MLflow UI to compare.")