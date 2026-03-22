import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# Load data
df = pd.read_csv("data/salary_data.csv")

# Features and target
X = df[["years_experience", "education_level"]]
y = df["salary"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.4f}")

# MLflow tracking (we'll go deeper on Day 3 — today just run it)
mlflow.set_experiment("salary-prediction")

with mlflow.start_run():
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(model, "model")
    print("Run logged to MLflow!")