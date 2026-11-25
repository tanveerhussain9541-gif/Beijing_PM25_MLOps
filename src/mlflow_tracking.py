from pathlib import Path
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Project configuration: change these only if your files differ.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_features.csv"
TARGET = "target_pm25_next_hour"
# 2. Read the prepared training and test data.
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
if TARGET not in train_df.columns or TARGET not in test_df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found.")
# 3. Use numeric input columns only for this baseline.
numeric_columns = train_df.select_dtypes(include="number").columns.tolist()
feature_columns = [c for c in numeric_columns if c != TARGET]
if not feature_columns:
    raise ValueError("No numeric feature columns were found.")
X_train = train_df[feature_columns]
y_train = train_df[TARGET]
X_test = test_df[feature_columns]
y_test = test_df[TARGET]
# 4. Build preprocessing and regression as one pipeline.
alpha_value = 1.0
model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=alpha_value)),
    ]
)
# 5. Configure local MLflow tracking and experiment.
tracking_path = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{tracking_path.as_posix()}")
mlflow.set_experiment("Beijing_PM25_Forecasting")
# 6. Train, evaluate and record one MLflow run.
with mlflow.start_run(run_name="Ridge_baseline_alpha_1"):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mlflow.log_param("model", "Ridge")
    mlflow.log_param("alpha", alpha_value)
    mlflow.log_param("feature_count", len(feature_columns))
    mlflow.log_param("target", TARGET)
    mlflow.log_param("training_rows", len(train_df))
    mlflow.log_param("test_rows", len(test_df))
    mlflow.log_metric("MAE", float(mae))
    mlflow.log_metric("RMSE", float(rmse))
    mlflow.set_tag("station", "Aotizhongxin")
    mlflow.set_tag("forecast_horizon", "1 hour")
    mlflow.set_tag("purpose", "baseline experiment")
    # Save predictions as a run artifact.
    results = pd.DataFrame(
        {"actual_PM2.5_next_hour": y_test, "predicted_PM2.5_next_hour": predictions}
    )
    results_path = PROJECT_ROOT / "predictions_for_mlflow.csv"
    results.to_csv(results_path, index=False)
    mlflow.log_artifact(str(results_path))
    # Log the fitted scikit-learn pipeline as a model artifact.
    mlflow.sklearn.log_model(
        model, artifact_path="ridge_model", skops_trusted_types=["numpy.dtype"]
    )
    print("MLflow run completed")
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("Features used:", feature_columns)
