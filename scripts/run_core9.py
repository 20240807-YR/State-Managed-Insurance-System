import mlflow

try:
    mlflow.set_tracking_uri("http://host.docker.internal:5001")
    mlflow.set_experiment("core9_redecision")

    with mlflow.start_run(run_name="core9_dummy_run"):
        mlflow.log_param("core", 9)
        mlflow.log_metric("dummy_score", 0.9)

except Exception as e:
    print("MLflow logging skipped:", e)

print("Running Core 9 state-based re-decision pipeline")