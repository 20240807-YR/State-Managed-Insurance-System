import mlflow

mlflow.set_tracking_uri("http://host.docker.internal:5001")
mlflow.set_experiment("core5_decision")

with mlflow.start_run(run_name="core5_dummy_run"):
    mlflow.log_param("core", 5)
    mlflow.log_metric("dummy_score", 1.0)

print("Running Core 5 decision pipeline")