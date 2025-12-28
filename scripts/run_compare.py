import os
import pandas as pd
import mlflow

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DERIVED_DIR = os.getenv(
    "DERIVED_DIR",
    os.path.join(BASE_DIR, "data", "derived")
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5001"
)

CORE5_PATH = os.path.join(DERIVED_DIR, "core5_decision_log.csv")
CORE9_PATH = os.path.join(DERIVED_DIR, "core9_state_based_decision_log.csv")
OUT_PATH   = os.path.join(DERIVED_DIR, "core9_final_summary.csv")

print("BASE_DIR:", BASE_DIR)
print("CORE5_PATH:", CORE5_PATH)
print("CORE9_PATH:", CORE9_PATH)
print("MLFLOW_TRACKING_URI:", MLFLOW_TRACKING_URI)

core5_df = pd.read_csv(CORE5_PATH)
core9_df = pd.read_csv(CORE9_PATH)

print("core5 shape:", core5_df.shape)
print("core9 shape:", core9_df.shape)

core5_cmp = core5_df.rename(
    columns={"intervention_flag": "intervention_core5"}
)

core9_cmp = core9_df.rename(
    columns={"intervention_flag_core9": "intervention_core9"}
)

cmp = core5_cmp.merge(
    core9_cmp,
    on=["asset_id", "date", "t_index"],
    how="inner",
    validate="one_to_one"
)

print("cmp shape:", cmp.shape)

def toggle_rate(series):
    return (series.diff().abs() > 0).mean()

toggle_core5 = (
    cmp.groupby("asset_id")["intervention_core5"]
    .apply(toggle_rate)
    .mean()
)

toggle_core9 = (
    cmp.groupby("asset_id")["intervention_core9"]
    .apply(toggle_rate)
    .mean()
)

false_core5 = cmp[
    (cmp["intervention_core5"] == 1) & (cmp["stabilized_x"] == False)
]

false_core9 = cmp[
    (cmp["intervention_core9"] == 1) & (cmp["stabilized_y"] == False)
]

stab_core5 = cmp[cmp["intervention_core5"] == 1]["stabilized_x"].mean()
stab_core9 = cmp[cmp["intervention_core9"] == 1]["stabilized_y"].mean()

summary = pd.DataFrame({
    "case": ["Core5", "Core9"],
    "toggle_rate": [toggle_core5, toggle_core9],
    "false_intervention": [len(false_core5), len(false_core9)],
    "stabilization_rate": [stab_core5, stab_core9],
})

summary.to_csv(OUT_PATH, index=False)
print("saved:", OUT_PATH)
print(summary)

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("decision_layer_comparison")

    with mlflow.start_run(run_name="Core5_decision"):
        mlflow.log_param("core", 5)
        mlflow.log_param("decision_type", "prediction_based")
        mlflow.log_param("key_definition", "asset_id,date,t_index")

        mlflow.log_metric("toggle_rate", toggle_core5)
        mlflow.log_metric("false_intervention", len(false_core5))
        mlflow.log_metric("stabilization_rate", stab_core5)


    with mlflow.start_run(run_name="Core9_state_based_decision"):
        mlflow.log_param("core", 9)
        mlflow.log_param("decision_type", "state_based")
        mlflow.log_param("state_model", "muHSM")
        mlflow.log_param("key_definition", "asset_id,date,t_index")

        mlflow.log_metric("toggle_rate", toggle_core9)
        mlflow.log_metric("false_intervention", len(false_core9))
        mlflow.log_metric("stabilization_rate", stab_core9)

    print("✅ MLflow Phase 6 logging completed")

except Exception as e:

    print("⚠️ MLflow logging skipped:", e)