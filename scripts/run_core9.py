import os
from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = Path(
    os.getenv("INPUT_DIR", BASE_DIR / "data" / "derived")
)
DERIVED_DIR = Path(
    os.getenv("DERIVED_DIR", BASE_DIR / "data" / "derived")
)

CORE5_PATH = INPUT_DIR / "core5_decision_log.csv"
MUHSM_PATH = INPUT_DIR / "muHSM_state_monitor.csv"

print("BASE_DIR:", BASE_DIR)
print("CORE5_PATH:", CORE5_PATH)
print("MUHSM_PATH:", MUHSM_PATH)


core5_df = pd.read_csv(CORE5_PATH)
muhsm_df = pd.read_csv(MUHSM_PATH)

core5_df["date"] = pd.to_datetime(core5_df["date"])
muhsm_df["date"] = pd.to_datetime(muhsm_df["date"])

print("core5_df shape:", core5_df.shape)
print("core5_df cols:", list(core5_df.columns))

print("muhsm_df shape:", muhsm_df.shape)
print("muhsm_df cols:", list(muhsm_df.columns))


cols_vec = ["HSI", "HDR", "recovery_margin", "observability_score"]

def last_non_null(s: pd.Series):
    s = s.dropna()
    return s.iloc[-1] if len(s) else np.nan

muhsm_dedup = (
    muhsm_df
    .sort_values(["asset_id", "date"])
    .groupby(["asset_id", "date"], as_index=False)
    .agg({c: last_non_null for c in cols_vec})
)

dup_after = (
    muhsm_dedup
    .groupby(["asset_id", "date"])
    .size()
    .reset_index(name="cnt")
    .query("cnt > 1")
)

print("muHSM dup after dedup:", len(dup_after))
assert len(dup_after) == 0, "❌ muHSM dedup 실패 (중복 key 잔존)"


df = core5_df.merge(
    muhsm_dedup,
    on=["asset_id", "date"],
    how="left",
    validate="many_to_one"
)

print("merge rows:", len(df))
assert len(df) == len(core5_df), "❌ merge row mismatch"


df["intervention_flag_core9"] = (
    (df["HSI"] < 0.5) &
    (df["HDR"] < -0.05) &
    (df["observability_score"] > 0.6)
).astype(int)


before = len(df)

df = df.drop_duplicates(
    subset=["asset_id", "date", "t_index"],
    keep="last"
)

after = len(df)

print(f"Core9 dedup applied: {before} → {after}")

dup_final = (
    df.groupby(["asset_id", "date", "t_index"])
    .size()
    .reset_index(name="cnt")
    .query("cnt > 1")
)

assert len(dup_final) == 0, "❌ Core9 최종 key 중복 존재"


out_cols = [
    "asset_id", "date", "t_index",
    "state_value", "degradation_rate",
    "HSI", "HDR", "recovery_margin", "observability_score",
    "intervention_flag_core9", "stabilized"
]

OUT_PATH = DERIVED_DIR / "core9_state_based_decision_log.csv"

df[out_cols].to_csv(OUT_PATH, index=False)
print("saved:", OUT_PATH)
print("final shape:", df[out_cols].shape)