import os
import pandas as pd

BASE_DIR = os.getenv(
    "BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

INPUT_DIR = os.getenv("INPUT_DIR", os.path.join(BASE_DIR, "data/derived"))
DERIVED_DIR = os.getenv("DERIVED_DIR", os.path.join(BASE_DIR, "data/derived"))

CORE5_PATH = os.path.join(INPUT_DIR, "core5_decision_log.csv")

print("BASE_DIR:", BASE_DIR)
print("CORE5_PATH:", CORE5_PATH)

core5_df = pd.read_csv(CORE5_PATH)

core5_df["date"] = pd.to_datetime(core5_df["date"])

print("core5_df shape (raw):", core5_df.shape)
print("core5_df cols:", list(core5_df.columns))

key_cols = ["asset_id", "date", "t_index"]

dup_cnt = (
    core5_df
    .groupby(key_cols)
    .size()
    .reset_index(name="cnt")
    .query("cnt > 1")
)

print("duplicate key rows:", len(dup_cnt))


before = len(core5_df)

core5_df = core5_df.drop_duplicates(
    subset=key_cols,
    keep="last"
)

after = len(core5_df)

print("Core5 dedup:", before, "→", after)

assert (
    core5_df.duplicated(subset=key_cols).sum() == 0
), "❌ Core5 key duplicate still exists"

OUT_PATH = os.path.join(DERIVED_DIR, "core5_decision_log.csv")

core5_df.to_csv(OUT_PATH, index=False)

print("saved:", OUT_PATH)
print("final shape:", core5_df.shape)