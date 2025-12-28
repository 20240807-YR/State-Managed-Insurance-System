import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

def run(script):
    script_path = BASE_DIR / "scripts" / script
    print("▶", sys.executable, script_path)
    subprocess.run([sys.executable, str(script_path)], check=True)

run("run_core5.py")
run("run_core9.py")
run("run_compare.py")

print("✅ Pipeline finished")