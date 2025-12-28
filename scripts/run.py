import subprocess
import sys

def run(cmd):
    print("▶", " ".join(cmd))
    subprocess.run(cmd, check=True)

run([sys.executable, "scripts/run_core5.py"])
run([sys.executable, "scripts/run_core9.py"])
run([sys.executable, "scripts/run_compare.py"])

print("✅ Pipeline finished")