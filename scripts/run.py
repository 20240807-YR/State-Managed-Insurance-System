import os
import subprocess

core = os.getenv("CORE", "5").strip()

if core == "9":
    subprocess.check_call(["python", "scripts/run_core9.py"])
else:
    subprocess.check_call(["python", "scripts/run_core5.py"])