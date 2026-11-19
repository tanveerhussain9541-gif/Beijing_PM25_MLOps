from pathlib import Path
import sys
import pandas as pd
import numpy as np
import sklearn

print("MLOps Environment Verification")
print("Python:", sys.version.split()[0])
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
project_root = Path(__file__).resolve().parent.parent
folders = [
    "data/raw",
    "data/processed",
    "src",
    "models",
    "notebooks",
    "reports",
    "reports/figures",
    "tests",
]
for folder in folders:
    path = project_root / folder
    print(folder, "OK" if path.is_dir() else "MISSING")
