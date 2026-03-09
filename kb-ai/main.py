import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.outlier_detector import OutlierDetector
from scripts.missing_data_analyzer import MissingDataAnalyzer
from scripts.duplicate_detector import DuplicateDetector
from scripts.data_type_validator import DataTypeValidator

DATA_FILE = Path(__file__).parent / "test-data.csv"


def compute_stats(series, window=10):
    values = series.dropna()
    arr = values.values
    rolling = values.rolling(window)
    return {
        "mean":         float(np.mean(series)),
        "stddev":       float(np.std(series)),
        "p95":          float(np.percentile(series, 95)),
        "p99":          float(np.percentile(series, 99)),
        "max":          float(np.max(series)),
        "max_delta":     float(values.diff().abs().max()),
        "rolling_mean": float(rolling.mean().iloc[-1]),
        "rolling_std":  float(rolling.std().mean()),
    }


def print_stats(column, stats):
    label = column[:60] + "..." if len(column) > 60 else column
    print(f"\nColumn: {label}")
    print("=" * 45)
    for key, val in stats.items():
        print(f"  {key:<8}  {val:>14,.2f}")
    print()


if __name__ == "__main__":
    schema = {
        "Time": {"type": "date", "format": "%H:%M:%S"},
        "Cluster": {"type": "float"}
    }

    domain_rules = {
        "Cluster": {"min": 0}
    }

    missing = MissingDataAnalyzer(DATA_FILE)
    missing.analyze()
    missing.print_report()

    data_type = DataTypeValidator(DATA_FILE, schema)
    data_type.validate()
    data_type.print_report()

    outliers = OutlierDetector(DATA_FILE, domain_rules=domain_rules)
    outliers.analyze_all(statistical_method="iqr", threshold=1.5)
    outliers.print_report()

    dupes = DuplicateDetector(DATA_FILE, key_columns=["Time"], fuzzy_threshold=0)
    dupes.analyze_all(include_fuzzy=False)
    dupes.print_report()

    # not needed for metrics
    #checker = ConsistencyChecker(DATA_FILE)
    #checker.validate_all()
    #checker.print_report()

    df = pd.read_csv(DATA_FILE)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        stats = compute_stats(df[col])
        print_stats(col, stats)


    df["Time"] = pd.to_date=time(df["Time"])

    plt.figure(figsize=(12,4))
    plt.plot(df["Time"], df.iloc[:,1])

    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()
