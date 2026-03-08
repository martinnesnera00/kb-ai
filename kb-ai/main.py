import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

DATA_FILE = Path(__file__).parent / "test-data.csv"


def compute_stats(series):
    values = series.dropna().values
    return {
        "mean":   float(np.mean(values)),
        "stddev": float(np.std(values)),
        "p95":    float(np.percentile(values, 95)),
        "p99":    float(np.percentile(values, 99)),
        "max":    float(np.max(values)),
        "delta":  float(np.max(values) - np.min(values)),
    }


def print_stats(column, stats):
    label = column[:60] + "..." if len(column) > 60 else column
    print(f"\nColumn: {label}")
    print("=" * 45)
    for key, val in stats.items():
        print(f"  {key:<8}  {val:>14,.2f}")
    print()


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        stats = compute_stats(df[col])
        print_stats(col, stats)
