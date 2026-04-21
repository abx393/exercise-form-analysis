import sys
import pandas as pd
import matplotlib.pyplot as plt

def analyze_timestamps(file_path):
    try:
        df = pd.read_csv(file_path)

        if 'absolute_timestamp' not in df.columns:
            print("Error: 'absolute_timestamp' column not found.")
            return

        timestamps = df['absolute_timestamp']

        # Compute differences
        diffs = timestamps.diff().dropna()

        # Check ascending
        is_ascending = (diffs > 0).all()
        print(f"Strictly ascending: {is_ascending}")

        # Basic stats
        print("\nDiff statistics:")
        print(diffs.describe())

        # Count violations
        violations = diffs <= 0
        num_violations = violations.sum()
        print(f"\nNumber of non-ascending steps: {num_violations}")

        if num_violations > 0:
            violation_indices = diffs[violations].index.tolist()
            print("First few violation indices:", violation_indices[:10])

        # Plot diffs
        plt.figure()
        plt.plot(diffs.values)
        plt.title("Timestamp Differences")
        plt.xlabel("Sample Index")
        plt.ylabel("Delta (absolute_timestamp)")
        plt.grid(True)

        # Optional: highlight violations
        if num_violations > 0:
            plt.scatter(
                diffs[violations].index,
                diffs[violations].values
            )

        plt.show()

    except Exception as e:
        print(f"Error reading file: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_timestamps.py <file.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    analyze_timestamps(file_path)

