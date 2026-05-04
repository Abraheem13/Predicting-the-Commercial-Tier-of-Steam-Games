"""
One-off diagnostic to investigate column names and dtype oddities
before fixing 02_clean.py. Run once, review output, then we adapt.
"""
import pandas as pd
from pathlib import Path

CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "games.csv"

# Try reading first 5 lines raw to see what columns actually look like
print("=" * 70)
print("RAW FIRST LINE (header):")
print("=" * 70)
with open(CSV, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i == 0:
            print(line[:2000])
            break

# Load the CSV
df = pd.read_csv(CSV, low_memory=False)
print(f"\nShape: {df.shape}")
print(f"\nAll columns ({len(df.columns)}):")
for i, c in enumerate(df.columns):
    print(f"  {i:2d}  '{c}'  dtype={df[c].dtype}")

# Investigate Price oddness
print("\n" + "=" * 70)
print("PRICE INVESTIGATION")
print("=" * 70)
print(f"Price dtype: {df['Price'].dtype}")
print(f"Price value counts (top 20):")
print(df["Price"].value_counts().head(20))
print(f"\nUnique non-integer-looking Price values (sample):")
# If Price is int, there are none; but check values just above integers
sample = df["Price"].sample(20, random_state=0)
print(sample.tolist())

# Investigate Required age
print("\n" + "=" * 70)
print("REQUIRED AGE INVESTIGATION")
print("=" * 70)
print(df["Required age"].describe())
print(f"\nValue counts (top 10):")
print(df["Required age"].value_counts().head(10))
print(f"\nRows where Required age > 30 (sample of 5):")
weird = df[df["Required age"] > 30]
print(f"Count: {len(weird)}")
if len(weird) > 0:
    print(weird[["Name", "Required age", "Price"]].head())

# Estimated owners format
print("\n" + "=" * 70)
print("ESTIMATED OWNERS INVESTIGATION")
print("=" * 70)
print(f"Dtype: {df['Estimated owners'].dtype}")
print(f"\nValue counts (top 20):")
print(df["Estimated owners"].value_counts().head(20))
print(f"\nSample unique values: {df['Estimated owners'].unique()[:10]}")