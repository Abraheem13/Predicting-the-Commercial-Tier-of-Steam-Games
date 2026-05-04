"""
Stage 1 - Initial inspection.

Loads from games.json (CSV has column-shift corruption from unescaped commas).
"""
import json
import pandas as pd
from utils import DATA_RAW, DATA_INTERIM, FIGURES

json_path = DATA_RAW / "games.json"
print(f"Loading {json_path} (~800 MB, will take ~20-40s)...")
with open(json_path, "r", encoding="utf-8") as f:
    raw = json.load(f)
print(f"Loaded {len(raw):,} records from JSON")

records = []
for app_id, game in raw.items():
    game = dict(game)
    game["appid"] = app_id
    records.append(game)

df = pd.DataFrame(records)
print(f"\nShape            : {df.shape}")
print(f"Memory (deep)    : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

print("\nDtypes:")
print(df.dtypes)
print("\nColumns:")
print(df.columns.tolist())

sample_cols = [c for c in
               ["appid", "name", "release_date", "estimated_owners",
                "peak_ccu", "price", "required_age",
                "positive", "negative", "metacritic_score"]
               if c in df.columns]
print("\nSample rows for key fields:")
print(df[sample_cols].head(5).to_string())

print("\nestimated_owners dtype:", df["estimated_owners"].dtype)
print("estimated_owners sample:", df["estimated_owners"].unique()[:10])
print("\nprice dtype:", df["price"].dtype)
print("price sample:", df["price"].sample(10, random_state=0).tolist())
print("\nrequired_age value counts (top 10):")
print(df["required_age"].value_counts().head(10))

print("\nNumeric summary:")
print(df.select_dtypes(include=["number"]).describe().T)

print("\nNaN counts (top 20):")
print(df.isna().sum().sort_values(ascending=False).head(20))

# Normalise every object column for parquet compatibility.
# - list/dict cells get JSON-encoded
# - everything else gets stringified (covers mixed int/str like score_rank)
def _normalise(x):
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    if isinstance(x, (list, dict)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(_normalise)

parquet_out = DATA_INTERIM / "raw_from_json.parquet"
df.to_parquet(parquet_out, index=False)
print(f"\nSaved -> {parquet_out}")

summary_path = FIGURES / "01_inspection_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"Source: games.json (CSV had column-shift corruption)\n")
    f.write(f"Records: {len(df):,}\n")
    f.write(f"Columns: {df.shape[1]}\n")
    f.write(f"Memory (deep): {df.memory_usage(deep=True).sum() / 1e6:.1f} MB\n\n")
    f.write("Dtypes:\n")
    f.write(df.dtypes.to_string())
    f.write("\n\nNaN counts (top 20):\n")
    f.write(df.isna().sum().sort_values(ascending=False).head(20).to_string())
print(f"Summary -> {summary_path}")
