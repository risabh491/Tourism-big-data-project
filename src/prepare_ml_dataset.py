from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "tourism_cleaned.csv"
DEST = ROOT / "output" / "ml_dataset.csv"
TARGET = "International tourism, number of arrivals"

def prepare():
    df = pd.read_csv(SOURCE)
    # This source is annual, country-level data, not monthly destination demand.
    df = df[["Country Name", "Year", TARGET, "GDP", "Population"]].copy()
    df = df.dropna(subset=["Country Name", "Year", TARGET])
    df = df[df[TARGET] >= 0].sort_values(["Country Name", "Year"])
    df["lag_1_year"] = df.groupby("Country Name")[TARGET].shift(1)
    df["growth_pct"] = df.groupby("Country Name")[TARGET].pct_change() * 100
    df["gdp_per_capita"] = df["GDP"] / df["Population"].where(df["Population"] > 0)
    df.to_csv(DEST, index=False)
    print(f"Saved {len(df):,} records to {DEST}")
    print("Rows without previous-year lag are retained in the CSV; model training filters them.")
    return df

if __name__ == "__main__":
    prepare()
