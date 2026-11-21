from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data Sets"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

def load_and_clean():
    world_path = DATA / "world_data.csv"
    codes_path = DATA / "country_code.csv"
    if not world_path.exists() or not codes_path.exists():
        raise FileNotFoundError("Expected Data Sets/world_data.csv and country_code.csv")

    world = pd.read_csv(world_path)
    codes = pd.read_csv(codes_path)

    world = world.rename(columns={
        "country": "Country Name",
        "date": "Year",
        "GDP (current US$)": "GDP"
    })
    required = [
        "Country Name", "Year",
        "International tourism, number of arrivals",
        "International tourism, number of departures",
        "GDP", "Population"
    ]
    missing = [c for c in required if c not in world.columns]
    if missing:
        raise ValueError(f"Missing columns in world_data.csv: {missing}")

    world["Year"] = pd.to_numeric(world["Year"], errors="coerce")
    numeric = required[2:]
    for col in numeric:
        world[col] = pd.to_numeric(world[col], errors="coerce")

    world = world.dropna(subset=["Country Name", "Year"]).copy()
    world["Year"] = world["Year"].astype(int)
    # Keep unavailable indicators as null; do not replace missing observations with zero.
    merged = world.merge(codes, on="Country Name", how="left")
    merged = merged.drop_duplicates(subset=["Country Name", "Year"])
    merged = merged.sort_values(["Country Name", "Year"])
    merged["GDP per Capita"] = merged["GDP"] / merged["Population"].where(merged["Population"] > 0)
    merged.to_csv(OUT / "tourism_cleaned.csv", index=False)
    merged.to_excel(OUT / "tourism_cleaned.xlsx", index=False)
    return merged

if __name__ == "__main__":
    df = load_and_clean()
    print(f"Cleaned {len(df):,} country-year records.")
    print(f"Countries with codes: {df['Country Code'].notna().sum():,}")
    print(f"Saved to {OUT}")
