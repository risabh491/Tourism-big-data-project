import pandas as pd

def run_analysis(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    arrivals = "International tourism, number of arrivals"
    departures = "International tourism, number of departures"

    inbound = (df.groupby("Country Name", as_index=False)[arrivals]
                 .sum(min_count=1).sort_values(arrivals, ascending=False))
    inbound["rank"] = range(1, len(inbound) + 1)
    inbound.to_csv(output_dir / "inbound_country_ranking.csv", index=False)

    outbound = (df.groupby("Country Name", as_index=False)[departures]
                  .sum(min_count=1).sort_values(departures, ascending=False))
    outbound["rank"] = range(1, len(outbound) + 1)
    outbound.to_csv(output_dir / "outbound_country_ranking.csv", index=False)

    yearly = (df.groupby("Year", as_index=False)[[arrivals, departures, "GDP", "Population"]]
                .sum(min_count=1).sort_values("Year"))
    yearly.to_csv(output_dir / "global_yearly_trends.csv", index=False)

    top10 = inbound.head(10)["Country Name"]
    df[df["Country Name"].isin(top10)].to_csv(output_dir / "top10_inbound_trends.csv", index=False)

    india = df[df["Country Name"].eq("India")].sort_values("Year")
    india.to_csv(output_dir / "india_tourism_trends.csv", index=False)

    print("Top 10 inbound tourism destinations (sum of available years):")
    print(inbound.head(10).to_string(index=False))
    print("\nNote: totals sum available annual observations; missing values are ignored.")
