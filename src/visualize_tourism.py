import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ARR = "International tourism, number of arrivals"
DEP = "International tourism, number of departures"

def make_charts(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    top = (df.groupby("Country Name")[ARR].sum(min_count=1)
             .dropna().nlargest(15).sort_values())
    if not top.empty:
        fig, ax = plt.subplots(figsize=(11, 7))
        top.plot(kind="barh", ax=ax)
        ax.set(title="Top 15 countries by inbound tourist arrivals",
               xlabel="Arrivals (sum of available annual observations)", ylabel="")
        fig.tight_layout()
        fig.savefig(output_dir / "top_inbound_countries.png", dpi=160)
        plt.close(fig)

    global_year = df.groupby("Year")[[ARR, DEP]].sum(min_count=1).sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    global_year.plot(ax=ax, marker="o")
    ax.set(title="Global tourism arrivals and departures",
           xlabel="Year", ylabel="Reported people / trips")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output_dir / "global_tourism_trends.png", dpi=160)
    plt.close(fig)

    india = df[df["Country Name"].eq("India")].sort_values("Year")
    if not india.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(india["Year"], india[ARR], marker="o", label="Inbound arrivals")
        ax.plot(india["Year"], india[DEP], marker="o", label="Outbound departures")
        ax.set(title="India: international tourism", xlabel="Year", ylabel="Reported people / trips")
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(output_dir / "india_tourism.png", dpi=160)
        plt.close(fig)
