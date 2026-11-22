# Tourism Data Analysis — Python Edition

This version is adapted to the uploaded `Tourism-Data-Analysis-main` archive and its actual World Bank-derived datasets.

## Stack
Python, pandas, NumPy, Matplotlib, scikit-learn. No Java, C++, Scala, or PySpark is used. Power BI can import the CSV files from `output/`.

## Dataset supplied
- `Data Sets/world_data.csv`: 2,926 country-year rows with arrivals, departures, GDP, and population.
- `Data Sets/country_code.csv`: country names and ISO codes.
- `Data Sets/tourism_data.xlsx`: original workbook included in the archive.

The notebook describes World Bank indicators for 2010–2020. The scripts use the supplied CSV rather than requiring an API download.

## Setup (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## Optional exploratory ML
```powershell
python src/prepare_ml_dataset.py
python src/train_model.py
```

The ML task predicts annual international arrivals by country using the previous available year's arrivals and country/economic fields. It is an exploratory country-year model, not monthly travel-demand forecasting. The source data spans a limited period and contains missing observations; results should not be represented as robust forecasts without more continuous data and a separate future holdout.

## Outputs
- `output/tourism_cleaned.csv` and `.xlsx`
- inbound/outbound country rankings
- global yearly trends
- India and top-10 country data
- PNG charts in `output/plots/`
- optional `output/ml_dataset.csv`, `model_comparison.csv`, `tourism_predictions.csv`
- optional serialized model in `models/`

## Power BI
Use **Get Data → Text/CSV** and import the output CSVs. Useful report pages: Global Overview, Inbound Tourism, Outbound Tourism, Country Comparison, India Profile. The supplied dataset does not contain hotel occupancy, so do not create occupancy KPIs without adding a separate reliable occupancy dataseet.
