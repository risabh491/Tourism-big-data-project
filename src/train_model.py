from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output" / "ml_dataset.csv"
MODEL_DIR = ROOT / "models"
OUT = ROOT / "output"
MODEL_DIR.mkdir(exist_ok=True)

def train():
    df = pd.read_csv(DATA)
    target = "International tourism, number of arrivals"
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year", target, "lag_1_year"]).copy()
    # Hold out the latest available year, train only on earlier years.
    test_year = int(df["Year"].max())
    train_df, test_df = df[df.Year < test_year], df[df.Year == test_year]
    if train_df.empty or test_df.empty:
        raise ValueError("Not enough distinct years for a chronological holdout.")
    features = ["Country Name", "Year", "lag_1_year", "GDP", "Population", "gdp_per_capita"]
    X_train, X_test = train_df[features], test_df[features]
    y_train, y_test = train_df[target], test_df[target]

    numeric = ["Year", "lag_1_year", "GDP", "Population", "gdp_per_capita"]
    categorical = ["Country Name"]
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("country", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OneHotEncoder(handle_unknown="ignore")), categorical)
    ])
    models = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2,
                                               random_state=42, n_jobs=-1)
    }
    rows, fitted = [], {}
    for name, estimator in models.items():
        pipe = make_pipeline(prep, estimator)
        pipe.fit(X_train, y_train)
        pred = np.maximum(pipe.predict(X_test), 0)
        rows.append({"model": name,
                     "test_year": test_year,
                     "MAE": mean_absolute_error(y_test, pred),
                     "RMSE": mean_squared_error(y_test, pred) ** 0.5,
                     "R2": r2_score(y_test, pred) if len(y_test) > 1 else np.nan})
        fitted[name] = (pipe, pred)
    scores = pd.DataFrame(rows).sort_values("RMSE")
    scores.to_csv(OUT / "model_comparison.csv", index=False)
    best = scores.iloc[0]["model"]
    pipe, pred = fitted[best]
    joblib.dump(pipe, MODEL_DIR / "tourism_arrivals_model.joblib")
    result = test_df[["Country Name", "Year", target]].copy()
    result["predicted_arrivals"] = pred
    result.to_csv(OUT / "tourism_predictions.csv", index=False)
    print(scores.to_string(index=False))
    print(f"\nSaved exploratory best model: {best}")
    print("Caution: this is a country-year observational dataset; latest-year holdout is not a monthly demand forecast.")
if __name__ == "__main__":
    train()
