from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_cleaning import load_and_clean
from tourism_analysis import run_analysis
from visualize_tourism import make_charts

def main():
    print("=== Tourism Data Analysis: Python pipeline ===")
    df = load_and_clean()
    run_analysis(df, ROOT / "output")
    make_charts(df, ROOT / "output" / "plots")
    print("\nCompleted. See the output/ folder.")
    print("Optional ML steps:")
    print("  python src/prepare_ml_dataset.py")
    print("  python src/train_model.py")

if __name__ == "__main__":
    main()
