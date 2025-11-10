import pandas as pd


def csv_to_dataframe(csv_path: str) -> pd.DataFrame:
    """Reads a CSV file and returns it as a pandas DataFrame."""
    return pd.read_csv(csv_path)