import pandas as pd
import config 

def load_raw_data():
    """Loads the raw King County house data from the CSV file."""
    try:
        df = pd.read_csv(config.RAW_DATA_PATH)
        print(f"Successfully loaded data from {config.RAW_DATA_PATH}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at {config.RAW_DATA_PATH}")
        return None
    except Exception as e:
        print(f"ERROR: Could not load data. {e}")
        return None

if __name__ == '__main__':
    # For testing this module directly
    raw_df = load_raw_data()
    if raw_df is not None:
        print("\nFirst 5 rows of the raw data:")
        print(raw_df.head())
        print("\nData Info:")
        raw_df.info()