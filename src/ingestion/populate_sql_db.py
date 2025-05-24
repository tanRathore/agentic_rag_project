import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

current_file_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from src import config as app_config

def get_db_engine():
    """Creates and returns a SQLAlchemy engine."""
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env')) 
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set.")
    
    print(f"Connecting to SQL database at: {db_url.replace(os.getenv('POSTGRES_PASSWORD', 'password'), '********')}")
    try:
        engine = create_engine(db_url)
        # Test connection
        with engine.connect() as connection:
            print("Successfully connected to PostgreSQL database!")
        return engine
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        raise

def create_table_and_ingest_data(engine, table_name="house_sales", sample_size=None):
    """
    Creates a table (if it doesn't exist) and ingests data from kc_house_data.csv.
    Args:
        engine: SQLAlchemy engine instance.
        table_name: Name of the table to create/populate.
        sample_size: Number of rows to ingest from CSV (None for all).
    """
    print(f"Loading raw data from: {app_config.RAW_DATA_PATH}")
    try:
        df = pd.read_csv(app_config.RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at {app_config.RAW_DATA_PATH}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if sample_size is not None and sample_size > 0:
        df = df.head(sample_size)
        print(f"Using a sample of {sample_size} rows for SQL ingestion.")
    else:
        print(f"Preparing to ingest all {len(df)} rows into SQL.")
    try:
        df['date'] = pd.to_datetime(df['date']) 
    except Exception as e:
        print(f"Warning: Could not convert 'date' column to datetime: {e}. It will be ingested as text if it fails.")


    print(f"Attempting to ingest data into SQL table: {table_name}")
    try:
        with engine.connect() as connection:
            print(f"Dropping table '{table_name}' if it exists...")
            connection.execute(text(f"DROP TABLE IF EXISTS {table_name};"))
            connection.commit() 
            print("Table dropped (if existed).")
            df.to_sql(table_name, con=engine, if_exists='replace', index=False, chunksize=1000)
            result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name};"))
            count = result.scalar_one()
            connection.commit() 
            print(f"Successfully ingested {count} rows into table '{table_name}'.")

    except Exception as e:
        print(f"Error during SQL table creation or data ingestion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting SQL DB population script...")
    try:
        sql_engine = get_db_engine()
        create_table_and_ingest_data(sql_engine, sample_size=1000) 
        print("SQL DB population script finished.")
    except Exception as e:
        print(f"Script failed: {e}")