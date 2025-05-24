import pandas as pd
import config

def select_and_clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects relevant features, handles missing values, and performs basic cleaning.
    """
    relevant_cols = [
        'id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
        'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
        'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
        'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'
    ]
    processed_df = df[relevant_cols].copy()
    for col in ['waterfront', 'view', 'sqft_basement', 'yr_renovated']:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].fillna(0)
    try:
        processed_df['sale_year'] = pd.to_datetime(processed_df['date']).dt.year
    except Exception as e:
        print(f"Warning: Could not parse 'date' column to extract year: {e}")
        processed_df['sale_year'] = None 
    for col in ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade', 'yr_built', 'yr_renovated', 'zipcode', 'sale_year']:
        if col in processed_df.columns:
            try:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0).astype(int)
            except Exception as e:
                 print(f"Warning: Could not convert column {col} to int: {e}")


    print(f"Selected features and handled basic missing values. Shape: {processed_df.shape}")
    return processed_df


def create_textual_description(row: pd.Series) -> str:
    """
    Creates a textual description for a single row (property) of the DataFrame.
    This description will be embedded.
    """
    desc_parts = []
    if pd.notna(row.get('price')) and pd.notna(row.get('sale_year')):
        desc_parts.append(f"Sold for ${int(row['price']):,} in {int(row['sale_year'])}.")
    elif pd.notna(row.get('price')):
        desc_parts.append(f"Priced at ${int(row['price']):,}.")

    core_features = []
    if pd.notna(row.get('bedrooms')) and row['bedrooms'] > 0:
        core_features.append(f"{int(row['bedrooms'])} bedrooms")
    if pd.notna(row.get('bathrooms')) and row['bathrooms'] > 0:
        core_features.append(f"{float(row['bathrooms']):.1f} bathrooms") 
    if pd.notna(row.get('sqft_living')) and row['sqft_living'] > 0:
        core_features.append(f"{int(row['sqft_living'])} sq ft of living space")

    if core_features:
        desc_parts.append(f"This property features {', '.join(core_features)}.")
    if pd.notna(row.get('floors')) and row['floors'] > 0:
        desc_parts.append(f"It has {float(row['floors']):.1f} floors.")
    if pd.notna(row.get('yr_built')) and row['yr_built'] > 0:
        desc_parts.append(f"Built in {int(row['yr_built'])}.")
    if pd.notna(row.get('yr_renovated')) and row['yr_renovated'] > 0: 
        desc_parts.append(f"Renovated in {int(row['yr_renovated'])}.")
    if pd.notna(row.get('zipcode')):
        desc_parts.append(f"Located in zipcode {int(row['zipcode'])}.")

    if pd.notna(row.get('waterfront')) and row['waterfront'] == 1:
        desc_parts.append("It boasts a waterfront view.")
    if pd.notna(row.get('view')) and row['view'] > 0: # View is rated 0-4
        desc_parts.append(f"It has a view rated {int(row['view'])} out of 4.")
    if pd.notna(row.get('condition')) and row['condition'] > 0: # Condition 1-5
        desc_parts.append(f"The property condition is rated {int(row['condition'])} out of 5.")
    if pd.notna(row.get('grade')) and row['grade'] > 0: # Grade 1-13
        desc_parts.append(f"Construction grade is {int(row['grade'])} out of 13.")
    if pd.notna(row.get('sqft_lot')) and row['sqft_lot'] > 0:
        desc_parts.append(f"The lot size is {int(row['sqft_lot'])} sq ft.")

    description = " ".join(desc_parts)
    return description.strip()


def preprocess_data_for_vectorization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline: selects features, cleans, and adds textual descriptions.
    """
    cleaned_df = select_and_clean_features(df)
    
    print("Generating textual descriptions for each property...")
    cleaned_df['description'] = cleaned_df.apply(create_textual_description, axis=1)
    
    return cleaned_df


if __name__ == '__main__':
    from src.ingestion.load_data import load_raw_data
    raw_df = load_raw_data()
    if raw_df is not None:
        print("\n--- Starting Preprocessing ---")
        processed_df_for_vec = preprocess_data_for_vectorization(raw_df.head(5)) 
        print("\n--- Preprocessing Complete ---")
        if processed_df_for_vec is not None:
            print("\nProcessed data with descriptions (sample):")
            for index, row in processed_df_for_vec.iterrows():
                print(f"\nID: {row['id']}")
                print(f"  Description: {row['description']}")
                print(f"  Price: {row.get('price', 'N/A')}, Bed: {row.get('bedrooms', 'N/A')}, Bath: {row.get('bathrooms', 'N/A')}")