import os
import pandas as pd

def clean_evictions(file_path):
    # Load the 'Raw Data' sheet
    df = pd.read_excel(file_path, sheet_name='Raw Data')

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Parse notice date
    df['notice date'] = pd.to_datetime(df['notice date'], errors='coerce')

    # Extract year from notice date
    df['year'] = df['notice date'].dt.year

    # Convert rent owed to numeric
    df['rent owed'] = pd.to_numeric(df['rent owed'], errors='coerce')

    # Group by ZIP and Year
    grouped = df.groupby(['zip', 'year']).agg(
        eviction_count=('notice date', 'count'),
        avg_rent_owed=('rent owed', 'mean')
    ).reset_index()

    return grouped

# === File paths and output ===
if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'socioeconomic')
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    FILE_NAME = 'Evictions_Data_2.2023_to_11.2024 (11.27.24).xlsx'
    file_path = os.path.join(RAW_DIR, FILE_NAME)

    print(f"Cleaning file: {FILE_NAME}")
    cleaned = clean_evictions(file_path)

    output_path = os.path.join(PROCESSED_DIR, 'evictions_by_zip_year.csv')
    cleaned.to_csv(output_path, index=False)
    print(f"Cleaned eviction data saved to: {output_path}")
