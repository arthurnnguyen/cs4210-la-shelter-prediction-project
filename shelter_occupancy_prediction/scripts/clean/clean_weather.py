import os
import pandas as pd


def clean_weather(file_path):
    df = pd.read_csv(file_path)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Parse date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Extract year
    df['year'] = df['date'].dt.year

    # Convert key columns to numeric
    df['precip (in)'] = pd.to_numeric(df['precip (in)'], errors='coerce')
    df['avg air temp (f)'] = pd.to_numeric(df['avg air temp (f)'], errors='coerce')

    # Group by year: get total rainfall and average temperature
    result = df.groupby('year').agg(
        total_rainfall_in=('precip (in)', 'sum'),
        avg_temp_f=('avg air temp (f)', 'mean')
    ).reset_index()

    return result


# === Paths ===
if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'weather')
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    FILE_NAME = 'jan21_dec23.csv'
    file_path = os.path.join(RAW_DIR, FILE_NAME)

    print(f"Processing weather data: {FILE_NAME}")
    cleaned = clean_weather(file_path)

    output_path = os.path.join(PROCESSED_DIR, 'weather_by_year.csv')
    cleaned.to_csv(output_path, index=False)
    print(f"Cleaned weather data saved to: {output_path}")
