import os
import pandas as pd


def clean_unemployment_from_sheet(sheet_df, year):
    # First row is likely header text, so skip it
    data_columns = ['Month', 'Labor Force', 'Employment', 'Unemployment', 'Unemp. Rate']

    df = sheet_df.iloc[1:, 0:5].copy()
    df.columns = data_columns
    df['Year'] = year

    # Ensure Unemp. Rate is numeric
    df['Unemp. Rate'] = pd.to_numeric(df['Unemp. Rate'], errors='coerce')

    # Compute average unemployment rate for the year
    return df.groupby('Year', as_index=False)['Unemp. Rate'].mean().rename(
        columns={'Unemp. Rate': 'avg_unemployment_rate'})


if __name__ == "__main__":
    # --- File paths ---
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'socioeconomic')
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    FILE_NAME = 'unemployment.xlsx'
    file_path = os.path.join(RAW_DIR, FILE_NAME)

    # Load raw file (no headers)
    raw = pd.read_excel(file_path, header=None)

    # Split into 2 blocks — assume 13 rows per year (1 header + 12 months)
    rows_per_year = 13
    year_2023_block = raw.iloc[0:rows_per_year]
    year_2024_block = raw.iloc[rows_per_year:rows_per_year * 2]

    # Clean both years
    cleaned_2023 = clean_unemployment_from_sheet(year_2023_block, 2023)
    cleaned_2024 = clean_unemployment_from_sheet(year_2024_block, 2024)

    # Combine and save
    final = pd.concat([cleaned_2023, cleaned_2024], ignore_index=True)
    output_path = os.path.join(PROCESSED_DIR, 'unemployment_avg_by_year.csv')
    final.to_csv(output_path, index=False)

    print(f"Cleaned unemployment data saved to: {output_path}")
