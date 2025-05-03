import os
import pandas as pd

def clean_homeless_count(file_path):
    df = pd.read_excel(file_path, sheet_name='Counts')

    # Normalize column names
    df.columns = df.columns.str.lower()

    # Compute total homeless (adjust if needed)
    df['total_homeless'] = (
        df.get('totsheltpeople', 0) +
        df.get('totthpeople', 0) +
        df.get('totshpeople', 0)
    )

    # Group by SPA and Year
    return df.groupby(['spa', 'year'], as_index=False)['total_homeless'].sum()

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'hc')
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    files = ['hc23-data-by-census-subtract.xlsx', 'hc24-data-by-census-subtract.xlsx']
    cleaned_data = []

    for file in files:
        file_path = os.path.join(RAW_DIR, file)
        print(f"Cleaning {file}...")
        cleaned = clean_homeless_count(file_path)
        cleaned_data.append(cleaned)

    combined = pd.concat(cleaned_data, ignore_index=True)
    output_path = os.path.join(PROCESSED_DIR, 'homeless_by_spa.csv')
    combined.to_csv(output_path, index=False)
    print(f"Cleaned homeless count data saved to {output_path}")
