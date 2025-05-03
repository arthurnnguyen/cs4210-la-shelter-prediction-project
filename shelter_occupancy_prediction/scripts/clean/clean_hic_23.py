import os
import pandas as pd

# --- Setup project root and paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

RAW_HIC_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'hic', '2023-housing-inventory-count.xlsx')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(PROCESSED_DIR, 'clean_hic_2023.csv')

# --- Read specific sheet from Excel ---
sheet_name = '2023 HIC - Shelter Projects'

df = pd.read_excel(
    RAW_HIC_PATH,
    sheet_name=sheet_name,
    usecols=[
        'Row #', 'Year', 'Proj. Type', 'Organization Name', 'Project Name',
        'City', 'Zip', 'SPA', 'Total Beds', 'Utilization Rate'
    ]
)

# --- Rename columns for clarity ---
df = df.rename(columns={
    'Row #': 'Shelter ID',
    'Proj. Type': 'Shelter Type',
    'Organization Name': 'Organization',
    'Project Name': 'Project Name',
    'Zip': 'ZIP',
    'Total Beds': 'Capacity',
})

# --- Clean data ---
# Drop rows where shelter ID or capacity is missing
df = df.dropna(subset=['Shelter ID', 'Capacity'])

# remove any duplicates
df = df.drop_duplicates(subset='Shelter ID')

# --- Save to CSV ---
df.to_csv(OUTPUT_FILE, index=False)
print(f"Cleaned HIC data saved to: {OUTPUT_FILE}")
