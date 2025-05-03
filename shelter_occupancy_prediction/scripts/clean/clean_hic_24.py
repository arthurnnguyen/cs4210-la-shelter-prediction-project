import os
import pandas as pd

# === Set up paths ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

RAW_HIC_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'hic', '2024-housing-inventory-count.xlsx')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUT_FILE = os.path.join(PROCESSED_DIR, 'clean_hic_2024.csv')
os.makedirs(PROCESSED_DIR, exist_ok=True)

sheet_name = '2024 HIC - Shelter Projects'


# ---Load the sheet ---
df = pd.read_excel(
    RAW_HIC_PATH,
    sheet_name=sheet_name,
    usecols=[
        'Year', 'Proj. Type', 'Organization Name', 'Project Name',
        'City', 'SPA', 'Total Beds', 'Utilization Rate'
    ]
)

# --- Rename columns for consistency ---
df = df.rename(columns={
    'Proj. Type': 'Shelter Type',
    'Organization Name': 'Organization',
    'Project Name': 'Project Name',
    'Total Beds': 'Capacity'
})

# --- Create a synthetic Shelter ID from project name + city ---
df['Shelter ID'] = df['Project Name'].str.lower().str.strip() + "_" + df['City'].str.lower().str.strip()

# --- Drop rows with missing key data ---
df = df.dropna(subset=['Shelter ID', 'Capacity'])

# --- drop duplicates based on synthetic Shelter ID ---
df = df.drop_duplicates(subset='Shelter ID')

# --- Reorder columns ---
df = df[['Shelter ID', 'Year', 'Shelter Type', 'Organization', 'Project Name',
         'City', 'SPA', 'Capacity', 'Utilization Rate']]

# --- Save cleaned file ---
df.to_csv(OUTPUT_FILE, index=False)
print(f"Cleaned 2024 HIC saved to: {OUTPUT_FILE}")
