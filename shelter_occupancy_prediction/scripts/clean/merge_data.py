import os
import pandas as pd


def normalize_columns(df):
    df.columns = df.columns.str.strip().str.upper()
    return df


def load_and_prepare_hic(hic_2023_path, hic_2024_path):
    hic_2023 = normalize_columns(pd.read_csv(hic_2023_path))
    hic_2024 = normalize_columns(pd.read_csv(hic_2024_path))

    hic_2023["YEAR"] = 2023
    hic_2024["YEAR"] = 2024

    hic = pd.concat([hic_2023, hic_2024], ignore_index=True)
    return hic


def merge_all(hic, pit, evictions, weather, unemployment):
    # Normalize column names
    hic = normalize_columns(hic)
    pit = normalize_columns(pit)
    evictions = normalize_columns(evictions)
    weather = normalize_columns(weather)
    unemployment = normalize_columns(unemployment)

    # Merge HIC + PIT by SPA and YEAR
    df = pd.merge(hic, pit, on=["SPA", "YEAR"], how="left")

    # Merge weather and unemployment by YEAR
    df = pd.merge(df, weather, on="YEAR", how="left")
    df = pd.merge(df, unemployment, on="YEAR", how="left")

    # Ensure ZIP codes are strings with leading zeros
    if "ZIP" in df.columns and "ZIP" in evictions.columns:
        df["ZIP"] = df["ZIP"].astype(str).str.zfill(5)
        evictions["ZIP"] = evictions["ZIP"].astype(str).str.zfill(5)
        df = pd.merge(df, evictions, on=["ZIP", "YEAR"], how="left")
    else:
        print("Skipping eviction merge — ZIP not found in both datasets.")

    # Clean up duplicated eviction columns (if present)
    if "EVICTION_COUNT_X" in df.columns and "EVICTION_COUNT_Y" in df.columns:
        df = df.drop(columns=["EVICTION_COUNT_X", "AVG_RENT_OWED_X"])
        df = df.rename(columns={
            "EVICTION_COUNT_Y": "EVICTION_COUNT",
            "AVG_RENT_OWED_Y": "AVG_RENT_OWED"
        })

    # Rename CAPACITY column if needed
    if "CAPACITY" in df.columns:
        df = df.rename(columns={"CAPACITY": "SHELTER_CAPACITY"})

    # Compute strain score
    df["STRAIN_SCORE"] = df["TOTAL_HOMELESS"] / df["SHELTER_CAPACITY"]

    return df


if __name__ == "__main__":
    # Match your folder structure
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".."))
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "shelter_occupancy_prediction", "data", "processed")
    OUTPUT_PATH = os.path.join(PROCESSED_DIR, "merged_data.csv")

    # File paths
    hic_2023 = os.path.join(PROCESSED_DIR, "clean_hic_2023.csv")
    hic_2024 = os.path.join(PROCESSED_DIR, "clean_hic_2024.csv")
    pit_file = os.path.join(PROCESSED_DIR, "homeless_by_spa.csv")
    eviction_file = os.path.join(PROCESSED_DIR, "evictions_by_zip_year.csv")
    weather_file = os.path.join(PROCESSED_DIR, "weather_by_year.csv")
    unemployment_file = os.path.join(PROCESSED_DIR, "unemployment_avg_by_year.csv")

    # Load data
    hic = load_and_prepare_hic(hic_2023, hic_2024)
    pit = pd.read_csv(pit_file)
    evictions = pd.read_csv(eviction_file)
    weather = pd.read_csv(weather_file)
    unemployment = pd.read_csv(unemployment_file)

    # Merge and export
    merged_df = merge_all(hic, pit, evictions, weather, unemployment)
    merged_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Merged dataset saved to: {OUTPUT_PATH}")
