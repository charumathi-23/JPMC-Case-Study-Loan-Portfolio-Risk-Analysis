"""
Data Cleaning Script for CTC Risk Innovation Loan Portfolio Analysis

This script performs essential data cleaning and preprocessing on the raw loan portfolio dataset.
It handles missing values, standardizes formats, and ensures data quality for risk analysis.

Key cleaning steps:
1. Data loading and initial cleanup
2. Standardization of column formats
3. Missing value handling
4. Data type conversions
5. Quality assurance checks
"""

import pandas as pd
import numpy as np

def load_and_clean_data():
    """
    Load and clean the loan portfolio dataset.
    
    Returns:
        pd.DataFrame: Cleaned dataset ready for analysis
        dict: Report of cleaning operations performed
    """
    # Load raw data with robust parsing settings
    print("Loading raw dataset...")
    df = pd.read_csv("CTC_Risk_Innovation_Loans_Dataset.csv", sep=',', engine='python')
    
    # Standardize column names by removing whitespace
    df.columns = [c.strip() for c in df.columns]
    initial_rows = len(df)
    print(f"Initial dataset size: {initial_rows} rows")

    # Clean string columns by removing whitespace
    # This is crucial for consistent string matching and grouping
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        df[col] = df[col].str.strip()
    
    # Remove duplicate entries to prevent double-counting in risk calculations
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    print(f"Removed {duplicates_removed} duplicate rows")

    # Convert numeric columns to appropriate data types
    # These columns are critical for risk calculations
    num_cols = ["Total_Loan_Amount", "Drawn_Amount", "Time_to_Maturity_Years"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values in critical columns
    # Missing values in these fields could significantly impact risk calculations
    critical_cols = num_cols + ["Client_Internal_Rating", "Client_Industry"]
    cleanup_report = {}
    
    for col in critical_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            cleanup_report[col] = missing
            df = df[df[col].notnull()]  # Remove rows with missing critical data
            
    # Data quality checks
    print("\nData Quality Report:")
    print("-" * 50)
    print(f"Final dataset size: {len(df)} rows")
    if cleanup_report:
        print("\nMissing values removed:")
        for col, count in cleanup_report.items():
            print(f"- {col}: {count} rows")
    
    # Save cleaned dataset
    output_file = "CTC_Risk_Innovation_Loans_Dataset_cleaned.csv"
    df.to_csv(output_file, index=False)
    print(f"\nCleaned dataset saved to: {output_file}")
    
    return df, cleanup_report

if __name__ == "__main__":
    df, cleanup_report = load_and_clean_data()
