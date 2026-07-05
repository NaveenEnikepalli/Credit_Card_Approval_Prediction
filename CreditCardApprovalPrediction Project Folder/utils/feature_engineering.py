"""
Feature engineering utility module for Credit Card Approval Prediction.
Contains functions for transforming raw features, converting units, and formatting columns.
"""

import pandas as pd

def perform_feature_engineering(df):
    """
    Executes feature engineering on the input dataframe:
    - Converts 'DAYS_BIRTH' to 'AGE_YEARS' (if DAYS_BIRTH is present).
    - Converts 'DAYS_EMPLOYED' to 'EMPLOYMENT_YEARS' (if DAYS_EMPLOYED is present).
    
    Args:
        df (pd.DataFrame): The input dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with engineered features.
    """
    df_engineered = df.copy()
    
    # Convert DAYS_BIRTH to AGE_YEARS if present
    if 'DAYS_BIRTH' in df_engineered.columns:
        print("[FEATURE ENGINEERING] Converting 'DAYS_BIRTH' to 'AGE_YEARS'...")
        # DAYS_BIRTH is typically negative (days back from current date)
        df_engineered['AGE_YEARS'] = df_engineered['DAYS_BIRTH'].abs() / 365.0
        df_engineered.drop(columns=['DAYS_BIRTH'], inplace=True)
        print("  - Conversion complete. 'DAYS_BIRTH' dropped.")
        
    # Convert DAYS_EMPLOYED to EMPLOYMENT_YEARS if present
    if 'DAYS_EMPLOYED' in df_engineered.columns:
        print("[FEATURE ENGINEERING] Converting 'DAYS_EMPLOYED' to 'EMPLOYMENT_YEARS'...")
        # DAYS_EMPLOYED is typically negative. Anomalous positive value (like 365243) represents retired/unemployed.
        df_engineered['EMPLOYMENT_YEARS'] = df_engineered['DAYS_EMPLOYED'].apply(
            lambda x: 0.0 if x > 0 else abs(x) / 365.0
        )
        df_engineered.drop(columns=['DAYS_EMPLOYED'], inplace=True)
        print("  - Conversion complete. 'DAYS_EMPLOYED' dropped.")
        
    # Standard check: confirm presence of age and employment features
    if 'AGE_YEARS' in df_engineered.columns:
        print(f"[FEATURE ENGINEERING] 'AGE_YEARS' verified. Range: {df_engineered['AGE_YEARS'].min():.1f} to {df_engineered['AGE_YEARS'].max():.1f} years.")
    if 'EMPLOYMENT_YEARS' in df_engineered.columns:
        print(f"[FEATURE ENGINEERING] 'EMPLOYMENT_YEARS' verified. Range: {df_engineered['EMPLOYMENT_YEARS'].min():.1f} to {df_engineered['EMPLOYMENT_YEARS'].max():.1f} years.")
        
    return df_engineered
