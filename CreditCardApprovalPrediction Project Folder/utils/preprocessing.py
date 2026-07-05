"""
Preprocessing utility module for Credit Card Approval Prediction.
Contains functions for loading data, handling duplicates, imputing missing values,
encoding categorical features, and scaling numerical features.
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    """
    Loads a dataset from the specified CSV file path.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    print(f"[PREPROCESSING] Loading dataset from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"[PREPROCESSING] Dataset loaded. Shape: {df.shape}")
    return df

def check_and_remove_duplicates(df):
    """
    Checks for duplicate rows and removes them if found.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe without duplicates.
    """
    duplicate_count = df.duplicated().sum()
    print(f"[PREPROCESSING] Number of duplicate records found: {duplicate_count}")
    
    if duplicate_count > 0:
        df_cleaned = df.drop_duplicates()
        print(f"[PREPROCESSING] Removed {duplicate_count} duplicate records. New Shape: {df_cleaned.shape}")
        return df_cleaned
    else:
        print("[PREPROCESSING] No duplicate records to remove.")
        return df.copy()

def check_and_handle_missing_values(df):
    """
    Prints missing values per column and handles them using appropriate imputation.
    - Numerical columns are filled with their median value.
    - Categorical columns are filled with their mode value.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        pd.DataFrame: Imputed dataframe.
    """
    print("[PREPROCESSING] Checking for missing values...")
    missing_info = df.isnull().sum()
    columns_with_missing = missing_info[missing_info > 0]
    
    if len(columns_with_missing) > 0:
        print("[PREPROCESSING] Missing values found:")
        for col, count in columns_with_missing.items():
            print(f"  - {col}: {count} missing values")
            
        df_imputed = df.copy()
        for col in df_imputed.columns:
            if df_imputed[col].isnull().sum() > 0:
                if df_imputed[col].dtype == 'object':
                    # Impute categorical columns with mode
                    mode_val = df_imputed[col].mode()[0]
                    df_imputed[col] = df_imputed[col].fillna(mode_val)
                    print(f"  - Imputed categorical column '{col}' with mode: '{mode_val}'")
                else:
                    # Impute numerical columns with median
                    median_val = df_imputed[col].median()
                    df_imputed[col] = df_imputed[col].fillna(median_val)
                    print(f"  - Imputed numerical column '{col}' with median: {median_val}")
        return df_imputed
    else:
        print("[PREPROCESSING] No missing values detected.")
        return df.copy()

def encode_categorical_features(df, target_col='TARGET'):
    """
    Encodes object type columns using LabelEncoder.
    Stores and returns fitted encoders in a dictionary.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        target_col (str): The name of the target column to exclude from feature encoding.
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical columns.
        dict: Dictionary mapping column names to fitted LabelEncoder instances.
    """
    df_encoded = df.copy()
    encoders = {}
    
    # Identify categorical columns excluding the target column if it's object type
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
        
    print(f"[PREPROCESSING] Encoding categorical features: {categorical_cols}")
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
        print(f"  - Encoded '{col}' with {len(le.classes_)} unique classes.")
        
    return df_encoded, encoders

def scale_numerical_features(X_train, X_test, numerical_cols):
    """
    Scales the specified numerical columns using StandardScaler fitted on X_train.
    
    Args:
        X_train (pd.DataFrame): Training features dataframe.
        X_test (pd.DataFrame): Testing features dataframe.
        numerical_cols (list): List of numerical column names to scale.
        
    Returns:
        pd.DataFrame: Scaled X_train dataframe copy.
        pd.DataFrame: Scaled X_test dataframe copy.
        StandardScaler: Fitted StandardScaler instance.
    """
    print(f"[PREPROCESSING] Scaling numerical columns: {numerical_cols}")
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    scaler = StandardScaler()
    
    # Fit and transform training set
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    # Transform test set
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print("[PREPROCESSING] Scaling completed successfully.")
    return X_train_scaled, X_test_scaled, scaler
