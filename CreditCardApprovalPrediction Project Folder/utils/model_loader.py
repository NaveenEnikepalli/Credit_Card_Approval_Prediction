"""
Model loader utility module for Credit Card Approval Prediction.
Handles loading machine learning model files, checking and generating feature_columns.pkl,
and executing compatibility audits at application startup.
"""

import os
import json
import joblib
import pandas as pd

def load_feature_columns(models_dir="models", dataset_path=None):
    """
    Loads feature_columns.pkl. If it does not exist, automatically regenerates it 
    from the columns of the training dataset (excluding 'TARGET') and saves it.
    """
    path = os.path.join(models_dir, "feature_columns.pkl")
    if not os.path.exists(path):
        print(f"[MODEL LOADER] '{path}' not found. Regenerating from dataset...")
        
        # Determine dataset path
        if dataset_path is None:
            dataset_path = os.path.join("dataset", "CreditCardApproval_Selected_Features.csv")
            
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Cannot regenerate feature_columns.pkl: dataset not found at {dataset_path}"
            )
            
        # Read column names (except TARGET)
        df_temp = pd.read_csv(dataset_path, nrows=5)
        columns = [col for col in df_temp.columns if col != 'TARGET']
        
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(columns, path)
        print(f"[MODEL LOADER] Created and saved feature columns to: {path}")
        print(f"  - Features: {columns}")
        
    return joblib.load(path)

def load_model(models_dir="models"):
    """Loads the trained machine learning model from best_model.pkl."""
    path = os.path.join(models_dir, "best_model.pkl")
    return joblib.load(path)

def load_scaler(models_dir="models"):
    """Loads the fitted standard scaler from scaler.pkl."""
    path = os.path.join(models_dir, "scaler.pkl")
    return joblib.load(path)

def load_encoders(models_dir="models"):
    """Loads the fitted label encoders dictionary from encoder.pkl."""
    path = os.path.join(models_dir, "encoder.pkl")
    return joblib.load(path)

def load_model_info(models_dir="models"):
    """Loads model information metadata from model_info.json."""
    path = os.path.join(models_dir, "model_info.json")
    with open(path, 'r') as f:
        return json.load(f)

class ModelRegistry:
    """
    Singleton registry that loads all ML files exactly once at Flask startup.
    Avoids expensive reloads on every prediction request.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self, models_dir="models"):
        if self.initialized:
            return
            
        print("[MODEL LOADER] Initializing Model Registry at startup...")
        
        # Load all components
        self.feature_columns = load_feature_columns(models_dir)
        self.model = load_model(models_dir)
        self.scaler = load_scaler(models_dir)
        self.encoders = load_encoders(models_dir)
        self.model_info = load_model_info(models_dir)
        
        # Verify compatibility
        self.verify_model_compatibility()
        
        self.initialized = True
        print("[MODEL LOADER] Model Registry initialized successfully.")

    def verify_model_compatibility(self):
        """
        Runs compatibility audits between model features and loaded metadata configurations.
        """
        print("[MODEL LOADER] Auditing model compatibility...")
        
        # 1. Verify object types
        assert self.model is not None, "Model is not loaded."
        assert self.scaler is not None, "Scaler is not loaded."
        assert self.encoders is not None, "LabelEncoders are not loaded."
        assert self.feature_columns is not None, "Feature columns list is not loaded."
        assert self.model_info is not None, "Model metadata JSON is not loaded."
        
        # 2. Check feature sizes
        expected_features_count = len(self.feature_columns)
        
        # For sklearn models and XGBoost, retrieve input features shape attribute
        if hasattr(self.model, "n_features_in_"):
            model_features_count = self.model.n_features_in_
            if model_features_count != expected_features_count:
                raise ValueError(
                    f"Compatibility check failed: Model expects {model_features_count} inputs, "
                    f"but feature_columns.pkl lists {expected_features_count} features."
                )
                
        # 3. Check Scaler size
        if hasattr(self.scaler, "n_features_in_"):
            scaler_features_count = self.scaler.n_features_in_
            # Scaler is fitted on numerical features only. Let's count numerical columns
            # Categorical columns: CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY, NAME_INCOME_TYPE,
            # NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, ORGANIZATION_TYPE. (Total 8)
            # Total columns (17) - Categorical columns (8) = Numerical columns (9).
            numerical_cols_count = len([
                col for col in self.feature_columns 
                if col not in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
                               'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
                               'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'ORGANIZATION_TYPE']
            ])
            if scaler_features_count != numerical_cols_count:
                raise ValueError(
                    f"Compatibility check failed: Scaler expects {scaler_features_count} inputs, "
                    f"but we mapped {numerical_cols_count} numerical features."
                )
                
        print(f"  - Model loaded: '{self.model_info.get('best_model', 'Unknown')}'")
        print(f"  - Total feature columns verified: {expected_features_count}")
        print("  - All model compatibility checks passed successfully.")
