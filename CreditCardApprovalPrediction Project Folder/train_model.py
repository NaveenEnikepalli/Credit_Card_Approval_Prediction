"""
Main machine learning training and orchestration script.
Integrates data preprocessing pipeline (Phase 3) and model building, evaluation,
comparison, and serialization under class imbalance corrections (Phase 4.5).
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Import preprocessing utilities
from utils.preprocessing import (
    load_data,
    check_and_remove_duplicates,
    check_and_handle_missing_values,
    encode_categorical_features,
    scale_numerical_features
)
from utils.feature_engineering import perform_feature_engineering
from utils.helper import generate_univariate_plots, generate_multivariate_plots

# Import model training utilities
from utils.model_trainer import (
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
    train_xgboost,
    evaluate_model,
    compare_models,
    save_best_model
)

def main():
    # Define file paths
    dataset_path = os.path.join("dataset", "CreditCardApproval_Selected_Features.csv")
    plots_dir = os.path.join("static", "images", "plots")
    evaluation_plots_dir = os.path.join("static", "images")
    models_dir = "models"
    
    # Ensure models and static directories exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(evaluation_plots_dir, exist_ok=True)
    
    print("="*80)
    print("   CREDIT CARD APPROVAL PREDICTION - PREPROCESSING PIPELINE (PHASE 3)   ")
    print("="*80)
    
    # 1. Load dataset
    df = load_data(dataset_path)
    
    # 2. Display basic information
    print("\n" + "-"*40 + " DATA INSPECTION " + "-"*40)
    print(f"Shape of Dataset: {df.shape}")
    
    print("\n--- First 5 Rows (Head) ---")
    print(df.head())
    
    print("\n--- Dataset Info ---")
    df.info()
    
    print("\n--- Dataset Statistical Summary ---")
    print(df.describe(include='all'))
    
    # 3 & 4. Check and remove duplicates
    print("\n" + "-"*40 + " DUPLICATE ANALYSIS " + "-"*40)
    df = check_and_remove_duplicates(df)
    
    # 5 & 6. Check and handle missing values
    print("\n" + "-"*40 + " MISSING VALUES ANALYSIS " + "-"*40)
    df = check_and_handle_missing_values(df)
    
    # Target Class Distribution Analysis & Visualization
    approved_count = int((df['TARGET'] == 0).sum())
    rejected_count = int((df['TARGET'] == 1).sum())
    total_count = len(df)
    approved_pct = (approved_count / total_count) * 100
    rejected_pct = (rejected_count / total_count) * 100
    
    print("\n" + "-"*40 + " TARGET CLASS DISTRIBUTION " + "-"*40)
    print(f"Number of Approved Samples (0): {approved_count} ({approved_pct:.2f}%)")
    print(f"Number of Rejected Samples (1): {rejected_count} ({rejected_pct:.2f}%)")
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(6, 5))
    sns.barplot(x=['Approved (0)', 'Rejected (1)'], y=[approved_count, rejected_count], hue=['Approved (0)', 'Rejected (1)'], palette='Set2', legend=False)
    plt.title('Target Class Distribution', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Class (Status)', fontsize=10)
    plt.ylabel('Number of Samples', fontsize=10)
    
    # Annotate bar charts
    for i, count in enumerate([approved_count, rejected_count]):
        pct = [approved_pct, rejected_pct][i]
        plt.text(i, count + (total_count * 0.01), f"{count}\n({pct:.1f}%)", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    target_dist_path = os.path.join(evaluation_plots_dir, "Target_Distribution.png")
    plt.savefig(target_dist_path, dpi=150)
    plt.close()
    print(f"Saved target distribution bar chart to: {target_dist_path}")
    
    # 7 & 8. Generate Univariate & Multivariate Plots
    print("\n" + "-"*40 + " DATA VISUALIZATION " + "-"*40)
    generate_univariate_plots(df, plots_dir)
    generate_multivariate_plots(df, plots_dir)
    
    # 9. Feature Engineering (Convert age and employment days to years if present)
    print("\n" + "-"*40 + " FEATURE ENGINEERING " + "-"*40)
    df = perform_feature_engineering(df)
    
    # 10. Encode categorical columns using LabelEncoder
    print("\n" + "-"*40 + " CATEGORICAL ENCODING " + "-"*40)
    df_encoded, encoders = encode_categorical_features(df, target_col='TARGET')
    
    # 11. Separate features and target
    print("\n" + "-"*40 + " FEATURES & TARGET SEPARATION " + "-"*40)
    if 'TARGET' not in df_encoded.columns:
        raise ValueError("Target column 'TARGET' not found in dataset columns.")
        
    X = df_encoded.drop(columns=['TARGET'])
    y = df_encoded['TARGET']
    print(f"Features matrix shape X: {X.shape}")
    print(f"Target vector shape y: {y.shape}")
    
    # 12. Split dataset into training and testing sets
    print("\n" + "-"*40 + " TRAIN-TEST SPLIT " + "-"*40)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # 13. Scale numerical columns using StandardScaler
    print("\n" + "-"*40 + " FEATURE SCALING " + "-"*40)
    # Identify numerical columns in X_train
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_train_scaled, X_test_scaled, scaler = scale_numerical_features(
        X_train, X_test, numerical_cols
    )
    
    # 14. Save encoders, scaler, and feature columns using Joblib into models/
    encoder_save_path = os.path.join(models_dir, "encoder.pkl")
    scaler_save_path = os.path.join(models_dir, "scaler.pkl")
    feature_cols_save_path = os.path.join(models_dir, "feature_columns.pkl")
    
    print("\n" + "-"*40 + " SERIALIZATION " + "-"*40)
    joblib.dump(encoders, encoder_save_path)
    print(f"[SERIALIZATION] Saved LabelEncoders to: {encoder_save_path}")
    joblib.dump(scaler, scaler_save_path)
    print(f"[SERIALIZATION] Saved StandardScaler to: {scaler_save_path}")
    joblib.dump(X.columns.tolist(), feature_cols_save_path)
    print(f"[SERIALIZATION] Saved Feature Columns list to: {feature_cols_save_path}")
    
    # ===== START PHASE 4.5: MODEL RETRAINING WITH IMBALANCE HANDLED =====
    print("\n" + "="*80)
    print("           MODEL BUILDING, EVALUATION & COMPARISON PHASE (PHASE 4.5)          ")
    print("="*80)
    
    # Calculate scale_pos_weight dynamically based on training labels y_train
    num_negatives = sum(y_train == 0)
    num_positives = sum(y_train == 1)
    scale_pos_weight = num_negatives / num_positives
    
    print(f"[IMBALANCE HANDLING] Training label counts - Low Risk (0): {num_negatives}, High Risk (1): {num_positives}")
    print(f"[IMBALANCE HANDLING] Dynamic scale_pos_weight computed: {scale_pos_weight:.4f}")
    
    # 1. Train models independently
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    dt_model = train_decision_tree(X_train_scaled, y_train)
    rf_model = train_random_forest(X_train_scaled, y_train)
    xgb_model = train_xgboost(X_train_scaled, y_train, scale_pos_weight)
    
    # 2. Evaluate every model
    results = []
    results.append(evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression", evaluation_plots_dir))
    results.append(evaluate_model(dt_model, X_test_scaled, y_test, "Decision Tree", evaluation_plots_dir))
    results.append(evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest", evaluation_plots_dir))
    results.append(evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost", evaluation_plots_dir))
    
    # 3. Model comparison & selection
    best_model_name, best_metrics, explanation = compare_models(results, evaluation_plots_dir)
    
    # 4. Save the best performing model only
    model_objects = {
        "Logistic Regression": lr_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }
    best_model_obj = model_objects[best_model_name]
    save_best_model(best_model_obj, best_model_name, best_metrics, models_dir)
    
    # 5. Output exact requested completion blocks
    df_results = pd.DataFrame(results)
    df_display = df_results.copy()
    for col in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        df_display[col] = df_display[col].map('{:.4f}'.format)
        
    print("\n" + "="*80)
    print("                       PREPROCESSING & RETRAINING OUTPUT SUMMARY                ")
    print("="*80)
    print("Original class distribution:")
    print(f"  - Approved (0): {approved_count} ({approved_pct:.2f}%)")
    print(f"  - Rejected (1): {rejected_count} ({rejected_pct:.2f}%)")
    
    print("\nMethod used to handle imbalance:")
    print(f"  - Logistic Regression, Decision Tree, Random Forest: class_weight='balanced'")
    print(f"  - XGBoost: scale_pos_weight={scale_pos_weight:.4f} calculated dynamically from training set")
    
    print("\nUpdated evaluation metrics:")
    print(df_display.to_string(index=False))
    
    print(f"\nSelected best model:")
    print(f"  - Name: {best_model_name}")
    
    print(f"\nReason for selecting the best model:")
    print(f"  - {explanation}")
    
    print("\nTraining completed successfully")
    print("="*80)

if __name__ == '__main__':
    main()
