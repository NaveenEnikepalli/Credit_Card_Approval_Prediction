"""
Model trainer utility module for Credit Card Approval Prediction.
Handles class imbalance using weighting, calculates evaluation metrics (including Balanced Accuracy and ROC-AUC),
plots metrics, generates confusion matrices, compares models, and serializes the best classifier.
"""

import os
import json
import datetime
import joblib
import matplotlib
# Force Agg backend for headless environments to prevent GUI popups hanging executions
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report
)

def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model on the training set using class_weight='balanced'.
    """
    print("[MODEL TRAINING] Training Logistic Regression (class_weight='balanced')...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    """
    Trains a Decision Tree Classifier on the training set using class_weight='balanced'.
    """
    print("[MODEL TRAINING] Training Decision Tree Classifier (class_weight='balanced')...")
    model = DecisionTreeClassifier(max_depth=10, min_samples_split=20, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest Classifier on the training set using class_weight='balanced'.
    """
    print("[MODEL TRAINING] Training Random Forest Classifier (class_weight='balanced')...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, scale_pos_weight):
    """
    Trains an XGBoost Classifier on the training set using the computed scale_pos_weight
    and monotonic constraints to ensure logical risk predictions.
    """
    print(f"[MODEL TRAINING] Training XGBoost Classifier (scale_pos_weight={scale_pos_weight:.4f}) with monotonic constraints...")
    
    constraints = {
        'CNT_CHILDREN': 1,
        'CNT_FAM_MEMBERS': 1,
        'REGION_RATING_CLIENT': 1,
        'AMT_REQ_CREDIT_BUREAU_YEAR': 1,
        'ANNUITY_TO_INCOME_RATIO': 1,
        'AGE_YEARS': -1,
        'EMPLOYMENT_YEARS': -1,
        'AMT_INCOME_TOTAL': -1,
        'AMT_CREDIT': -1,
        'INCOME_TO_CREDIT_RATIO': -1,
        'CREDIT_TO_ANNUITY_RATIO': -1,
        'INCOME_PER_FAMILY_MEMBER': -1
    }
    
    # Map features to constraints corresponding to the column order of X_train
    mono_list = []
    for col in X_train.columns:
        mono_list.append(constraints.get(col, 0))
    mono_str = "(" + ",".join(map(str, mono_list)) + ")"
    
    model = XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        scale_pos_weight=scale_pos_weight, 
        random_state=42, 
        eval_metric='logloss', 
        n_jobs=-1,
        monotone_constraints=mono_str
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name, output_dir):
    """
    Evaluates a model's performance on the test set.
    Calculates standard and imbalanced metrics, prints report, and saves confusion matrix plot.
    
    Args:
        model: Trained scikit-learn or xgboost model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        model_name (str): The display name of the model.
        output_dir (str): Directory where plots should be saved.
        
    Returns:
        dict: Evaluation metrics scores.
    """
    print(f"\n" + "-"*30 + f" EVALUATING: {model_name} " + "-"*30)
    
    # 1. Predict class and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 2. Compute metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    # 3. Print evaluations to console
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Precision:         {prec:.4f}")
    print(f"Recall:            {rec:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print(f"ROC-AUC Score:     {roc_auc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    
    # 4. Save Confusion Matrix plot
    name_map = {
        "Logistic Regression": "Logistic_ConfusionMatrix.png",
        "Decision Tree": "DecisionTree_ConfusionMatrix.png",
        "Random Forest": "RandomForest_ConfusionMatrix.png",
        "XGBoost": "XGBoost_ConfusionMatrix.png"
    }
    filename = name_map.get(model_name, f"{model_name.replace(' ', '')}_ConfusionMatrix.png")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, filename)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Approved (0)', 'Rejected (1)'],
                yticklabels=['Approved (0)', 'Rejected (1)'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('True Label', fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix for {model_name} to: {plot_path}")
    
    return {
        'model_name': model_name,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

def compare_models(results, output_dir):
    """
    Compares metrics across all trained models.
    Prints a comparison table, plots comparison charts, and selects the best model.
    The best model is selected based on:
    1. Highest F1 Score
    2. Highest Balanced Accuracy
    3. Highest ROC-AUC Score
    
    Args:
        results (list): List of dictionaries containing evaluation metrics.
        output_dir (str): Directory where visual comparison charts should be saved.
        
    Returns:
        tuple: Best model display name, its metrics dictionary, and detailed reasoning text.
    """
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*90)
    print("                           MODEL COMPARISON SUMMARY (BALANCED)                         ")
    print("="*90)
    # Format scores for display table
    df_display = df_results.copy()
    for col in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        df_display[col] = df_display[col].map('{:.4f}'.format)
    
    print(df_display.to_string(index=False))
    print("="*90)
    
    # Sort results to find best model based on F1 Score -> Balanced Accuracy -> ROC-AUC
    # We sort ascending=False by sorting negated values or by using sorted() with custom key
    sorted_results = sorted(
        results, 
        key=lambda x: (x['f1_score'], x['balanced_accuracy'], x['roc_auc']), 
        reverse=True
    )
    
    best_row = sorted_results[0]
    best_model_name = best_row['model_name']
    
    # Generate detailed selection explanation
    reasoning = (
        f"Selected best model: {best_model_name}.\n"
        f"Reason: Out of all evaluated models, it achieved the highest F1 Score ({best_row['f1_score']:.4f}). "
        f"Furthermore, it demonstrated a Balanced Accuracy of {best_row['balanced_accuracy']:.4f} "
        f"and an ROC-AUC score of {best_row['roc_auc']:.4f}, demonstrating balanced class prediction performance."
    )
    
    print(f"\n[MODEL SELECTION] {reasoning}")
    
    # Save the 6 comparison plots
    metrics_to_plot = {
        'accuracy': 'Accuracy_Comparison.png',
        'balanced_accuracy': 'Balanced_Accuracy_Comparison.png',
        'precision': 'Precision_Comparison.png',
        'recall': 'Recall_Comparison.png',
        'f1_score': 'F1_Comparison.png',
        'roc_auc': 'ROC_AUC_Comparison.png'
    }
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    for metric, filename in metrics_to_plot.items():
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(
            data=df_results, 
            x='model_name', 
            y=metric, 
            hue='model_name',
            palette='Blues_d',
            legend=False
        )
        
        # Add labels on top of the bars
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.4f}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom',
                            xytext=(0, 5),
                            textcoords='offset points',
                            fontsize=10,
                            fontweight='bold')
                            
        plt.title(f'Model Comparison - {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Model Name', fontsize=11)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=11)
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved comparison plot to: {plot_path}")
        
    return best_model_name, best_row, reasoning

def save_best_model(best_model, best_model_name, metrics, models_dir):
    """
    Saves the best-performing model using Joblib and exports execution metadata to JSON.
    
    Args:
        best_model: The trained scikit-learn or xgboost model object.
        best_model_name (str): Name of the best model.
        metrics (dict): Dict of metric values of the best model.
        models_dir (str): Directory where models should be saved.
    """
    model_save_path = os.path.join(models_dir, "best_model.pkl")
    metadata_save_path = os.path.join(models_dir, "model_info.json")
    
    # 1. Save model pickle (overwrites previous)
    joblib.dump(best_model, model_save_path)
    print(f"[SERIALIZATION] Overwritten best model ({best_model_name}) at: {model_save_path}")
    
    # 2. Build metadata dictionary (overwrites previous)
    metadata = {
        "best_model": best_model_name,
        "accuracy": float(metrics['accuracy']),
        "balanced_accuracy": float(metrics['balanced_accuracy']),
        "precision": float(metrics['precision']),
        "recall": float(metrics['recall']),
        "f1_score": float(metrics['f1_score']),
        "roc_auc": float(metrics['roc_auc']),
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": "CreditCardApproval_Selected_Features.csv"
    }
    
    # 3. Save JSON metadata
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"[SERIALIZATION] Overwritten training metadata at: {metadata_save_path}")
