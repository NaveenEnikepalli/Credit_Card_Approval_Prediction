"""
Helper visualization utility module for Credit Card Approval Prediction.
Contains functions for plotting and saving univariate and multivariate analysis plots.
"""

import os
import matplotlib
# Force Agg backend for headless environments to prevent GUI popups hanging executions
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def generate_univariate_plots(df, output_dir):
    """
    Generates count plots and histograms for key features and target, saving them to output_dir.
    
    Args:
        df (pd.DataFrame): Dataframe to analyze.
        output_dir (str): Directory where plot images should be saved.
    """
    print(f"[VISUALIZATION] Generating univariate plots into: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set global plotting style
    sns.set_theme(style="whitegrid")
    
    # 1. Target Count Plot (Imbalance check)
    if 'TARGET' in df.columns:
        plt.figure(figsize=(6, 5))
        sns.countplot(data=df, x='TARGET', palette='Set2')
        plt.title('Distribution of Target (Approval Status)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Target Class (0: Approved/Low Risk, 1: Rejected/High Risk)', fontsize=11)
        plt.ylabel('Count', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'univariate_target_countplot.png'), dpi=150)
        plt.close()
        print("  - Saved 'univariate_target_countplot.png'")
        
    # 2. Count plots for key categorical columns
    categorical_to_plot = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE']
    for col in categorical_to_plot:
        if col in df.columns:
            plt.figure(figsize=(8, 5))
            # Sort categories by frequency
            order = df[col].value_counts().index
            sns.countplot(data=df, x=col, order=order, palette='Blues_r')
            plt.title(f'Count Plot for {col}', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel(col, fontsize=11)
            plt.ylabel('Count', fontsize=11)
            plt.xticks(rotation=45 if df[col].nunique() > 3 else 0, ha='right' if df[col].nunique() > 3 else 'center')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'univariate_{col.lower()}_countplot.png'), dpi=150)
            plt.close()
            print(f"  - Saved 'univariate_{col.lower()}_countplot.png'")

    # 3. Histograms for key numerical columns
    numerical_to_plot = ['AGE_YEARS', 'EMPLOYMENT_YEARS', 'AMT_INCOME_TOTAL', 'AMT_CREDIT']
    for col in numerical_to_plot:
        if col in df.columns:
            plt.figure(figsize=(8, 5))
            # Log transform scale for highly skewed continuous values (like income)
            if col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT']:
                # Drop null values for safety, use log scale
                sns.histplot(data=df[col].dropna(), kde=True, bins=30, color='dodgerblue', log_scale=True)
                plt.title(f'Distribution of {col} (Log Scale)', fontsize=14, fontweight='bold', pad=15)
            else:
                sns.histplot(data=df[col].dropna(), kde=True, bins=30, color='dodgerblue')
                plt.title(f'Distribution of {col}', fontsize=14, fontweight='bold', pad=15)
                
            plt.xlabel(col, fontsize=11)
            plt.ylabel('Density / Count', fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'univariate_{col.lower()}_histogram.png'), dpi=150)
            plt.close()
            print(f"  - Saved 'univariate_{col.lower()}_histogram.png'")

def generate_multivariate_plots(df, output_dir):
    """
    Generates a correlation heatmap of all numerical columns and saves it to output_dir.
    
    Args:
        df (pd.DataFrame): Dataframe to analyze.
        output_dir (str): Directory where plot images should be saved.
    """
    print(f"[VISUALIZATION] Generating multivariate plots into: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract numerical columns
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    
    # Calculate Correlation matrix
    corr = numerical_df.corr()
    
    # Heatmap Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr, 
        annot=True, 
        fmt='.2f', 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        center=0,
        linewidths=0.5, 
        square=True, 
        cbar_kws={"shrink": .8}
    )
    plt.title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multivariate_correlation_heatmap.png'), dpi=150)
    plt.close()
    print("  - Saved 'multivariate_correlation_heatmap.png'")
