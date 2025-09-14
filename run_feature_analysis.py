"""
Feature Importance Analysis Script
==================================

Usage: python run_feature_analysis.py

This script will:
1. Load and preprocess the LSEG dataset.
2. Engineer all available features.
3. Train a RandomForestRegressor model on the data.
4. Calculate and rank the importance of each feature.
5. Save a plot of the top 25 features to the output directory.
6. Print the full ranked list of features to the console.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Temporarily add utils to path
sys.path.append(str(Path(__file__).parent / 'utils'))

try:
    from config import get_config
    from utils.lseg_data_loader import LSEGDataLoader
    from utils.data_processing import prepare_data_for_feature_analysis
except ImportError as e:
    print(f"❌ Error: Failed to import necessary modules. {e}")
    print("💡 Make sure you are in the project's root directory and all dependencies are installed.")
    sys.exit(1)

def run_feature_importance_analysis(config: dict):
    """
    Loads data, trains a model, and calculates feature importance.
    """
    print("1. Initializing LSEG Data Loader...")
    try:
        lseg_loader = LSEGDataLoader()
        # Load auction data which contains the core features
        merged_df = lseg_loader.load_auction_data()
        print("   ✅ LSEG data loaded and preprocessed successfully.")
    except Exception as e:
        print(f"   ❌ Failed to load LSEG data: {e}")
        return

    if merged_df.empty:
        print("   ❌ No data loaded, cannot proceed with analysis.")
        return

    print("2. Preparing data for feature analysis...")
    X, y = prepare_data_for_feature_analysis(merged_df)

    if X is None or y is None:
        print("   ❌ Data preparation failed. Aborting.")
        return
        
    print(f"   ✅ Data prepared. Shape of feature matrix: {X.shape}")

    # Ensure target variable is not in features
    if 'target' in X.columns:
        X = X.drop(columns=['target'])

    print("3. Training RandomForestRegressor to determine feature importance...")
    # Using a simpler model optimized for feature importance, not for forecasting accuracy
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    print("   ✅ Model training complete.")

    print("4. Calculating and ranking feature importances...")
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    print("   ✅ Feature importances calculated.")
    return feature_importance_df

def save_importance_plot(feature_importance_df: pd.DataFrame, config: dict):
    """
    Saves a bar plot of the top N most important features.
    """
    if feature_importance_df is None:
        return
        
    output_dir = Path(config.get('output_dir', 'output_plots'))
    output_dir.mkdir(exist_ok=True)
    
    top_n = 25
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(
        x='importance',
        y='feature',
        data=feature_importance_df.head(top_n),
        palette='viridis',
        ax=ax
    )
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=16, weight='bold')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature Name', fontsize=12)
    plt.tight_layout()
    
    plot_path = output_dir / 'feature_importance_analysis.png'
    plt.savefig(plot_path, dpi=300)
    print(f"   ✅ Feature importance plot saved to: {plot_path}")


def main():
    """
    Main execution function
    """
    print("🚀 Running Feature Importance Analysis")
    print("=" * 50)
    
    config = get_config()
    
    feature_importance_df = run_feature_importance_analysis(config)
    
    if feature_importance_df is not None:
        save_importance_plot(feature_importance_df, config)
        
        print("\nTop 25 Most Important Features:")
        print(feature_importance_df.head(25).to_string())
        
        print(f"\n🎊 SUCCESS! Feature analysis complete.")
    else:
        print("\n💥 FAILED! Could not complete feature analysis.")
        sys.exit(1)

if __name__ == "__main__":
    main()
