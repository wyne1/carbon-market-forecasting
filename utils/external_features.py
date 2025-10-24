"""
External Variables Feature Engineering
=======================================

This module handles feature engineering for external market variables
(Brent Oil, TTF Gas, EU Inflation) that influence carbon prices.
"""

import pandas as pd
import numpy as np


def engineer_external_features(df):
    """
    Engineer features from external variables
    Only called if external variables are present
    
    Args:
        df: DataFrame with potential external variables
        
    Returns:
        DataFrame with engineered external features
    """
    external_cols = ['Brent_Oil', 'TTF_Gas', 'EU_Inflation']
    
    # Check if external variables exist
    has_external = all(col in df.columns for col in external_cols)
    
    if not has_external:
        print("⚠️ External variables not found, skipping external feature engineering")
        return df
    
    print("🔧 Engineering external variable features...")
    
    # 1. Price change features
    df['Brent_Oil_pct_change'] = df['Brent_Oil'].pct_change()
    df['TTF_Gas_pct_change'] = df['TTF_Gas'].pct_change()
    df['EU_Inflation_change'] = df['EU_Inflation'].diff()
    
    # 2. Moving averages (7-day and 30-day)
    for col in ['Brent_Oil', 'TTF_Gas']:
        df[f'{col}_MA7'] = df[col].rolling(window=7).mean()
        df[f'{col}_MA30'] = df[col].rolling(window=30).mean()
        df[f'{col}_EMA7'] = df[col].ewm(span=7, adjust=False).mean()
        df[f'{col}_EMA30'] = df[col].ewm(span=30, adjust=False).mean()
    
    # 3. Volatility (rolling std)
    df['Brent_Oil_volatility'] = df['Brent_Oil'].rolling(window=7).std()
    df['TTF_Gas_volatility'] = df['TTF_Gas'].rolling(window=7).std()
    
    # 4. Interaction with carbon price (if Auc Price exists)
    if 'Auc Price' in df.columns:
        df['Auc_Brent_Ratio'] = df['Auc Price'] / df['Brent_Oil']
        df['Auc_Gas_Ratio'] = df['Auc Price'] / df['TTF_Gas']
        df['Auc_Brent_Interaction'] = df['Auc Price'] * df['Brent_Oil']
        df['Auc_Gas_Interaction'] = df['Auc Price'] * df['TTF_Gas']
    
    # 5. Energy complex correlation (Brent vs Gas)
    df['Brent_Gas_Ratio'] = df['Brent_Oil'] / df['TTF_Gas']
    df['Brent_Gas_Spread'] = df['Brent_Oil'] - df['TTF_Gas']
    
    # 6. Lagged external variables (1 and 7 days)
    for col in external_cols:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag7'] = df[col].shift(7)
    
    # 7. Cross-correlation features (momentum alignment)
    df['Brent_TTF_momentum_correlation'] = (
        df['Brent_Oil_pct_change'] * df['TTF_Gas_pct_change']
    )
    
    # 8. Inflation-adjusted features
    # Convert inflation rate to multiplier (e.g., 2% = 1.02)
    inflation_multiplier = 1 + (df['EU_Inflation'] / 100)
    if 'Auc Price' in df.columns:
        df['Auc_Price_Real'] = df['Auc Price'] / inflation_multiplier
    df['Brent_Oil_Real'] = df['Brent_Oil'] / inflation_multiplier
    df['TTF_Gas_Real'] = df['TTF_Gas'] / inflation_multiplier
    
    # Fill NaN values created by feature engineering
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    # Count engineered features
    new_features = [c for c in df.columns if any(x in c for x in external_cols) and c not in external_cols]
    print(f"✅ Added {len(new_features)} external features")
    
    return df
