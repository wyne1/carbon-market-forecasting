import pandas as pd
import numpy as np
from utils.dataset import MarketData, DataPreprocessor
from pathlib import Path


def prepare_data(merged_df):
    FEATURES = merged_df.columns.tolist()
    LABEL_COLS = ['Auc Price']

    preprocessor = DataPreprocessor(features=FEATURES, label_columns=LABEL_COLS, input_width=7, label_width=7, shift=1)
    train_df, test_df, val_df = preprocessor.train_test_data(merged_df)
    train_df, test_df, val_df = preprocessor.normalize(train_df, test_df, val_df)
    return train_df, test_df, val_df, preprocessor

def reverse_normalize(predictions_df: pd.DataFrame, train_mean: float, train_std: float) -> pd.DataFrame:
    """
    Reverse normalize the 'Auc Price' in the predictions DataFrame.

    Parameters:
    - predictions_df: DataFrame containing the predictions with normalized 'Auc Price'
    - train_mean: Mean of the training data used for normalization
    - train_std: Standard deviation of the training data used for normalization

    Returns:
    - DataFrame with the 'Auc Price' reverse normalized
    """
    # Reverse normalization formula
    predictions_df['Auc Price'] = (predictions_df['Auc Price'] * train_std) + train_mean
    return predictions_df

import pandas as pd
import numpy as np
from utils.dataset import MarketData, DataPreprocessor
from utils.lseg_data_loader import LSEGDataLoader
from pathlib import Path

def prepare_data_from_lseg(lseg_loader: LSEGDataLoader, 
                          start_date: str = None, 
                          end_date: str = None):
    """
    Prepare data from LSEG Data Library
    
    Args:
        lseg_loader: Initialized LSEGDataLoader instance
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
    
    Returns:
        Tuple of (train_df, test_df, val_df, preprocessor)
    """
    # Load auction data from LSEG
    auction_df = lseg_loader.load_auction_data(start_date, end_date)
    
    # Load ICE data if available
    ice_df = lseg_loader.load_ice_data(start_date, end_date)
    
    # Merge with ICE data if available
    if not ice_df.empty:
        merged_df = pd.merge(auction_df, ice_df, on='Date', how='outer')
        merged_df = MarketData.fill_missing_values(merged_df)
    else:
        merged_df = auction_df
    
    # Ensure date range and resampling
    merged_df = merged_df.set_index('Date').resample('D').mean().reset_index()
    merged_df = merged_df[7:]  # Skip first week
    
    # Fill forward missing values
    auc_cols = ['Auc Price', 'Median Price', 'Cover Ratio', 'Spot Value',
                'Auction Spot Diff', 'Median Spot Diff', 'Premium/discount-settle']
    merged_df.loc[:, auc_cols] = merged_df[auc_cols].ffill()
    
    # Apply feature engineering
    merged_df = DataPreprocessor.engineer_auction_features(merged_df)
    
    # Prepare train/test/val splits
    FEATURES = merged_df.columns.tolist()
    LABEL_COLS = ['Auc Price']
    
    preprocessor = DataPreprocessor(
        features=FEATURES, 
        label_columns=LABEL_COLS, 
        input_width=.7, 
        label_width=7, 
        shift=1
    )
    
    train_df, test_df, val_df = preprocessor.train_test_data(merged_df)
    train_df, test_df, val_df = preprocessor.normalize(train_df, test_df, val_df)
    
    # Drop any columns with all NaN values
    train_df = train_df.dropna(axis=1)
    test_df = test_df.dropna(axis=1)
    val_df = val_df.dropna(axis=1)
    
    return train_df, test_df, val_df, preprocessor

def prepare_data_for_feature_analysis(df: pd.DataFrame):
    """
    Prepares data for feature importance analysis.

    Args:
        df: The input DataFrame with all features.

    Returns:
        A tuple of (X, y) where X is the feature matrix and y is the target vector.
    """
    df = df.copy()
    
    # Define the target variable: the next day's auction price
    df['target'] = df['Auc Price'].shift(-1)
    
    # Drop rows with NaN in the target (the last row)
    df = df.dropna(subset=['target'])
    
    # Define features (X) and target (y)
    y = df['target']
    
    # Drop non-feature columns
    X = df.drop(columns=['target', 'Auc Price', 'Date', 'Auction_Type', 'Day of Week'], errors='ignore')

    # Fill any remaining NaNs in features with the median of the column
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)

    # Ensure all data is numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1, how='all') # Drop columns that are all NaN after coercion
    
    # Re-check for NaNs after coercion and fill
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)

    if X.isnull().values.any():
        # For simplicity, drop rows with any NaNs left
        nan_rows = y[X.isnull().any(axis=1)].index
        X = X.dropna()
        y = y.drop(index=nan_rows)


    return X, y

