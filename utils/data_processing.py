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


