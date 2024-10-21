import pandas as pd
import numpy as np

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
    print(f"Train STD: {train_std} | Train Mean: {train_mean}")
    # Reverse normalization formula
    predictions_df['Auc Price'] = (predictions_df['Auc Price'] * train_std) + train_mean
    return predictions_df