"""
Transformer Utilities for Carbon Market Forecasting
====================================================

Helper functions and utilities for Transformer-based predictions
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Optional, List
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from utils.transformer_model import TransformerCarbonModel, create_transformer_model
from utils.model_utils import generate_predictions, generate_recent_predictions, check_gradient


def prepare_transformer_data(
    df: pd.DataFrame,
    input_steps: int = 7,
    output_steps: int = 7,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data specifically for Transformer model
    
    Args:
        df: Input DataFrame
        input_steps: Number of input time steps
        output_steps: Number of output time steps
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
    
    Returns:
        Tuple of train, val, test data and labels
    """
    # Normalize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Create sequences
    X, y = [], []
    for i in range(input_steps, len(scaled_data) - output_steps + 1):
        X.append(scaled_data[i - input_steps:i])
        y.append(scaled_data[i:i + output_steps])
    
    X, y = np.array(X), np.array(y)
    
    # Split data
    n_train = int(len(X) * train_ratio)
    n_val = int(len(X) * val_ratio)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def generate_transformer_predictions(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    input_width: int = 7,
    out_steps: int = 7
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Generate predictions using Transformer model
    
    Args:
        model: Trained Transformer model
        test_df: Test DataFrame
        input_width: Input window size
        out_steps: Output steps
    
    Returns:
        Tuple of (predictions_df, recent_predictions, trend)
    """
    # Generate standard predictions
    predictions_df = generate_predictions(model, test_df, input_width, out_steps)
    
    # Generate recent predictions
    recent_preds, trend = generate_recent_predictions(model, test_df, input_width, out_steps)
    
    return predictions_df, recent_preds, trend


def calculate_attention_weights(
    model: tf.keras.Model,
    input_data: np.ndarray
) -> np.ndarray:
    """
    Extract attention weights from Transformer model
    
    Args:
        model: Transformer model
        input_data: Input data
    
    Returns:
        Attention weights array
    """
    # Create a model that outputs attention weights
    attention_weights = []
    
    # Get intermediate outputs
    for layer in model.layers:
        if 'transformer_block' in layer.name:
            # Extract attention weights from transformer blocks
            intermediate_model = tf.keras.Model(
                inputs=model.input,
                outputs=layer.output
            )
            intermediate_output = intermediate_model.predict(input_data)
            attention_weights.append(intermediate_output)
    
    return np.array(attention_weights) if attention_weights else None


def visualize_transformer_attention(
    attention_weights: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize attention weights from Transformer
    
    Args:
        attention_weights: Attention weight matrix
        feature_names: List of feature names
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    if attention_weights is None or len(attention_weights) == 0:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Average attention across all heads and layers
    avg_attention = np.mean(attention_weights, axis=(0, 1))
    
    # Temporal attention pattern
    axes[0].imshow(avg_attention, cmap='Blues', aspect='auto')
    axes[0].set_title('Temporal Attention Pattern')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Time Steps')
    axes[0].colorbar()
    
    # Feature importance from attention
    feature_importance = np.mean(avg_attention, axis=0)
    axes[1].barh(range(len(feature_names[:10])), feature_importance[:10])
    axes[1].set_yticks(range(len(feature_names[:10])))
    axes[1].set_yticklabels(feature_names[:10])
    axes[1].set_xlabel('Attention Weight')
    axes[1].set_title('Top 10 Feature Importance from Attention')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_transformer_vs_baseline(
    transformer_preds: pd.DataFrame,
    baseline_preds: pd.DataFrame,
    actual_values: pd.DataFrame,
    metric: str = 'mae'
) -> Dict[str, float]:
    """
    Compare Transformer predictions with baseline model
    
    Args:
        transformer_preds: Transformer model predictions
        baseline_preds: Baseline model predictions
        actual_values: Actual values
        metric: Metric to use for comparison
    
    Returns:
        Dictionary with comparison metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    metrics = {}
    
    # Align indices
    common_idx = transformer_preds.index.intersection(
        baseline_preds.index
    ).intersection(actual_values.index)
    
    if len(common_idx) == 0:
        return metrics
    
    # Calculate metrics for Transformer
    if metric == 'mae':
        metrics['transformer_mae'] = mean_absolute_error(
            actual_values.loc[common_idx, 'Auc Price'],
            transformer_preds.loc[common_idx, 'Auc Price']
        )
        metrics['baseline_mae'] = mean_absolute_error(
            actual_values.loc[common_idx, 'Auc Price'],
            baseline_preds.loc[common_idx, 'Auc Price']
        )
    elif metric == 'rmse':
        metrics['transformer_rmse'] = np.sqrt(mean_squared_error(
            actual_values.loc[common_idx, 'Auc Price'],
            transformer_preds.loc[common_idx, 'Auc Price']
        ))
        metrics['baseline_rmse'] = np.sqrt(mean_squared_error(
            actual_values.loc[common_idx, 'Auc Price'],
            baseline_preds.loc[common_idx, 'Auc Price']
        ))
    
    # R2 score
    metrics['transformer_r2'] = r2_score(
        actual_values.loc[common_idx, 'Auc Price'],
        transformer_preds.loc[common_idx, 'Auc Price']
    )
    metrics['baseline_r2'] = r2_score(
        actual_values.loc[common_idx, 'Auc Price'],
        baseline_preds.loc[common_idx, 'Auc Price']
    )
    
    # Improvement percentage
    if metric == 'mae':
        metrics['improvement_pct'] = (
            (metrics['baseline_mae'] - metrics['transformer_mae']) / 
            metrics['baseline_mae'] * 100
        )
    
    return metrics


def plot_transformer_predictions(
    test_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    recent_preds: pd.DataFrame,
    title: str = "Transformer Model Predictions"
) -> plt.Figure:
    """
    Plot Transformer model predictions
    
    Args:
        test_df: Test data
        predictions_df: Historical predictions
        recent_preds: Recent/future predictions
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    
    # Plot 1: Historical predictions
    axes[0].plot(test_df.index, test_df['Auc Price'], 
                label='Actual', color='black', alpha=0.7, linewidth=1.5)
    axes[0].plot(predictions_df.index, predictions_df['Auc Price'], 
                label='Transformer Predictions', color='blue', alpha=0.7)
    
    # Add confidence bands (if available)
    if 'Auc Price_upper' in predictions_df.columns:
        axes[0].fill_between(
            predictions_df.index,
            predictions_df['Auc Price_lower'],
            predictions_df['Auc Price_upper'],
            alpha=0.2, color='blue', label='Confidence Band'
        )
    
    axes[0].set_title(f'{title} - Historical Performance')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price (€/tCO2)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Recent predictions with trend
    last_30_days = test_df.tail(30)
    axes[1].plot(last_30_days.index, last_30_days['Auc Price'], 
                'o-', label='Recent Actual', color='black', markersize=4)
    axes[1].plot(recent_preds.index, recent_preds['Auc Price'], 
                's-', label='Future Predictions', color='red', markersize=6)
    
    # Add trend arrow
    if len(recent_preds) > 1:
        trend_start = recent_preds['Auc Price'].iloc[0]
        trend_end = recent_preds['Auc Price'].iloc[-1]
        trend_color = 'green' if trend_end > trend_start else 'red'
        axes[1].annotate('', 
                        xy=(recent_preds.index[-1], trend_end),
                        xytext=(recent_preds.index[0], trend_start),
                        arrowprops=dict(arrowstyle='->', color=trend_color, lw=2))
    
    axes[1].set_title('Recent & Future Predictions')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Price (€/tCO2)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def generate_transformer_report(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    metrics: Dict[str, float],
    save_path: str = 'transformer_report.txt'
) -> str:
    """
    Generate a text report for Transformer model performance
    
    Args:
        model: Trained Transformer model
        test_df: Test data
        predictions_df: Model predictions
        metrics: Performance metrics
        save_path: Path to save report
    
    Returns:
        Report string
    """
    report = []
    report.append("=" * 60)
    report.append("TRANSFORMER MODEL PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Model Architecture
    report.append("MODEL ARCHITECTURE")
    report.append("-" * 40)
    report.append(f"Model Type: Transformer with Attention Mechanism")
    report.append(f"Total Parameters: {model.count_params():,}")
    report.append(f"Input Shape: {model.input_shape}")
    report.append(f"Output Shape: {model.output_shape}")
    report.append("")
    
    # Performance Metrics
    report.append("PERFORMANCE METRICS")
    report.append("-" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            report.append(f"{key}: {value:.4f}")
        else:
            report.append(f"{key}: {value}")
    report.append("")
    
    # Prediction Statistics
    report.append("PREDICTION STATISTICS")
    report.append("-" * 40)
    pred_stats = predictions_df['Auc Price'].describe()
    report.append(f"Mean Prediction: €{pred_stats['mean']:.2f}")
    report.append(f"Std Dev: €{pred_stats['std']:.2f}")
    report.append(f"Min: €{pred_stats['min']:.2f}")
    report.append(f"Max: €{pred_stats['max']:.2f}")
    report.append("")
    
    # Recent Performance
    report.append("RECENT PERFORMANCE (Last 30 days)")
    report.append("-" * 40)
    recent_actual = test_df['Auc Price'].tail(30)
    recent_pred = predictions_df['Auc Price'].tail(30)
    
    if len(recent_pred) > 0 and len(recent_actual) > 0:
        recent_mae = np.mean(np.abs(recent_actual.values - recent_pred.values))
        recent_mape = np.mean(np.abs((recent_actual.values - recent_pred.values) / recent_actual.values)) * 100
        report.append(f"Recent MAE: €{recent_mae:.2f}")
        report.append(f"Recent MAPE: {recent_mape:.2f}%")
    
    report.append("")
    report.append("=" * 60)
    
    # Join and save
    report_text = "\n".join(report)
    
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    return report_text


# Integration functions for existing pipeline
def create_transformer_for_pipeline(num_features: int, out_steps: int = 7) -> tf.keras.Model:
    """
    Create Transformer model that's compatible with existing pipeline
    
    This function wraps the Transformer model creation to match
    the interface expected by the existing codebase.
    """
    return create_transformer_model(num_features, out_steps)


def train_transformer_for_pipeline(
    model: tf.keras.Model,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessor,
    max_epochs: int = 50
):
    """
    Train Transformer model using existing pipeline interface
    
    This function wraps the Transformer training to match
    the interface expected by the existing codebase.
    """
    from utils.transformer_model import train_transformer_model
    return train_transformer_model(
        model, train_df, val_df, test_df, preprocessor, max_epochs
    )
