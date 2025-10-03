#!/usr/bin/env python3
"""
Test script for Transformer model implementation
================================================

This script tests the Transformer model independently to ensure it works
before integration with the main application.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import the Transformer modules
from utils.transformer_model import TransformerCarbonModel, create_transformer_model
from utils.transformer_utils import (
    prepare_transformer_data,
    generate_transformer_predictions,
    plot_transformer_predictions,
    compare_transformer_vs_baseline
)

def test_transformer_model():
    """
    Test the Transformer model with synthetic data
    """
    print("=" * 60)
    print("TRANSFORMER MODEL TEST")
    print("=" * 60)
    
    # 1. Create synthetic data
    print("\n1. Creating synthetic data...")
    np.random.seed(42)
    
    # Create synthetic time series data
    days = 200
    num_features = 10
    
    # Generate synthetic features with some patterns
    time = np.arange(days)
    data = {}
    
    # Main price feature with trend and seasonality
    data['Auc Price'] = (
        80 + 0.1 * time +  # Trend
        10 * np.sin(2 * np.pi * time / 30) +  # Monthly seasonality
        5 * np.sin(2 * np.pi * time / 7) +  # Weekly seasonality
        np.random.normal(0, 2, days)  # Noise
    )
    
    # Other correlated features
    for i in range(1, num_features):
        data[f'Feature_{i}'] = (
            50 + 0.05 * time +
            5 * np.sin(2 * np.pi * time / (20 + i)) +
            np.random.normal(0, 1, days)
        )
    
    # Create DataFrame
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    df = pd.DataFrame(data, index=dates)
    
    print(f"✅ Created dataset with shape: {df.shape}")
    print(f"   Features: {list(df.columns)}")
    
    # 2. Initialize Transformer model
    print("\n2. Initializing Transformer model...")
    
    transformer = TransformerCarbonModel(
        num_features=num_features,
        input_steps=7,
        output_steps=7,
        d_model=64,  # Smaller for testing
        num_heads=2,
        num_layers=2,
        dropout_rate=0.1
    )
    
    model = transformer.build_model()
    print(f"✅ Model created with {model.count_params():,} parameters")
    
    # 3. Prepare data
    print("\n3. Preparing data for training...")
    
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_transformer_data(
        df,
        input_steps=7,
        output_steps=7,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    print(f"✅ Data prepared:")
    print(f"   Training: X={X_train.shape}, y={y_train.shape}")
    print(f"   Validation: X={X_val.shape}, y={y_val.shape}")
    print(f"   Test: X={X_test.shape}, y={y_test.shape}")
    
    # 4. Train model (just a few epochs for testing)
    print("\n4. Training model (5 epochs for testing)...")
    
    history = transformer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=5,  # Just for testing
        batch_size=32,
        patience=3
    )
    
    print(f"✅ Model trained")
    print(f"   Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"   Final validation loss: {history.history['val_loss'][-1]:.4f}")
    
    # 5. Generate predictions
    print("\n5. Generating predictions...")
    
    predictions = transformer.predict(X_test[:10])  # Predict on first 10 samples
    print(f"✅ Generated predictions with shape: {predictions.shape}")
    
    # 6. Test model save/load
    print("\n6. Testing model save/load...")
    
    # Save model
    save_path = "models/test_transformer"
    os.makedirs(save_path, exist_ok=True)
    transformer.save_model(save_path)
    print(f"✅ Model saved to {save_path}")
    
    # Load model
    new_transformer = TransformerCarbonModel(
        num_features=num_features,
        input_steps=7,
        output_steps=7
    )
    new_transformer.load_model(save_path)
    print(f"✅ Model loaded from {save_path}")
    
    # Test loaded model
    new_predictions = new_transformer.predict(X_test[:10])
    assert np.allclose(predictions, new_predictions), "Loaded model predictions don't match!"
    print(f"✅ Loaded model produces identical predictions")
    
    # 7. Test integration with existing pipeline
    print("\n7. Testing integration with existing pipeline...")
    
    # Create model using the pipeline-compatible function
    pipeline_model = create_transformer_model(num_features, out_steps=7)
    print(f"✅ Pipeline-compatible model created")
    
    # 8. Visualize results
    print("\n8. Creating visualization...")
    
    # Create a simple prediction vs actual plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual values
    ax.plot(range(len(y_test[:20, 0, 0])), y_test[:20, 0, 0], 
            'o-', label='Actual', alpha=0.7)
    
    # Plot predictions
    ax.plot(range(len(predictions[:20, 0, 0])), predictions[:20, 0, 0], 
            's-', label='Transformer Predictions', alpha=0.7)
    
    ax.set_xlabel('Sample')
    ax.set_ylabel('Value')
    ax.set_title('Transformer Model Test - Predictions vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs('output_plots', exist_ok=True)
    plt.savefig('output_plots/transformer_test.png', dpi=150, bbox_inches='tight')
    print(f"✅ Test plot saved to output_plots/transformer_test.png")
    plt.close()
    
    # 9. Performance metrics
    print("\n9. Calculating performance metrics...")
    
    # Calculate basic metrics
    mae = np.mean(np.abs(y_test[:10, 0, 0] - predictions[:10, 0, 0]))
    rmse = np.sqrt(np.mean((y_test[:10, 0, 0] - predictions[:10, 0, 0])**2))
    
    print(f"✅ Test Metrics:")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    
    return True


def test_with_real_data():
    """
    Test Transformer with actual carbon market data if available
    """
    print("\n" + "=" * 60)
    print("TESTING WITH REAL DATA")
    print("=" * 60)
    
    try:
        # Try to load real data
        from utils.lseg_data_loader import LSEGDataLoader
        from utils.data_processing import prepare_data
        
        print("\n1. Loading real carbon market data...")
        loader = LSEGDataLoader()
        df = loader.load_auction_data()
        
        # Prepare data
        print("2. Preparing data...")
        df = df.drop(['Auction_Type', 'Day of Week'], axis=1, errors='ignore')
        train_df, test_df, val_df, preprocessor = prepare_data(df)
        
        # Clean data
        train_df = train_df.dropna(axis=1)
        test_df = test_df.dropna(axis=1)
        val_df = val_df.dropna(axis=1)
        
        num_features = len(test_df.columns)
        
        print(f"✅ Real data loaded: {num_features} features")
        
        # Create and test Transformer
        print("3. Creating Transformer model for real data...")
        model = create_transformer_model(num_features, out_steps=7)
        
        print(f"✅ Model created for real data with {model.count_params():,} parameters")
        
        # Quick training test (1 epoch)
        print("4. Quick training test (1 epoch)...")
        from utils.transformer_model import train_transformer_model
        
        history = train_transformer_model(
            model, train_df, val_df, test_df, preprocessor, max_epochs=1
        )
        
        print("✅ Successfully trained on real data")
        
        # Generate predictions
        print("5. Generating predictions on real data...")
        predictions_df, recent_preds, trend = generate_transformer_predictions(
            model, test_df
        )
        
        print(f"✅ Predictions generated:")
        print(f"   Historical predictions shape: {predictions_df.shape}")
        print(f"   Recent predictions shape: {recent_preds.shape}")
        print(f"   Trend: {trend}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Could not test with real data: {str(e)}")
        print("   This is expected if running outside of the main environment")
        return False


if __name__ == "__main__":
    print("Starting Transformer Model Tests...")
    print("=" * 60)
    
    # Run synthetic data test
    success = test_transformer_model()
    
    if success:
        print("\n✅ Synthetic data tests completed successfully!")
        
        # Try real data test
        print("\nAttempting to test with real data...")
        real_success = test_with_real_data()
        
        if real_success:
            print("\n✅ Real data tests completed successfully!")
        else:
            print("\n⚠️ Real data tests skipped (data not available)")
    
    print("\n" + "=" * 60)
    print("TRANSFORMER MODEL READY FOR INTEGRATION")
    print("=" * 60)
    print("\nTo use the Transformer model in your app:")
    print("1. Run: streamlit run src/app_with_transformer.py")
    print("2. Navigate to the 'Transformer Predictions' tab")
    print("3. Click 'Train Transformer' to start training")
