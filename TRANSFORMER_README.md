# Transformer Model Implementation for Carbon Market Forecasting

## 🤖 Overview

This implementation adds a state-of-the-art **Transformer-based model** to your carbon market forecasting system. Transformers use self-attention mechanisms to capture complex temporal dependencies in time series data, often outperforming traditional LSTM and CNN models.

## 🚀 Quick Start

### 1. Test the Implementation
First, verify the Transformer model works correctly:
```bash
python test_transformer.py
```

### 2. Run the Updated App
Launch the Streamlit app with the new Transformer tab:
```bash
streamlit run src/app_with_transformer.py
```

### 3. Navigate to Transformer Tab
In the app, click on the "🤖 Transformer Predictions" tab (4th tab)

### 4. Train the Model
Click "Train Transformer" to start training. The model will:
- Load your carbon market data
- Preprocess it using your existing pipeline
- Train a Transformer model with attention mechanisms
- Generate 7-day forward predictions

## 📁 New Files Created

### Core Implementation
- **`utils/transformer_model.py`**: Core Transformer architecture
  - Multi-head attention mechanism
  - Positional encoding
  - Transformer blocks
  - Temporal Fusion Transformer implementation

- **`utils/transformer_utils.py`**: Utility functions
  - Data preparation for Transformer
  - Prediction generation
  - Visualization functions
  - Model comparison utilities
  - Report generation

- **`src/app_with_transformer.py`**: Updated Streamlit app
  - New Transformer tab
  - Training interface
  - Prediction visualization
  - Performance comparison with baseline
  - Backtesting capabilities

- **`test_transformer.py`**: Test script
  - Validates model implementation
  - Tests with synthetic data
  - Checks integration with existing pipeline

## 🏗️ Architecture

### Transformer Model Structure

```
Input (7 days, N features)
    ↓
Input Embedding (Dense Layer)
    ↓
Positional Encoding
    ↓
Transformer Block 1
├── Multi-Head Attention (4 heads)
├── Layer Normalization
├── Feed-Forward Network
└── Residual Connections
    ↓
Transformer Block 2
├── Multi-Head Attention (4 heads)
├── Layer Normalization
├── Feed-Forward Network
└── Residual Connections
    ↓
Global Average Pooling
    ↓
Dense Layers (256 → 128)
    ↓
Output Reshape (7 days, N features)
```

### Key Components

1. **Multi-Head Attention**: Allows the model to focus on different time periods simultaneously
2. **Positional Encoding**: Adds temporal information to the model
3. **Residual Connections**: Helps with gradient flow and training stability
4. **Layer Normalization**: Improves training stability

## 🎯 Features in the Transformer Tab

### Training Section
- **Configurable epochs**: Adjust training duration (10-100 epochs)
- **Real-time training**: Shows progress during training
- **Model persistence**: Save trained models for later use

### Performance Metrics
- **MAE Comparison**: Compare Transformer vs CNN baseline
- **R² Score**: Model fit quality
- **Improvement percentage**: Quantified performance gains
- **Trend detection**: Bullish/bearish signal generation

### Visualizations
1. **Historical Performance**: Shows model predictions vs actual prices
2. **7-Day Forecast**: Future price predictions with trend indicators
3. **Training History**: Loss curves for model convergence monitoring
4. **Color-coded predictions**: Green for uptrend, red for downtrend

### Backtesting
- Run full backtesting with Transformer predictions
- Compare trading performance with baseline model
- Generate detailed performance reports

### Model Comparison Table
Shows side-by-side metrics:
- Total Return (%)
- Sharpe Ratio
- Win Rate (%)

## 📊 Expected Performance

Based on typical results, the Transformer model should provide:
- **5-15% improvement** in MAE over CNN baseline
- **Better long-range dependencies** capture
- **More stable predictions** during volatile periods
- **Higher Sharpe ratios** in backtesting

## 🔧 Configuration Options

### Model Parameters (in `transformer_model.py`)
```python
TransformerCarbonModel(
    num_features=10,      # Number of input features
    input_steps=7,        # Days of history to use
    output_steps=7,       # Days to predict forward
    d_model=128,          # Model dimension (increase for complexity)
    num_heads=4,          # Attention heads (must divide d_model)
    num_layers=2,         # Number of transformer blocks
    dropout_rate=0.1      # Regularization
)
```

### Training Parameters
```python
transformer.train(
    epochs=50,           # Training epochs
    batch_size=32,       # Batch size
    patience=10,         # Early stopping patience
)
```

## 🐛 Troubleshooting

### Issue: "Module not found" error
**Solution**: Ensure all dependencies are installed:
```bash
pip install tensorflow>=2.10.0 scikit-learn matplotlib pandas numpy
```

### Issue: Training is slow
**Solutions**:
1. Reduce `d_model` to 64
2. Reduce `num_layers` to 1
3. Decrease `max_epochs`
4. Use GPU if available

### Issue: Poor performance
**Solutions**:
1. Increase training epochs
2. Tune hyperparameters
3. Add more features to the dataset
4. Check data quality and preprocessing

### Issue: Memory errors
**Solutions**:
1. Reduce batch size
2. Reduce model size (d_model)
3. Use fewer transformer layers

## 🚦 Integration with Existing System

The Transformer implementation is designed to be **fully compatible** with your existing codebase:

1. **Uses same data pipeline**: Works with your `LSEGDataLoader` and preprocessing
2. **Compatible with backtesting**: Uses existing `backtest_model_with_metrics`
3. **MongoDB integration**: Saves predictions to same database
4. **Consistent interface**: Follows same pattern as CNN/LSTM models

## 📈 Advanced Usage

### Custom Transformer Architecture
Modify `utils/transformer_model.py` to experiment with:
- Different attention mechanisms (e.g., sparse attention)
- Variable selection networks
- Quantile predictions for uncertainty estimation
- Temporal convolutional layers

### Ensemble with Existing Models
Combine Transformer with CNN/LSTM:
```python
# Average predictions from multiple models
ensemble_pred = (transformer_pred + cnn_pred + lstm_pred) / 3
```

### Feature Importance Analysis
The attention weights can show which time periods are most important:
```python
attention_weights = calculate_attention_weights(model, input_data)
visualize_transformer_attention(attention_weights, feature_names)
```

## 🔮 Future Enhancements

Potential improvements to consider:
1. **Autoformer**: Decomposition-based Transformer for better trend capture
2. **Informer**: Efficient Transformer for longer sequences
3. **FEDformer**: Frequency-enhanced Transformer
4. **Uncertainty quantification**: Prediction intervals
5. **Multi-task learning**: Predict multiple horizons simultaneously

## 📝 Notes

- The Transformer model typically requires more data than CNN/LSTM
- Training time is longer but predictions are often more accurate
- Best suited for capturing long-range dependencies in the data
- Consider using GPU for faster training

## 🤝 Support

If you encounter any issues or need clarification:
1. Check the test script output: `python test_transformer.py`
2. Review the error messages in the Streamlit app
3. Verify data preprocessing is working correctly
4. Ensure all dependencies are up to date

---

**Successfully integrated Transformer model into your Carbon Market Forecasting System! 🎉**
