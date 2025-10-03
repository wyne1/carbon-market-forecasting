# Carbon Market Forecasting System

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=for-the-badge&logo=mongodb&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

> **Advanced Machine Learning System for EU Emissions Trading System (ETS) Carbon Price Prediction and Trading Strategy Optimization**

## 🌟 Overview

The Carbon Market Forecasting System is a comprehensive machine learning platform designed to predict European Union Emissions Trading System (EU ETS) carbon auction prices and optimize trading strategies. The system combines advanced deep learning techniques, sophisticated feature engineering, and automated backtesting to provide actionable insights for carbon market participants.

### Key Capabilities

- **🤖 Advanced ML Models**: LSTM and Convolutional Neural Networks for time series forecasting
- **📊 Automated Trading**: Complete backtesting engine with risk management
- **🎯 Real-time Predictions**: 7-day forward price forecasting with confidence intervals
- **📈 Interactive Dashboard**: Streamlit-based web interface for analysis and monitoring
- **📄 Professional Reporting**: Automated PDF report generation with comprehensive analytics
- **🗄️ Data Persistence**: MongoDB integration for prediction tracking and model performance analysis
- **⚡ Automated Pipeline**: One-click execution for complete end-to-end analysis

## 🏗️ Architecture

```
Carbon Market Forecasting System
├── 📊 Data Pipeline
│   ├── Multi-source data integration (COT, Auction, Options, ICE)
│   ├── Advanced feature engineering (80+ features)
│   └── Smart preprocessing with auction-aware handling
├── 🤖 ML Engine
│   ├── LSTM models for sequence learning
│   ├── Convolutional models for pattern recognition
│   └── Ensemble methods for improved accuracy
├── 💹 Trading System
│   ├── Signal generation based on predictions
│   ├── Risk management with stop-loss/take-profit
│   └── Comprehensive backtesting with performance metrics
├── 🖥️ User Interface
│   ├── Interactive Streamlit dashboard
│   ├── Real-time prediction visualization
│   └── Ensemble model training interface
└── 📄 Reporting System
    ├── Automated PDF report generation
    ├── MongoDB prediction storage
    └── Performance tracking and analytics
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MongoDB (optional, for prediction storage)
- 8GB+ RAM (recommended for model training)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/carbon-market-forecasting.git
cd carbon-market-forecasting

# Install dependencies
pip install pandas numpy tensorflow streamlit matplotlib seaborn pymongo ta-lib ta scikit-learn reportlab Pillow

# Install PDF dependencies (one-time setup)
python install_pdf_deps.py
```

### First Run

```bash
# Option 1: Complete automated analysis with PDF report
python run_analysis.py

# Option 2: Interactive dashboard
streamlit run app2.py

# Option 3: Generate PDF from existing results
python generate_pdf.py
```

## 📁 Project Structure

```
carbon-market-forecasting/
├── 📊 Core Analysis
│   ├── app2.py                     # Main Streamlit dashboard
│   ├── automated_pipeline.py       # Automated analysis pipeline
│   ├── run_analysis.py            # Simple execution script
│   └── config.py                  # Configuration management
├── 📄 Reporting
│   ├── pdf_report_generator.py    # Professional PDF generation
│   ├── generate_pdf.py            # Standalone PDF creator
│   └── install_pdf_deps.py        # PDF dependencies installer
├── 🛠️ Utilities
│   └── utils/
│       ├── backtesting.py          # Trading strategy backtesting
│       ├── data_processing.py      # Data preparation and normalization
│       ├── dataset.py             # Data loading and feature engineering
│       ├── model_utils.py         # Model creation and training
│       ├── plotting.py            # Visualization functions
│       ├── prediction_utils.py    # Prediction generation
│       ├── mongodb_utils.py       # Database integration
│       ├── smart_preprocessing.py # Auction-aware preprocessing
│       └── windowgenerator.py     # Time series windowing
├── 📊 Data
│   └── data/
│       ├── latest_data_jul.xlsx   # Current dataset
│       └── [historical datasets]  # Previous data versions
├── 🤖 Models
│   └── models/
│       ├── multi_conv_model.keras # Trained convolutional model
│       └── multi_lstm_model.keras # Trained LSTM model
├── 📓 Notebooks
│   └── notebooks/
│       ├── finalNotebook.ipynb    # Complete analysis workflow
│       ├── Multi-step-models.ipynb # Model development
│       ├── eda.ipynb              # Exploratory data analysis
│       └── [research notebooks]   # Development and testing
├── 📈 Outputs
│   └── output_plots/              # Generated reports and visualizations
│       ├── comprehensive_report_*.pdf
│       ├── equity_curve_*.png
│       ├── predictions_*.png
│       ├── trades_visualization_*.png
│       ├── performance_metrics_*.csv
│       └── trade_log_*.csv
└── 📚 Documentation
    ├── README.md                  # This file
    ├── carbon-market-doc.md       # Technical documentation
    ├── CONTRIBUTING.md            # Contribution guidelines
    └── LICENSE                    # MIT License
```

## 🎯 Features

### 1. Data Processing & Feature Engineering

- **Multi-Source Integration**: COT (Commitment of Traders), Auction Data, Options Markets, ICE Exchange Data
- **Advanced Feature Engineering**: 80+ engineered features including:
  - Time-based features (seasonality, day-of-week effects)
  - Technical indicators (RSI, MACD, Bollinger Bands)
  - Statistical features (rolling means, volatility measures)
  - Price dynamics (momentum, rate of change)
  - Interaction features (multiplicative and additive combinations)

### 2. Machine Learning Models

#### Convolutional Neural Network
```python
# Multi-step Convolutional Model
Sequential([
    Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    Dense(out_steps*num_features),
    Reshape([out_steps, num_features])
])
```

#### LSTM with Residual Connections
```python
# Residual LSTM for sequence learning
ResidualWrapper(
    tf.keras.Sequential([
        LSTM(32, return_sequences=True),
        Dense(len(FEATURES), kernel_initializer=tf.initializers.zeros())
    ])
)
```

### 3. Trading Strategy Engine

- **Signal Generation**: Prediction-based buy/sell signals
- **Risk Management**: Configurable take-profit and stop-loss levels
- **Position Sizing**: Flexible capital allocation strategies
- **Performance Metrics**: Comprehensive backtesting with:
  - Total Return & CAGR
  - Maximum Drawdown
  - Sharpe & Sortino Ratios
  - Win Rate & Profit Factor

### 4. Interactive Dashboard

<img src="images/dashboard4.png" alt="Dashboard" width="600"/>

Three main sections:
- **Backtesting Tab**: Strategy optimization and performance analysis
- **Predictions Tab**: Real-time forecasts and signal generation
- **Ensemble Tab**: Multi-model training and consensus building

### 5. Automated Reporting

Professional PDF reports including:
- Executive summary with key metrics
- Performance analysis with visualizations
- Trade-by-trade breakdown
- Risk assessment and recommendations
- Model insights and methodology

## 📊 Usage Examples

### Basic Analysis
```python
from automated_pipeline import AutomatedPipeline

# Run complete analysis
pipeline = AutomatedPipeline()
pipeline.run_full_pipeline()
```

### Custom Configuration
```python
# Custom trading parameters
config = {
    'initial_balance': 50000.0,
    'take_profit': 0.05,    # 5%
    'stop_loss': 0.025,     # 2.5%
    'position_size_fraction': 0.8,  # 80% of capital
    'max_epochs': 50
}

pipeline = AutomatedPipeline(config=config)
pipeline.run_full_pipeline()
```

### Interactive Mode
```bash
# Launch dashboard
streamlit run app2.py
```

## 🎛️ Configuration

Edit `config.py` for easy customization:

```python
# Trading Strategy Parameters
TRADING_CONFIG = {
    'initial_balance': 10000.0,
    'take_profit': 0.04,           # 4%
    'stop_loss': 0.03,             # 3%
    'position_size_fraction': 1.0,  # 100%
    'risk_free_rate': 0.01,        # 1%
}

# Model Training Parameters
MODEL_CONFIG = {
    'input_width': 7,              # Days of history
    'out_steps': 7,                # Prediction horizon
    'max_epochs': 40,              # Training epochs
}

# Quick preset switching
ACTIVE_PRESET = 'balanced'  # 'conservative', 'aggressive', or 'balanced'
```

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

| Metric | Description |
|--------|-------------|
| Total Return (%) | Overall percentage gain/loss |
| CAGR (%) | Compound Annual Growth Rate |
| Maximum Drawdown (%) | Largest peak-to-trough decline |
| Sharpe Ratio | Risk-adjusted returns |
| Sortino Ratio | Downside risk-adjusted returns |
| Win Rate (%) | Percentage of profitable trades |
| Profit Factor | Gross profit / Gross loss ratio |

## 🔧 Advanced Features

### Ensemble Modeling
```python
# Train multiple models for improved accuracy
for preds, trend, preprocessor, test_df in train_ensemble_models(data, num_models=5):
    # Combine predictions for consensus
    ensemble_predictions.append((preds, trend))
```

### Smart Preprocessing
```python
# Auction-aware data processing
preprocessor = SmartAuctionPreprocessor()
auction_df = preprocessor.preprocess_auction_data(raw_data)
```

### MongoDB Integration
```python
# Store predictions for tracking
collection = setup_mongodb_connection()
save_recent_predictions(collection, predictions, preprocessor)
```

## 📊 Model Performance

### Multi-Step Convolutional Model Results
<img src="images/convmodel3.png" alt="Conv Model Results" width="600"/>

The convolutional model demonstrates strong performance in capturing short-term price patterns and generating accurate multi-step forecasts.

## 🎯 Trading Results

### Backtesting Dashboard
<img src="images/dashboard4.png" alt="Trading Results" width="600"/>

Comprehensive backtesting interface showing:
- Real-time parameter adjustment
- Trade visualization on price charts
- Performance metrics dashboard
- Equity curve analysis

## 🔮 Prediction Interface

### Recent Predictions
<img src="images/dashboard3.png" alt="Predictions" width="600"/>

Interactive prediction interface featuring:
- 7-day forward forecasts
- Buy/sell signal visualization
- Historical prediction tracking
- Confidence intervals and trend analysis

## 🛠️ Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black .
isort .
```

### Adding New Features

1. **Data Sources**: Add new data loaders in `utils/data_processing.py`
2. **Models**: Implement new architectures in `utils/model_utils.py`
3. **Strategies**: Extend backtesting logic in `utils/backtesting.py`
4. **Visualizations**: Add charts in `utils/plotting.py`

### Model Development Workflow

```python
# 1. Data Preparation
merged_df, options_df = load_and_preprocess_data()

# 2. Feature Engineering
engineered_df = DataPreprocessor.engineer_auction_features(merged_df)

# 3. Model Training
model = create_model(num_features, out_steps)
history = train_model(model, train_df, val_df, test_df, preprocessor)

# 4. Evaluation
predictions_df, recent_preds, trend = generate_model_predictions(model, test_df)

# 5. Backtesting
trade_log_df, metrics, balance_df = backtest_model_with_metrics(...)
```

## 📋 API Reference

### Core Functions

```python
# Data Processing
load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame]
prepare_data(merged_df) -> Tuple[pd.DataFrame, ...]

# Model Operations
create_model(num_features, out_steps) -> tf.keras.Model
train_model(model, train_df, val_df, test_df, preprocessor) -> History

# Prediction Generation
generate_model_predictions(model, test_df) -> Tuple[pd.DataFrame, ...]

# Backtesting
backtest_model_with_metrics(...) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]

# Reporting
generate_pdf_report(output_dir) -> Path
```

## 🎨 Customization

### Custom Models
```python
def create_custom_model(num_features, out_steps):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(out_steps * num_features),
        tf.keras.layers.Reshape([out_steps, num_features])
    ])
    return model
```

### Custom Trading Strategies
```python
def custom_signal_generator(predictions_df, current_price):
    # Implement your custom logic
    signal = 'Buy' if custom_condition else 'Sell'
    return signal
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Errors**
   ```python
   # Reduce model complexity or batch size
   MODEL_CONFIG['max_epochs'] = 20
   ```

2. **Data Loading Issues**
   ```python
   # Check file paths and formats
   DATA_CONFIG['data_file'] = 'data/your_file.xlsx'
   ```

3. **MongoDB Connection**
   ```bash
   # Start MongoDB service
   brew services start mongodb/brew/mongodb-community
   # or
   sudo systemctl start mongod
   ```

4. **PDF Generation**
   ```bash
   # Install dependencies
   python install_pdf_deps.py
   ```

### Performance Optimization

```python
# Use GPU acceleration
import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')

# Reduce memory usage
tf.config.experimental.set_memory_growth(gpu, True)
```

## 📚 Documentation

- **Technical Guide**: See `carbon-market-doc.md` for detailed technical documentation
- **API Reference**: Complete function documentation in docstrings
- **Examples**: Jupyter notebooks in `notebooks/` directory
- **Contributing**: See `CONTRIBUTING.md` for development guidelines

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- New data sources and features
- Alternative model architectures
- Enhanced trading strategies
- Performance optimizations
- Documentation improvements

## 📊 Roadmap

### Near-term (Q1 2024)
- [ ] Integration with real-time data feeds
- [ ] Advanced ensemble methods (Autoformer, NIXTLA)
- [ ] Enhanced risk management features
- [ ] Multi-timeframe analysis

### Medium-term (Q2-Q3 2024)
- [ ] Reinforcement learning for strategy optimization
- [ ] Sentiment analysis integration
- [ ] Cross-market correlation analysis
- [ ] Cloud deployment and scaling

### Long-term (Q4 2024+)
- [ ] High-frequency trading capabilities
- [ ] Alternative data sources (satellite, weather)
- [ ] Advanced portfolio optimization
- [ ] Regulatory compliance tools

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow Team** for the machine learning framework
- **Streamlit Team** for the web application framework
- **Carbon Market Community** for domain expertise and feedback
- **Open Source Contributors** for various libraries and tools used

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/carbon-market-forecasting/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/carbon-market-forecasting/discussions)
- **Email**: your.email@domain.com

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/carbon-market-forecasting&type=Date)](https://star-history.com/#your-username/carbon-market-forecasting&Date)

---

**Built with ❤️ for the Carbon Market Community**

*Empowering data-driven decisions in the fight against climate change.*

# TODO
- Add Ensemble Model Predictions in the PDF.
- Add Price distribution chart in the PDF.
- MA / Cover Ratio features -- distance from auction