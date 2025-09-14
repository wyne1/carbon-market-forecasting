# config.py
"""
Configuration file for the Automated Carbon Market Analysis Pipeline
=====================================================================

Modify these parameters to customize the analysis without changing the main code.
"""

# Trading Strategy Parameters
TRADING_CONFIG = {
    'initial_balance': 10000.0,        # Starting capital in USD
    'take_profit': 0.04,               # Take profit level (4% = 0.04)
    'stop_loss': 0.03,                 # Stop loss level (3% = 0.03)
    'position_size_fraction': 1.0,     # Fraction of capital per trade (1.0 = 100%)
    'risk_free_rate': 0.01,            # Risk-free rate for Sharpe ratio (1% = 0.01)
}

# Model Training Parameters
MODEL_CONFIG = {
    'input_width': 7,                  # Days of historical data for prediction
    'out_steps': 7,                    # Days to predict forward
    'max_epochs': 40,                  # Maximum training epochs
}

# Data and Output Settings
DATA_CONFIG = {
    'data_file': 'data/latest_data_jul.xlsx',  # Path to your data file
    'output_dir': 'output_plots',              # Directory for saving plots
}

# MongoDB Settings
MONGODB_CONFIG = {
    'save_to_mongodb': True,           # Whether to save predictions to MongoDB
    'mongodb_host': 'localhost',       # MongoDB host
    'mongodb_port': 27017,             # MongoDB port
}

# Plotting Settings
PLOT_CONFIG = {
    'figure_size': (12, 8),           # Figure size (width, height) in inches
    'dpi': 300,                       # Resolution for saved plots
    'plot_style': 'default',          # Matplotlib style
}

# PDF Report Settings
PDF_CONFIG = {
    'generate_pdf': True,             # Whether to generate PDF report
    'pdf_page_size': 'A4',           # Page size (A4, letter)
    'include_charts': True,           # Include charts in PDF
    'max_recent_trades': 10,          # Number of recent trades to show
    'include_ensemble_analysis': True, # Include ensemble model analysis
    'ensemble_num_models': 5,         # Number of models for ensemble
    'ensemble_max_epochs': 30,        # Max epochs for ensemble training
}

# Combine all configurations
FULL_CONFIG = {
    **TRADING_CONFIG,
    **MODEL_CONFIG,
    **DATA_CONFIG,
    **MONGODB_CONFIG,
    **PLOT_CONFIG,
    **PDF_CONFIG
}

# Predefined configuration presets
PRESETS = {
    'conservative': {
        **FULL_CONFIG,
        'take_profit': 0.02,           # 2% profit target
        'stop_loss': 0.015,            # 1.5% stop loss
        'position_size_fraction': 0.5,  # Use only 50% of capital
        'ensemble_num_models': 3,      # Fewer models for speed
        'ensemble_max_epochs': 20,     # Shorter training
    },
    
    'aggressive': {
        **FULL_CONFIG,
        'take_profit': 0.08,           # 8% profit target
        'stop_loss': 0.05,             # 5% stop loss
        'position_size_fraction': 1.0,  # Use full capital
        'max_epochs': 60,              # More training
        'ensemble_num_models': 7,      # More models for accuracy
        'ensemble_max_epochs': 40,     # Longer training
    },
    
    'balanced': {
        **FULL_CONFIG,
        'take_profit': 0.04,           # 4% profit target
        'stop_loss': 0.02,             # 3% stop loss
        'position_size_fraction': 0.8,  # Use 80% of capital
        'ensemble_num_models': 10,      # Balanced approach
        'ensemble_max_epochs': 30,     # Moderate training
    }
}

# Quick config selector - change this to switch between presets
ACTIVE_PRESET = 'balanced'  # Options: 'conservative', 'aggressive', 'balanced', or None for FULL_CONFIG

def get_config():
    """
    Get the active configuration
    """
    if ACTIVE_PRESET and ACTIVE_PRESET in PRESETS:
        return PRESETS[ACTIVE_PRESET]
    else:
        return FULL_CONFIG