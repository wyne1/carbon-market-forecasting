from IPython.display import display
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from pymongo import MongoClient
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from utils.dataset import DataPreprocessor
from utils.data_processing import prepare_data
from utils.plotting import (display_performance_metrics, display_trade_log, plot_equity_curve, 
                            plot_model_results_with_trades, plot_recent_predictions, plot_ensemble_statistics, plot_ensemble_predictions_realtime)
from utils.model_utils import create_model, train_model, generate_model_predictions, train_ensemble_models
from utils.mongodb_utils import get_stored_predictions, setup_mongodb_connection, save_recent_predictions
from utils.backtesting import backtest_model_with_metrics
from utils.smart_preprocessing import SmartAuctionPreprocessor
from utils.lseg_data_loader import LSEGDataLoader

# Import Transformer utilities
from utils.transformer_model import create_transformer_model, train_transformer_model
from utils.transformer_utils import (
    generate_transformer_predictions,
    plot_transformer_predictions,
    compare_transformer_vs_baseline,
    generate_transformer_report,
    visualize_transformer_attention
)

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['carbon_market_predictions']
collection = db['recent_predictions']


# NEW: Modified load function for your app.py
@st.cache_data
def load_and_preprocess_data_smart():
    """
    REPLACE your current load_and_preprocess_data function with this
    """
    auction_loader = LSEGDataLoader()
    df = auction_loader.load_auction_data()
    return df


@st.cache_resource
def prepare_data_and_train_model(merged_df):
    merged_df = merged_df.drop(['Auction_Type', 'Day of Week'], axis=1, errors='ignore')
    
    print(f"Using the following columns: {merged_df.columns}")
    train_df, test_df, val_df, preprocessor = prepare_data(merged_df)
    
    train_df = train_df.dropna(axis=1)
    test_df = test_df.dropna(axis=1)
    val_df = val_df.dropna(axis=1)

    num_features = len(test_df.columns)
    OUT_STEPS = 7
    model = create_model(num_features, OUT_STEPS)
    history = train_model(model, train_df, val_df, test_df, preprocessor)
    predictions_df, recent_preds, trend = generate_model_predictions(model, test_df)
    
    return model, preprocessor, test_df, predictions_df, recent_preds, trend


@st.cache_resource
def prepare_and_train_transformer(merged_df, max_epochs=50):
    """
    Prepare data and train Transformer model
    """
    merged_df = merged_df.drop(['Auction_Type', 'Day of Week'], axis=1, errors='ignore')
    
    print(f"Training Transformer with columns: {merged_df.columns}")
    train_df, test_df, val_df, preprocessor = prepare_data(merged_df)
    
    train_df = train_df.dropna(axis=1)
    test_df = test_df.dropna(axis=1)
    val_df = val_df.dropna(axis=1)

    num_features = len(test_df.columns)
    OUT_STEPS = 7
    
    # Create and train Transformer model
    transformer_model = create_transformer_model(num_features, OUT_STEPS)
    history = train_transformer_model(
        transformer_model, train_df, val_df, test_df, preprocessor, max_epochs
    )
    
    # Generate predictions
    predictions_df, recent_preds, trend = generate_transformer_predictions(
        transformer_model, test_df
    )
    
    return transformer_model, preprocessor, test_df, predictions_df, recent_preds, trend, history


def main():
    st.set_page_config(page_title="Trading Strategy Backtesting Dashboard", layout="wide")
    
    # Add custom CSS
    st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    h1 {
        margin-top: 0rem;
    }
    .transformer-metric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🚀 Carbon Market Forecasting Dashboard")

    # Load data (this will be cached)
    merged_df = load_and_preprocess_data_smart()
    
    # Train baseline model (this will be cached)
    model, preprocessor, test_df, predictions_df, recent_preds, trend = prepare_data_and_train_model(merged_df)

    # Create tabs - Added Transformer tab
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Backtesting", 
        "🔮 Predictions", 
        "🎯 Ensemble Predictions", 
        "🤖 Transformer Predictions"
    ])

    with tab1:
        # Sidebar Inputs
        st.sidebar.header("Backtesting Parameters")
        initial_balance = st.sidebar.number_input("Initial Balance ($)", value=10000.0, min_value=0.0, step=1000.0)
        take_profit = st.sidebar.slider("Take Profit (%)", min_value=0.0, max_value=100.0, value=4.0, step=0.1)
        stop_loss = st.sidebar.slider("Stop Loss (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
        position_size_fraction = st.sidebar.slider("Position Size Fraction (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
        risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=1.0, min_value=0.0, max_value=10.0, step=0.1)

        st.sidebar.markdown("---")
        
        if st.sidebar.button("Re-Train Model"):
            with st.spinner("Re-training model..."):
                st.cache_resource.clear()
                st.success("Model cache cleared. The model will be retrained on the next run.")
            st.rerun()
        
        if st.sidebar.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                trade_log_df, performance_metrics, balance_history_df = backtest_model_with_metrics(
                    model,
                    test_df,
                    7,
                    7,
                    initial_balance,
                    take_profit / 100.0,
                    stop_loss / 100.0,
                    position_size_fraction=position_size_fraction / 100.0,
                    risk_free_rate=risk_free_rate / 100.0,
                    preprocessor=preprocessor
                )

            col1, col2 = st.columns([0.65, 0.35])
            with col1:
                st.subheader("Trade Log")
                display_trade_log(trade_log_df)

            with col2:
                st.subheader("Performance Metrics")
                display_performance_metrics(performance_metrics)

            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                st.subheader("Equity Curve")
                plot_equity_curve(balance_history_df)

            with col2:
                st.subheader("Trades")
                plot_model_results_with_trades(test_df, predictions_df, trade_log_df, preprocessor)

    with tab2:
        st.header("📈 Recent Predictions")
        plot_recent_predictions(recent_preds, trend, test_df, preprocessor)
        collection = setup_mongodb_connection()
        
        st.header("💾 Stored Predictions")
        with st.spinner("Loading stored predictions..."):
            stored_predictions = get_stored_predictions(collection)
            if stored_predictions:
                # Only show the most recent N predictions
                N = 20  # Change this number as desired
                recent_predictions = stored_predictions[-N:] if len(stored_predictions) > N else stored_predictions

                data = []
                for pred in recent_predictions:
                    try:
                        pred_diff = float(pred['pred_diff'])
                    except (ValueError, TypeError):
                        pred_diff = 0.0
                    
                    row = [pred['date'], pred['trade_direction'], pred_diff] + pred['predictions']
                    data.append(row)
                
                columns = ['Date', 'Trade Direction', 'Pred Diff'] + [f'Day {i+1}' for i in range(len(recent_predictions[0]['predictions']))]
                df = pd.DataFrame(data, columns=columns)
                df['Date'] = pd.to_datetime(df['Date']).dt.date

                def color_trade_direction(val):
                    color = 'green' if val == 'Buy' else 'red'
                    return f'color: {color}'
                
                st.dataframe(df.style
                             .map(color_trade_direction, subset=['Trade Direction'])
                             .highlight_max(axis=1, subset=df.columns[3:])
                             .format({'Pred Diff': '{:.4f}'})
                             , use_container_width=True, hide_index=True
                )
                st.caption(f"Showing {len(df)} most recent stored predictions.")
            else:
                st.info("No stored predictions found.")

        # st.header("Save Predictions")
        if st.button("Save Recent Predictions"):
            collection = setup_mongodb_connection()
            save_message = save_recent_predictions(collection, recent_preds, preprocessor)
            st.success(save_message)

    with tab3:
        st.header("🎯 Ensemble Predictions")
        
        col1, col2 = st.columns(2)
        with col1:
            num_models = st.number_input("Number of Models", min_value=2, max_value=30, value=3)
        with col2:
            max_epochs = st.number_input("Max Epochs", min_value=5, max_value=100, value=40)
        
        if st.button("Train Ensemble"):
            plot_container = st.empty()
            predictions_list = []
            progress_bar = st.progress(0)
            
            # Train models and update plot
            for i, (preds, trend, preprocessor, test_df) in enumerate(train_ensemble_models(merged_df, num_models, max_epochs)):
                predictions_list.append((preds, trend))
                plot_ensemble_predictions_realtime(predictions_list, test_df, preprocessor, plot_container)
                progress_bar.progress((i + 1) / num_models)
            
            # After all predictions are complete, show statistics
            st.subheader("Ensemble Analysis")
            plot_ensemble_statistics(predictions_list, test_df, preprocessor)

    with tab4:
        st.header("🤖 Transformer Model Predictions")
        
        # Add description
        with st.expander("ℹ️ About Transformer Models"):
            st.markdown("""
            **Transformer models** use self-attention mechanisms to capture complex temporal dependencies 
            in time series data. Key advantages:
            - 🎯 **Attention Mechanism**: Focuses on relevant time periods
            - 📊 **Long-range Dependencies**: Captures patterns across extended time periods
            - 🔄 **Parallel Processing**: Faster training than sequential models
            - 📈 **Better Accuracy**: Often outperforms traditional LSTM/CNN models
            """)
        
        # Training controls
        col1, col2, col3 = st.columns(3)
        with col1:
            transformer_epochs = st.number_input(
                "Training Epochs", 
                min_value=10, 
                max_value=100, 
                value=50,
                key="transformer_epochs"
            )
        with col2:
            retrain_transformer = st.button("🔄 Train Transformer", key="train_transformer")
        with col3:
            save_transformer = st.button("💾 Save Model", key="save_transformer")
        
        # Train or load Transformer model
        if retrain_transformer or 'transformer_model' not in st.session_state:
            with st.spinner("🚀 Training Transformer model... This may take a few minutes."):
                try:
                    (transformer_model, transformer_preprocessor, transformer_test_df, 
                     transformer_predictions_df, transformer_recent_preds, 
                     transformer_trend, transformer_history) = prepare_and_train_transformer(
                        merged_df, max_epochs=transformer_epochs
                    )
                    
                    # Store in session state
                    st.session_state.transformer_model = transformer_model
                    st.session_state.transformer_preprocessor = transformer_preprocessor
                    st.session_state.transformer_test_df = transformer_test_df
                    st.session_state.transformer_predictions_df = transformer_predictions_df
                    st.session_state.transformer_recent_preds = transformer_recent_preds
                    st.session_state.transformer_trend = transformer_trend
                    st.session_state.transformer_history = transformer_history
                    
                    st.success("✅ Transformer model trained successfully!")
                except Exception as e:
                    st.error(f"❌ Error training Transformer: {str(e)}")
                    st.stop()
        
        # Check if model exists in session state
        if 'transformer_model' in st.session_state:
            # Display Transformer predictions
            st.subheader("📊 Transformer Model Performance")
            
            # Compare with baseline
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate comparison metrics
            comparison = compare_transformer_vs_baseline(
                st.session_state.transformer_predictions_df,
                predictions_df,
                test_df,
                metric='mae'
            )
            
            with col1:
                st.metric(
                    "Transformer MAE",
                    f"€{comparison.get('transformer_mae', 0):.2f}",
                    delta=f"{comparison.get('improvement_pct', 0):.1f}% vs CNN"
                )
            
            with col2:
                st.metric(
                    "Transformer R²",
                    f"{comparison.get('transformer_r2', 0):.3f}",
                    delta=f"{(comparison.get('transformer_r2', 0) - comparison.get('baseline_r2', 0)):.3f}"
                )
            
            with col3:
                st.metric(
                    "Prediction Trend",
                    st.session_state.transformer_trend.upper(),
                    delta="7-day forecast"
                )
            
            with col4:
                if st.session_state.transformer_trend == 'positive':
                    st.metric("Signal", "BUY", delta="↑")
                else:
                    st.metric("Signal", "SELL", delta="↓")
            
            # Plot predictions
            st.subheader("📈 Transformer Predictions Visualization")
            
            # Create the plot
            fig = plot_transformer_predictions(
                st.session_state.transformer_test_df,
                st.session_state.transformer_predictions_df,
                st.session_state.transformer_recent_preds,
                title="Transformer Model - Carbon Price Predictions"
            )
            st.pyplot(fig)
            plt.close()
            
            # Training history
            if st.session_state.transformer_history:
                st.subheader("📉 Training History")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    history = st.session_state.transformer_history
                    ax.plot(history.history['loss'], label='Training Loss')
                    ax.plot(history.history['val_loss'], label='Validation Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title('Model Loss During Training')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    # Display recent predictions table
                    st.subheader("🔮 7-Day Forward Forecast")
                    recent = st.session_state.transformer_recent_preds[['Auc Price']].tail(7)
                    recent.columns = ['Predicted Price (€/tCO2)']
                    recent.index = pd.date_range(
                        start=test_df.index[-1] + pd.Timedelta(days=1),
                        periods=7
                    )
                    recent.index.name = 'Date'
                    
                    # Color code based on trend
                    def highlight_trend(s):
                        trend = np.gradient(s.values)
                        colors = ['background-color: #90EE90' if t > 0 else 'background-color: #FFB6C1' 
                                 for t in trend]
                        return colors
                    
                    st.dataframe(
                        recent.style.apply(highlight_trend).format("{:.2f}"),
                        use_container_width=True
                    )
            
            # Backtest with Transformer
            st.subheader("💹 Transformer Model Backtesting")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Run Transformer Backtest", key="transformer_backtest"):
                    with st.spinner("Running Transformer backtest..."):
                        transformer_trade_log, transformer_metrics, transformer_balance = backtest_model_with_metrics(
                            st.session_state.transformer_model,
                            st.session_state.transformer_test_df,
                            7, 7,
                            initial_balance,
                            take_profit / 100.0,
                            stop_loss / 100.0,
                            position_size_fraction / 100.0,
                            risk_free_rate / 100.0,
                            preprocessor=st.session_state.transformer_preprocessor
                        )
                        
                        st.session_state.transformer_trade_log = transformer_trade_log
                        st.session_state.transformer_metrics = transformer_metrics
                        st.session_state.transformer_balance = transformer_balance
            
            with col2:
                if st.button("Generate Report", key="transformer_report"):
                    if 'transformer_metrics' in st.session_state:
                        report = generate_transformer_report(
                            st.session_state.transformer_model,
                            st.session_state.transformer_test_df,
                            st.session_state.transformer_predictions_df,
                            st.session_state.transformer_metrics,
                            save_path='output_plots/transformer_report.txt'
                        )
                        st.text_area("Transformer Model Report", report, height=300)
                        st.success("Report saved to output_plots/transformer_report.txt")
            
            # Display backtest results if available
            if 'transformer_metrics' in st.session_state:
                st.markdown("---")
                col1, col2 = st.columns([0.6, 0.4])
                
                with col1:
                    st.subheader("Transformer Trade Performance")
                    display_trade_log(st.session_state.transformer_trade_log)
                
                with col2:
                    st.subheader("Transformer Metrics")
                    display_performance_metrics(st.session_state.transformer_metrics)
                    
                    # Add comparison with baseline
                    st.markdown("### 📊 Model Comparison")
                    if 'transformer_metrics' in st.session_state:
                        baseline_return = performance_metrics.get('Total Return (%)', 0) if 'performance_metrics' in locals() else 0
                        transformer_return = st.session_state.transformer_metrics.get('Total Return (%)', 0)
                        
                        comparison_df = pd.DataFrame({
                            'Metric': ['Total Return (%)', 'Sharpe Ratio', 'Win Rate (%)'],
                            'CNN Model': [
                                performance_metrics.get('Total Return (%)', 0) if 'performance_metrics' in locals() else 0,
                                performance_metrics.get('Sharpe Ratio', 0) if 'performance_metrics' in locals() else 0,
                                performance_metrics.get('Win Rate (%)', 0) if 'performance_metrics' in locals() else 0
                            ],
                            'Transformer': [
                                st.session_state.transformer_metrics.get('Total Return (%)', 0),
                                st.session_state.transformer_metrics.get('Sharpe Ratio', 0),
                                st.session_state.transformer_metrics.get('Win Rate (%)', 0)
                            ]
                        })
                        
                        st.dataframe(
                            comparison_df.style.highlight_max(axis=1, subset=['CNN Model', 'Transformer']),
                            use_container_width=True,
                            hide_index=True
                        )
            
            # Save Transformer predictions to MongoDB
            if st.button("💾 Save Transformer Predictions", key="save_transformer_preds"):
                collection = setup_mongodb_connection()
                # Add a tag to distinguish Transformer predictions
                transformer_recent = st.session_state.transformer_recent_preds.copy()
                save_message = save_recent_predictions(
                    collection, 
                    transformer_recent, 
                    st.session_state.transformer_preprocessor,
                    model_type="Transformer"  # Add this parameter to your save function if needed
                )
                st.success(f"Transformer {save_message}")
            
            # Save model
            if save_transformer:
                try:
                    st.session_state.transformer_model.save('models/transformer_carbon_model.keras')
                    st.success("✅ Transformer model saved to models/transformer_carbon_model.keras")
                except Exception as e:
                    st.error(f"❌ Error saving model: {str(e)}")
        
        else:
            st.info("🔄 Click 'Train Transformer' to start training the model.")


if __name__ == "__main__":
    collection = setup_mongodb_connection()
    main()
