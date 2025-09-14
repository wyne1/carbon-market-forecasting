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

from utils.dataset import DataPreprocessor
from utils.data_processing import prepare_data
from utils.plotting import (display_performance_metrics, display_trade_log, plot_equity_curve, 
                            plot_model_results_with_trades, plot_recent_predictions, plot_ensemble_statistics, plot_ensemble_predictions_realtime)
from utils.model_utils import create_model, train_model, generate_model_predictions, train_ensemble_models
from utils.mongodb_utils import get_stored_predictions, setup_mongodb_connection, save_recent_predictions
from utils.backtesting import backtest_model_with_metrics
from utils.smart_preprocessing import SmartAuctionPreprocessor
from utils.lseg_data_loader import LSEGDataLoader
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
    # Load new data
    df = pd.read_excel('data/latest_data_jul.xlsx').set_index('Unnamed: 0')

    auction_loader = LSEGDataLoader()
    df = auction_loader.load_auction_data()
    
    print(f"Merged DF: {df}")
    return df

@st.cache_resource
def prepare_data_and_train_model(merged_df):
    # print("Column Types:")
    # for col in merged_df.columns:
    #     print(f"{col}: {merged_df[col].dtype}")

    merged_df = merged_df.drop(['Auction_Type', 'Day of Week'], axis=1)
    

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


def main():
    st.set_page_config(page_title="Trading Strategy Backtesting Dashboard", layout="wide")
    # Add custom CSS to style the buttons
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
    </style>
    """, unsafe_allow_html=True)

    st.title("Trading Strategy Backtesting Dashboard")

    # Load data (this will be cached)
    merged_df = load_and_preprocess_data_smart()
    # merged_df.to_csv('testing_merged.csv')
    
    # Train model (this will be cached unless the cache was just cleared)
    model, preprocessor, test_df, predictions_df, recent_preds, trend = prepare_data_and_train_model(merged_df)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Backtesting", "Predictions", "Ensemble Predictions"])

    with tab1:
        # Sidebar Inputs
        st.sidebar.header("Backtesting Parameters")
        initial_balance = st.sidebar.number_input("Initial Balance ($)", value=10000.0, min_value=0.0, step=1000.0)
        take_profit = st.sidebar.slider("Take Profit (%)", min_value=0.0, max_value=100.0, value=4.0, step=0.1)
        stop_loss = st.sidebar.slider("Stop Loss (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
        position_size_fraction = st.sidebar.slider("Position Size Fraction (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
        risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=1.0, min_value=0.0, max_value=10.0, step=0.1)

        st.sidebar.markdown("---")
        
        # st.sidebar.header("Model and Data Inputs")
        # input_width = st.sidebar.number_input("Input Width (Days)", min_value=1, value=7, step=1)
        # out_steps = st.sidebar.number_input("Out Steps (Days)", min_value=1, value=7, step=1)

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
        st.header("Recent Predictions")
        plot_recent_predictions(recent_preds, trend, test_df, preprocessor)
        collection = setup_mongodb_connection()
        
        st.header("Stored Predictions")
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
        st.header("Ensemble Predictions")
        
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

if __name__ == "__main__":
    collection = setup_mongodb_connection()
    main()
    