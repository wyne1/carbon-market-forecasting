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

from utils.train_model import create_model, train_model
from utils.dataset import MarketData, DataPreprocessor, prepare_data
from utils.backtesting import backtest_model_with_metrics
from utils.plotting import (
    display_performance_metrics,
    display_trade_log,
    plot_equity_curve,
    plot_model_results_with_trades,
    plot_recent_predictions
)
from utils.prediction_utils import generate_model_predictions
from utils.mongodb_utils import get_stored_predictions, setup_mongodb_connection, save_recent_predictions
# MongoDB setup

client = MongoClient('mongodb://localhost:27017/')
db = client['carbon_market_predictions']
collection = db['recent_predictions']


@st.cache_data
def load_and_preprocess_data():
    cot_df, auction_df, eua_df, ta_df, fundamentals_df = MarketData.latest(Path('data'))
    cot_df = cot_df.set_index('Date').resample('W', origin='end').mean().reset_index()
    auction_df = auction_df.set_index('Date').resample('D').mean().reset_index()

    auction_df = auction_df[7:]
    auction_df.loc[:, 'Premium/discount-settle'] = auction_df['Premium/discount-settle'].ffill()
    auction_df.loc[:, ['Auc Price', 'Median Price', 'Cover Ratio', 'Spot Value',
    'Auction Spot Diff', 'Median Spot Diff', 'Premium/discount-settle']] = auction_df[['Auc Price', 'Median Price', 'Cover Ratio', 'Spot Value', 
                                                                                              'Auction Spot Diff', 'Median Spot Diff', 'Premium/discount-settle']].ffill()

    merged_df = DataPreprocessor.engineer_auction_features(auction_df)
    return merged_df

@st.cache_resource
def prepare_data_and_train_model(merged_df):
    train_df, test_df, val_df, preprocessor = prepare_data(merged_df)
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
    merged_df = load_and_preprocess_data()

    # Train model (this will be cached unless the cache was just cleared)
    model, preprocessor, test_df, predictions_df, recent_preds, trend = prepare_data_and_train_model(merged_df)

    # Create tabs
    tab1, tab2 = st.tabs(["Backtesting", "Predictions"])

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

            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                st.subheader("Trade Log")
                display_trade_log(trade_log_df)

            with col2:
                st.subheader("Performance Metrics")
                display_performance_metrics(performance_metrics)

            col1, col2 = st.columns(2)

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
                data = []
                for pred in stored_predictions:
                    try:
                        pred_diff = float(pred['pred_diff'])
                    except (ValueError, TypeError):
                        pred_diff = 0.0
                    
                    row = [pred['date'], pred['trade_direction'], pred_diff] + pred['predictions']
                    data.append(row)
                
                columns = ['Date', 'Trade Direction', 'Pred Diff'] + [f'Day {i+1}' for i in range(len(stored_predictions[0]['predictions']))]
                df = pd.DataFrame(data, columns=columns)
                df['Date'] = pd.to_datetime(df['Date']).dt.date

                def color_trade_direction(val):
                    color = 'green' if val == 'Buy' else 'red'
                    return f'color: {color}'
                
                st.dataframe(df.style
                             .map(color_trade_direction, subset=['Trade Direction'])
                             .highlight_max(axis=1, subset=df.columns[3:])
                             .format({'Pred Diff': '{:.4f}'})
                             , use_container_width=True
                )
            else:
                st.info("No stored predictions found.")

        # st.header("Save Predictions")
        if st.button("Save Recent Predictions"):
            collection = setup_mongodb_connection()
            save_message = save_recent_predictions(collection, recent_preds, preprocessor)
            st.success(save_message)

if __name__ == "__main__":
    collection = setup_mongodb_connection()
    main()
    