from IPython.display import display
import pandas as pd
import numpy as np
from random import randrange
import random
import math
from typing import Any, List, Dict, AnyStr, Optional
from pathlib import Path
from glob import glob

import tensorflow as tf

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# from matplotlib.backends.backend_agg import RendererAgg
import datetime

from utils.dataset import MarketData, DataPreprocessor, Plotting
from utils.windowgenerator import WindowGenerator, compile_and_fit
from utils.backtesting import backtest_model_with_metrics, generate_dummy_data
from utils.plotting import (
    display_performance_metrics,
    display_trade_log,
    plot_equity_curve,
    plot_model_results_with_trades,
    plot_recent_predictions,
    plot_drawdown_curve
)
from utils.data_processing import reverse_normalize
from utils.prediction_utils import generate_predictions, generate_recent_predictions, check_gradient
import pymongo
from pymongo import MongoClient
from datetime import datetime


client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string
db = client['carbon_market_predictions']  # Replace with your database name
collection = db['recent_predictions']

def get_stored_predictions(collection):
    stored_predictions = list(collection.find().sort("date", -1).limit(10))  # Get the 10 most recent entries
    return stored_predictions

def setup_mongodb_connection():
    client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string
    db = client['carbon_market_predictions']  # Replace with your database name
    collection = db['recent_predictions']
    return collection

# def save_recent_predictions(collection, recent_preds):
#     prediction_date = recent_preds.index[0]
#     auc_price_predictions = recent_preds['Auc Price'].tolist()
    
#     document = {
#         'date': prediction_date,
#         'predictions': auc_price_predictions
#     }
    
#     collection.update_one({'date': prediction_date}, {'$set': document}, upsert=True)
#     return f"Predictions for {prediction_date} saved successfully."

# def save_recent_predictions(collection, recent_preds):
#     prediction_date = recent_preds.index[0]
#     auc_price_predictions = recent_preds['Auc Price'].tolist()
    
#     # Calculate the trade direction
#     pred_diff = np.mean(recent_preds.iloc[1:]['Auc Price'].values) - recent_preds.iloc[1]['Auc Price']
#     trade_direction = 'Buy' if pred_diff > 0 else 'Sell'
    
#     document = {
#         'date': prediction_date,
#         'predictions': auc_price_predictions,
#         'trade_direction': trade_direction,
#         'pred_diff': float(pred_diff)  # Convert to float for MongoDB storage
#     }
    
#     collection.update_one({'date': prediction_date}, {'$set': document}, upsert=True)
#     return f"Predictions for {prediction_date} saved successfully. Trade direction: {trade_direction}"

def save_recent_predictions(collection, recent_preds_orig, preprocessor):
    recent_preds = recent_preds_orig.copy()
    prediction_date = recent_preds.index[0]
    
    # Reverse normalize the Auc Price
    auc_price_predictions = (recent_preds['Auc Price'] * preprocessor.train_std['Auc Price']) + preprocessor.train_mean['Auc Price']
    
    # Calculate the trade direction using the reverse normalized values
    pred_diff = np.mean(auc_price_predictions.iloc[1:].values) - auc_price_predictions.iloc[1]
    trade_direction = 'Buy' if pred_diff > 0 else 'Sell'
    
    document = {
        'date': prediction_date,
        'predictions': auc_price_predictions.tolist(),
        'trade_direction': trade_direction,
        'pred_diff': float(pred_diff)
    }
    
    collection.update_one({'date': prediction_date}, {'$set': document}, upsert=True)
    return f"Predictions for {prediction_date} saved successfully. Trade direction: {trade_direction}"


def prepare_data(merged_df):
    FEATURES = merged_df.columns.tolist()
    LABEL_COLS = ['Auc Price']

    preprocessor = DataPreprocessor(features=FEATURES, label_columns=LABEL_COLS, input_width=7, label_width=7, shift=1)
    train_df, test_df, val_df = preprocessor.train_test_data(merged_df)
    train_df, test_df, val_df = preprocessor.normalize(train_df, test_df, val_df)
    return train_df, test_df, val_df, preprocessor

def create_model(num_features, out_steps):
    CONV_WIDTH = 3
    multi_conv_model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
        tf.keras.layers.Dense(out_steps*num_features,
                              kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([out_steps, num_features])
    ])
    return multi_conv_model

def train_model(model, train_df, val_df, test_df, preprocessor):
    OUT_STEPS = 7
    INPUT_STEPS = 7
    multi_window = WindowGenerator(input_width=INPUT_STEPS,
                                   train_df=train_df, val_df=val_df, test_df=test_df,
                                   label_width=OUT_STEPS,
                                   shift=OUT_STEPS)

    history = preprocessor.compile_and_fit(model, multi_window, use_early_stopping=True, max_epochs=40)
    return history

def generate_model_predictions(model, test_df):
    INPUT_STEPS = 7
    OUT_STEPS = 7
    predictions_df = generate_predictions(model, test_df, INPUT_STEPS, OUT_STEPS)
    recent_preds, trend = generate_recent_predictions(model, test_df, INPUT_STEPS, OUT_STEPS)
    return predictions_df, recent_preds, trend


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


# merged_df = load_and_preprocess_data()
# train_df, test_df, val_df, preprocessor = prepare_data(merged_df)
# num_features = len(test_df.columns)
# OUT_STEPS = 7

# model = create_model(num_features, OUT_STEPS)
# history = train_model(model, train_df, val_df, test_df, preprocessor)
# predictions_df, recent_preds, trend = generate_model_predictions(model, test_df)

def main():
    st.set_page_config(page_title="Trading Strategy Backtesting Dashboard", layout="wide")
    st.title("Trading Strategy Backtesting Dashboard")

    # Load data (this will be cached)
    merged_df = load_and_preprocess_data()

    # Add Re-Train Model button to sidebar

    # Train model (this will be cached unless the cache was just cleared)
    model, preprocessor, test_df, predictions_df, recent_preds, trend = prepare_data_and_train_model(merged_df)

    # Sidebar Inputs
    st.sidebar.header("Backtesting Parameters")
    initial_balance = st.sidebar.number_input("Initial Balance ($)", value=10000.0, min_value=0.0, step=1000.0)
    take_profit = st.sidebar.slider("Take Profit (%)", min_value=0.0, max_value=100.0, value=4.0, step=0.1)
    stop_loss = st.sidebar.slider("Stop Loss (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
    position_size_fraction = st.sidebar.slider("Position Size Fraction (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=1.0, min_value=0.0, max_value=10.0, step=0.1)

    st.sidebar.markdown("---")
    st.sidebar.header("Model and Data Inputs")
    input_width = st.sidebar.number_input("Input Width (Days)", min_value=1, value=7, step=1)
    out_steps = st.sidebar.number_input("Out Steps (Days)", min_value=1, value=7, step=1)

    if st.sidebar.button("Re-Train Model"):
        with st.spinner("Re-training model..."):
            # Clear the cache for prepare_data_and_train_model
            st.cache_resource.clear()
            st.success("Model cache cleared. The model will be retrained on the next run.")
        st.rerun()
        
    if st.sidebar.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            trade_log_df, performance_metrics, balance_history_df = backtest_model_with_metrics(
                model,
                test_df,
                input_width,
                out_steps,
                initial_balance,
                take_profit / 100.0,
                stop_loss / 100.0,
                position_size_fraction=position_size_fraction / 100.0,
                risk_free_rate=risk_free_rate / 100.0,
                preprocessor=preprocessor  # Pass the preprocessor here
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
            # predictions_df_denormalized = reverse_normalize(predictions_df, preprocessor.train_mean['Auc Price'], preprocessor.train_std['Auc Price'])
            plot_model_results_with_trades(test_df, predictions_df, trade_log_df, preprocessor)

        st.header("Recent Predictions")
        plot_recent_predictions(recent_preds, trend, test_df, preprocessor)

    collection = setup_mongodb_connection()
    st.header("Stored Predictions")
    with st.spinner("Loading stored predictions..."):
        stored_predictions = get_stored_predictions(collection)
        if stored_predictions:
            data = []
            for pred in stored_predictions:
                # Convert pred_diff to float, use 0.0 if conversion fails
                try:
                    pred_diff = float(pred['pred_diff'])
                except (ValueError, TypeError):
                    pred_diff = 0.0
                
                row = [pred['date'], pred['trade_direction'], pred_diff] + pred['predictions']
                data.append(row)
            
            columns = ['Date', 'Trade Direction', 'Pred Diff'] + [f'Day {i+1}' for i in range(len(stored_predictions[0]['predictions']))]
            df = pd.DataFrame(data, columns=columns)
            df['Date'] = pd.to_datetime(df['Date']).dt.date  # Convert to date for better display
            
            # Apply color coding to the Trade Direction column
            def color_trade_direction(val):
                color = 'green' if val == 'Buy' else 'red'
                return f'color: {color}'
            
            st.dataframe(df.style
                         .map(color_trade_direction, subset=['Trade Direction'])
                         .highlight_max(axis=1, subset=df.columns[3:])
                         .format({'Pred Diff': '{:.4f}'})
            )
        else:
            st.info("No stored predictions found.")

    st.header("Save Predictions")
    if st.button("Save Recent Predictions"):
        collection = setup_mongodb_connection()
        save_message = save_recent_predictions(collection, recent_preds, preprocessor)
        st.success(save_message)

    print(f"Recent Predictions: {recent_preds}")
if __name__ == "__main__":
    collection = setup_mongodb_connection()
    main()