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

from utils.dataset import (
    MarketData,
    DataPreprocessor,
    Plotting
)
from utils.windowgenerator import WindowGenerator, compile_and_fit

from utils.backtesting import (
    backtest_model_with_metrics,
    generate_dummy_data
)
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

merged_df = load_and_preprocess_data()
train_df, test_df, val_df, preprocessor = prepare_data(merged_df)
num_features = len(test_df.columns)
OUT_STEPS = 7

model = create_model(num_features, OUT_STEPS)
history = train_model(model, train_df, val_df, test_df, preprocessor)
predictions_df, recent_preds, trend = generate_model_predictions(model, test_df)

def main():
    st.set_page_config(page_title="Trading Strategy Backtesting Dashboard", layout="wide")
    st.title("Trading Strategy Backtesting Dashboard")

    st.sidebar.header("Backtesting Parameters")

    initial_balance = st.sidebar.number_input("Initial Balance ($)", value=10000.0, min_value=0.0, step=1000.0)
    take_profit = st.sidebar.slider("Take Profit (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
    stop_loss = st.sidebar.slider("Stop Loss (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
    position_size_fraction = st.sidebar.slider(
        "Position Size Fraction (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0
    )
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (%)", value=1.0, min_value=0.0, max_value=10.0, step=0.1
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Model and Data Inputs")
    input_width = st.sidebar.number_input("Input Width (Days)", min_value=1, value=7, step=1)
    out_steps = st.sidebar.number_input("Out Steps (Days)", min_value=1, value=7, step=1)

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
            # predictions_df_denormalized = reverse_normalize(predictions_df, preprocessor.train_mean['Auc Price'], preprocessor.train_std['Auc Price'])
            plot_model_results_with_trades(test_df, predictions_df, trade_log_df, preprocessor)

        st.header("Recent Predictions")
        plot_recent_predictions(recent_preds, trend, test_df, preprocessor)

if __name__ == "__main__":
    main()