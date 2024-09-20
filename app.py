from IPython.display import display
import IPython
import pandas as pd
from pandas import Series
import numpy as np
from random import randrange

from matplotlib import pyplot
import matplotlib.pyplot as plt
# from statsmodels.tsa.seasonal import seasonal_decompose

# np.float_ = np.float64
# from prophet import Prophet

import math
from typing import Any, List, Dict, AnyStr, Optional
from pathlib import Path
from glob import glob

from utils.dataset import MarketData, DataPreprocessor, Plotting
from utils.windowgenerator import WindowGenerator, compile_and_fit

import tensorflow as tf
import talib



import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure that matplotlib plots display correctly in Streamlit
from matplotlib.backends.backend_agg import RendererAgg
import datetime
# _lock = RendererAgg.lock
# Placeholder for your backtesting function

cot_df, auction_df, eua_df, ta_df, fundamentals_df = MarketData.latest(Path('data'))
cot_df = cot_df.set_index('Date').resample('W', origin='end').mean().reset_index()
auction_df = auction_df.set_index('Date').resample('D').mean().reset_index()


auction_df = auction_df[7:]
auction_df.loc[:, 'Premium/discount-settle'] = auction_df['Premium/discount-settle'].ffill()
auction_df.loc[:, ['Auc Price', 'Median Price', 'Cover Ratio', 'Spot Value',
'Auction Spot Diff', 'Median Spot Diff', 'Premium/discount-settle']] = auction_df[['Auc Price', 'Median Price', 'Cover Ratio', 'Spot Value', 
                                                                                          'Auction Spot Diff', 'Median Spot Diff', 'Premium/discount-settle']].ffill()

merged_df = DataPreprocessor.engineer_auction_features(auction_df)
# Define features and labels
FEATURES = merged_df.columns.tolist()
LABEL_COLS = ['Auc Price']

preprocessor = DataPreprocessor(features=FEATURES, label_columns=LABEL_COLS, input_width=7, label_width=7, shift=1)
train_df, test_df, val_df = preprocessor.train_test_data(merged_df)
train_df, test_df, val_df = preprocessor.normalize(train_df, test_df, val_df)
num_features = len(test_df.columns)

OUT_STEPS = 7
INPUT_STEPS = 7
multi_window = WindowGenerator(input_width=INPUT_STEPS,
                               train_df=train_df, val_df=val_df, test_df=test_df,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = preprocessor.compile_and_fit(multi_conv_model, multi_window, use_early_stopping=True, max_epochs=50)


def generate_predictions(model, test_df, input_width, out_steps):
    features = test_df.columns
    num_features = len(features)
    predictions = []

    for idx, i in enumerate(range(input_width, len(test_df) - out_steps + 1, out_steps)):
        try:
            inputs = test_df[i - input_width:i].values
            inputs_reshaped = inputs.reshape((1, input_width, num_features))
            preds = model.predict(inputs_reshaped)
            predictions.append(preds[0])
        except Exception as e:
            print(f"Prediction error at index {i}: {e}")
            break

    # Create DataFrame for predictions
    predictions = np.concatenate(predictions, axis=0)
    pred_indices = test_df.index[input_width:input_width + len(predictions)]
    predictions_df = pd.DataFrame(predictions, columns=features, index=pred_indices)

    return predictions_df


def check_gradient(values):
        gradient = np.gradient(values)
        return 'positive' if np.all(gradient > 0) else 'negative'

def generate_recent_predictions(model, test_df, input_width, out_steps):
    features = test_df.columns
    num_features = len(features)
    inputs = test_df[-input_width:].values
    inputs_reshaped = inputs.reshape((1, input_width, num_features))
    preds = model.predict(inputs_reshaped)
    
    predictions = []
    predictions.append(preds[0])
    predictions = np.concatenate(predictions, axis=0)

    start_date = test_df.index[-1] + datetime.timedelta(days=1)
    date_range = pd.date_range(start = start_date, periods = out_steps)
    recent_preds = pd.DataFrame(predictions, index=date_range, columns=test_df.columns)
    
    trend = check_gradient(recent_preds['Auc Price'])
    recent_preds = pd.concat([test_df.iloc[[-1]], recent_preds])
    # pred_diff = np.mean(recent_preds.iloc[1:]['Auc Price'].values) - recent_preds.iloc[1]['Auc Price']

    # if pred_diff > 0:
    #     plt.text(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], "Buy", fontsize=8, verticalalignment='top')
    #     plt.plot(recent_preds.index, recent_preds['Auc Price'], color='green', marker='x', label="recent_prediction2", alpha=0.4)
    # else: 
    #     plt.text(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], "Sell", fontsize=8, verticalalignment='top')
    #     plt.plot(recent_preds.index, recent_preds['Auc Price'], color='red', marker='x', label="recent_prediction2", alpha=0.4)

    return recent_preds, trend


predictions_df = generate_predictions(multi_conv_model, test_df, INPUT_STEPS, OUT_STEPS)
recent_preds, trend = generate_recent_predictions(multi_conv_model, test_df, INPUT_STEPS, OUT_STEPS)

# from your_module import backtest_model_with_metrics
def backtest_model(model, test_df, input_width, out_steps, initial_balance, take_profit, stop_loss):
    """
    Backtesting function for the given model and test data.

    Parameters:
    - model: The trained prediction model.
    - test_df: DataFrame containing test data with 'Auc Price' column.
    - input_width: The number of past days used for making predictions.
    - out_steps: The number of future days the model predicts.
    - initial_balance: Initial amount of money in the account.
    - take_profit: Profit percentage at which to close the trade.
    - stop_loss: Loss percentage at which to close the trade.

    Returns:
    - trade_log_df: DataFrame containing the log of trades executed.
    - total_return: Total return from the backtesting.
    - balance_history: List containing the balance after each trade.
    """
    features = test_df.columns
    num_features = len(features)
    predictions = []

    # Initialize balance and trade log
    balance = initial_balance
    balance_history = [balance]
    trade_log = []
    position_size = initial_balance  # Using full balance per trade for simplicity

    # Generate predictions in steps of 'out_steps'
    for idx, i in enumerate(range(input_width, len(test_df) - out_steps + 1, out_steps)):
        try:
            inputs = test_df[i - input_width:i].values
            inputs_reshaped = inputs.reshape((1, input_width, num_features))
            preds = model.predict(inputs_reshaped)
            predictions.append(preds[0])
        except Exception as e:
            print(f"Prediction error at index {i}: {e}")
            break

    # Create DataFrame for predictions
    predictions = np.concatenate(predictions, axis=0)
    pred_indices = test_df.index[input_width:input_width + len(predictions)]
    predictions_df = pd.DataFrame(predictions, columns=features, index=pred_indices)

    # Simulate trades based on predictions
    for idx, i in enumerate(range(input_width, len(test_df) - out_steps + 1, out_steps)):
        entry_index = predictions_df.index[idx * out_steps]
        entry_price = test_df.loc[entry_index, 'Auc Price']
        prev_index = entry_index - pd.Timedelta(days=1)
        prev_price = test_df.loc[prev_index, 'Auc Price'] if prev_index in test_df.index else entry_price

        # Determine signal based on predicted mean price
        pred_mean = predictions_df['Auc Price'][idx * out_steps:(idx + 1) * out_steps].mean()
        signal = 'Buy' if pred_mean > prev_price else 'Sell'

        trade_closed = False
        exit_price = None
        exit_date = None

        # Simulate trade over the next 'out_steps' days
        for offset in range(1, out_steps + 1):
            current_index = entry_index + pd.Timedelta(days=offset)
            if current_index not in test_df.index:
                continue
            current_price = test_df.loc[current_index, 'Auc Price']

            # Calculate return percentage
            if signal == 'Buy':
                return_pct = (current_price - entry_price) / entry_price
                if return_pct >= take_profit:
                    exit_price = entry_price * (1 + take_profit)
                    exit_date = current_index
                    trade_closed = True
                    break
                elif return_pct <= -stop_loss:
                    exit_price = entry_price * (1 - stop_loss)
                    exit_date = current_index
                    trade_closed = True
                    break
            else:  # Sell signal
                return_pct = (entry_price - current_price) / entry_price
                if return_pct >= take_profit:
                    exit_price = entry_price * (1 - take_profit)
                    exit_date = current_index
                    trade_closed = True
                    break
                elif return_pct <= -stop_loss:
                    exit_price = entry_price * (1 + stop_loss)
                    exit_date = current_index
                    trade_closed = True
                    break

        # If trade not closed, close at the last available price
        if not trade_closed:
            last_index = entry_index + pd.Timedelta(days=out_steps)
            while last_index not in test_df.index and last_index > entry_index:
                last_index -= pd.Timedelta(days=1)
            if last_index in test_df.index:
                exit_price = test_df.loc[last_index, 'Auc Price']
                exit_date = last_index
            else:
                # If exit price not found, skip the trade
                print(f"Exit price not found for trade starting on {entry_index}")
                continue

        # Calculate profit or loss
        if signal == 'Buy':
            profit_loss = (exit_price - entry_price) / entry_price * position_size
        else:  # Sell
            profit_loss = (entry_price - exit_price) / entry_price * position_size

        # Update balance and log the trade
        balance += profit_loss
        balance_history.append(balance)
        trade_log.append({
            'Entry Date': entry_index,
            'Exit Date': exit_date,
            'Signal': signal,
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Profit/Loss': profit_loss,
            'Balance': balance
        })

    # Convert trade log to DataFrame
    trade_log_df = pd.DataFrame(trade_log)

    # Calculate total return
    total_return = (balance - initial_balance) / initial_balance

    return trade_log_df, total_return, balance_history


def backtest_model_with_metrics(model, test_df, input_width, out_steps,initial_balance, take_profit, stop_loss,position_size_fraction=1,risk_free_rate=0.01):
    features = test_df.columns
    num_features = len(features)
    predictions = []

    # Initialize balance and trade log
    balance = initial_balance
    balance_history = []
    trade_log = []

    # For performance metrics
    returns = []
    equity_curve = [initial_balance]

    # Generate predictions in steps of 'out_steps'
    for idx, i in enumerate(range(input_width, len(test_df) - out_steps + 1, out_steps)):
        try:
            inputs = test_df[i - input_width:i].values
            inputs_reshaped = inputs.reshape((1, input_width, num_features))
            preds = model.predict(inputs_reshaped)
            predictions.append(preds[0])
        except Exception as e:
            print(f"Prediction error at index {i}: {e}")
            break

    # Create DataFrame for predictions
    predictions = np.concatenate(predictions, axis=0)
    pred_indices = test_df.index[input_width:input_width + len(predictions)]
    predictions_df = pd.DataFrame(
        predictions, columns=features, index=pred_indices)

    # Simulate trades based on predictions
    for idx, i in enumerate(range(input_width, len(test_df) - out_steps + 1, out_steps)):
        entry_index = predictions_df.index[idx * out_steps]
        entry_price = test_df.loc[entry_index, 'Auc Price']
        prev_index = entry_index - pd.Timedelta(days=1)
        prev_price = test_df.loc[prev_index,
                                 'Auc Price'] if prev_index in test_df.index else entry_price

        # Determine signal based on predicted mean price
        pred_mean = predictions_df['Auc Price'][idx *
                                                out_steps:(idx + 1) * out_steps].mean()
        signal = 'Buy' if pred_mean > prev_price else 'Sell'

        trade_closed = False
        exit_price = None
        exit_date = None

        # Position sizing
        position_size = balance * position_size_fraction

        # Simulate trade over the next 'out_steps' days
        for offset in range(1, out_steps + 1):
            current_index = entry_index + pd.Timedelta(days=offset)
            if current_index not in test_df.index:
                continue
            current_price = test_df.loc[current_index, 'Auc Price']

            # Calculate return percentage
            if signal == 'Buy':
                return_pct = (current_price - entry_price) / entry_price
                if return_pct >= take_profit:
                    exit_price = entry_price * (1 + take_profit)
                    exit_date = current_index
                    trade_closed = True
                    break
                elif return_pct <= -stop_loss:
                    exit_price = entry_price * (1 - stop_loss)
                    exit_date = current_index
                    trade_closed = True
                    break
            else:  # Sell signal
                return_pct = (entry_price - current_price) / entry_price
                if return_pct >= take_profit:
                    exit_price = entry_price * (1 - take_profit)
                    exit_date = current_index
                    trade_closed = True
                    break
                elif return_pct <= -stop_loss:
                    exit_price = entry_price * (1 + stop_loss)
                    exit_date = current_index
                    trade_closed = True
                    break

        # If trade not closed, close at the last available price
        if not trade_closed:
            last_index = entry_index + pd.Timedelta(days=out_steps)
            while last_index not in test_df.index and last_index > entry_index:
                last_index -= pd.Timedelta(days=1)
            if last_index in test_df.index:
                exit_price = test_df.loc[last_index, 'Auc Price']
                exit_date = last_index
            else:
                # If exit price not found, skip the trade
                print(f"Exit price not found for trade starting on {
                      entry_index}")
                continue

        # Calculate profit or loss
        if signal == 'Buy':
            profit_loss = (exit_price - entry_price) / \
                entry_price * position_size
        else:  # Sell
            profit_loss = (entry_price - exit_price) / \
                entry_price * position_size

        # Update balance and log the trade
        balance += profit_loss
        # Return percentage for this trade
        returns.append(profit_loss / position_size)
        equity_curve.append(balance)
        balance_history.append({'Date': exit_date, 'Balance': balance})

        trade_log.append({
            'Entry Date': entry_index,
            'Exit Date': exit_date,
            'Signal': signal,
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Profit/Loss': profit_loss,
            'Return (%)': (profit_loss / position_size) * 100,
            'Balance': balance
        })

    # Convert trade log and balance history to DataFrames
    trade_log_df = pd.DataFrame(trade_log)
    balance_history_df = pd.DataFrame(balance_history).set_index('Date')

    # Calculate performance metrics
    total_return = (balance - initial_balance) / initial_balance
    num_years = (test_df.index[-1] - test_df.index[input_width]).days / 365.25
    CAGR = (balance / initial_balance) ** (1 /
                                           num_years) - 1 if num_years > 0 else 0

    # Maximum Drawdown
    equity_series = pd.Series(equity_curve)
    cumulative_returns = equity_series / equity_series.cummax() - 1
    max_drawdown = cumulative_returns.min()

    # Sharpe Ratio
    excess_returns = np.array(returns) - (risk_free_rate / 252)
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * \
        np.sqrt(252) if np.std(excess_returns) != 0 else np.nan

    # Sortino Ratio
    negative_returns = excess_returns[excess_returns < 0]
    downside_std = np.std(negative_returns)
    sortino_ratio = np.mean(excess_returns) / downside_std * \
        np.sqrt(252) if downside_std != 0 else np.nan

    # Win Rate
    num_trades = len(trade_log_df)
    num_wins = len(trade_log_df[trade_log_df['Profit/Loss'] > 0])
    win_rate = num_wins / num_trades if num_trades > 0 else np.nan

    # Average Profit per Trade
    avg_profit = trade_log_df['Profit/Loss'].mean()

    # Profit Factor
    gross_profit = trade_log_df[trade_log_df['Profit/Loss']
                                > 0]['Profit/Loss'].sum()
    gross_loss = - \
        trade_log_df[trade_log_df['Profit/Loss'] < 0]['Profit/Loss'].sum()
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan

    performance_metrics = {
        'Total Return (%)': total_return * 100,
        'CAGR (%)': CAGR * 100,
        'Maximum Drawdown (%)': max_drawdown * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Win Rate (%)': win_rate * 100,
        'Average Profit per Trade': avg_profit,
        'Profit Factor': profit_factor
    }

    return trade_log_df, performance_metrics, balance_history_df

def main():
    st.set_page_config(page_title="Trading Strategy Backtesting Dashboard", layout="wide")
    st.title("Trading Strategy Backtesting Dashboard")

    # Sidebar inputs
    st.sidebar.header("Backtesting Parameters")

    initial_balance = st.sidebar.number_input("Initial Balance ($)", value=10000.0, min_value=0.0, step=1000.0)

    take_profit = st.sidebar.slider("Take Profit (%)", min_value=0.0, max_value=100.0, value=8.0, step=0.1)
    stop_loss = st.sidebar.slider("Stop Loss (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)

    position_size_fraction = st.sidebar.slider(
        "Position Size Fraction (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0
    )

    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (%)", value=1.0, min_value=0.0, max_value=10.0, step=0.1
    )

    st.sidebar.markdown("---")

    st.sidebar.header("Model and Data Inputs")

    # Upload model
    # model_file = st.sidebar.text_input(
    #     "Placeholder for the other text input widget",
    #     "Model File Name",
    #     key="name",
    # )
    # if model_file is not None:
    #     # Placeholder for loading the model
    #     # model = load_model_function(model_file)
    #     st.sidebar.success("Model uploaded successfully.")
    # else:
    #     st.sidebar.warning("Please upload your trained model.")

    # Upload test data
    # data_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])
    # if data_file is not None:
    #     test_df = pd.read_csv(data_file, parse_dates=True, index_col=0)
    #     st.sidebar.success("Test data uploaded successfully.")
    # else:
    #     st.sidebar.warning("Please upload your test data.")

    # test_df = pd.read_csv('data/test_data.csv').set_index('Date')
    # Input width and out steps
    input_width = st.sidebar.number_input("Input Width (Days)", min_value=1, value=7, step=1)
    out_steps = st.sidebar.number_input("Out Steps (Days)", min_value=1, value=7, step=1)

    # Run backtesting when button is clicked
    if st.sidebar.button("Run Backtest"):
        

        with st.spinner("Running backtest..."):

            # model = tf.keras.models.load_model(f'models/{model_file}.keras')
            # For example:
            trade_log_df, performance_metrics, balance_history_df = backtest_model_with_metrics(
                multi_conv_model,
                test_df,
                input_width,
                out_steps,
                initial_balance,
                take_profit / 100.0,
                stop_loss / 100.0,
                position_size_fraction=position_size_fraction / 100.0,
                risk_free_rate=risk_free_rate / 100.0
            )

            # trade_log_df2, performance_metrics2, balance_history_df2  = backtest_model(multi_conv_model,
            #     test_df,
            #     input_width,
            #     out_steps,
            #     initial_balance,
            #     take_profit / 100.0,
            #     stop_loss / 100.0
            #     )
            
            # For demonstration purposes, we'll create some dummy data
            # Remove this section when integrating with your actual functions
            # trade_log_df, performance_metrics, balance_history_df = generate_dummy_data()

        # Display outputs
        # st.header("Performance Metrics")
        # display_performance_metrics(performance_metrics)
        
        # st.header("Trade Log")
        # display_trade_log(trade_log_df)

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
            plot_model_results_with_trades(test_df, predictions_df, trade_log_df)
            # st.subheader("Drawdown Curve")
            # plot_drawdown_curve(balance_history_df)

        st.header("Recent Predictions")
        plot_recent_predictions(recent_preds, trend, test_df)

def display_performance_metrics(performance_metrics):
    # Convert the metrics dictionary to a DataFrame for better display
    metrics_df = pd.DataFrame.from_dict(performance_metrics, orient='index', columns=['Value'])
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={'index': 'Metric'}, inplace=True)

    st.dataframe(metrics_df)

def display_trade_log(trade_log_df):
    st.dataframe(trade_log_df)

# def plot_equity_curve(balance_history_df):
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.lineplot(data=balance_history_df, x='Date', y='Balance', ax=ax)
#     ax.set_title('Equity Curve')
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Account Balance ($)')
#     ax.grid(True)
#     st.pyplot(fig)

def plot_equity_curve(balance_history_df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.dates as mdates
    import pandas as pd
    import numpy as np

    # Ensure the 'Date' column is of datetime type
    balance_history_df = balance_history_df.reset_index()
    balance_history_df['Date'] = pd.to_datetime(balance_history_df['Date'])

    # Calculate cumulative returns and drawdowns
    balance_history_df['Returns'] = balance_history_df['Balance'].pct_change().fillna(0)
    balance_history_df['Cumulative Return'] = (1 + balance_history_df['Returns']).cumprod() - 1
    balance_history_df['Cumulative Max'] = balance_history_df['Balance'].cummax()
    balance_history_df['Drawdown'] = balance_history_df['Balance'] / balance_history_df['Cumulative Max'] - 1

    # Set the style
    sns.set_style('whitegrid')
    sns.set_palette('tab10')

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot the equity curve
    sns.lineplot(data=balance_history_df, x='Date', y='Balance', ax=ax1, color='blue', label='Equity Curve')

    # Shade drawdown areas
    ax1.fill_between(balance_history_df['Date'], balance_history_df['Balance'], balance_history_df['Cumulative Max'],
                     where=balance_history_df['Balance'] < balance_history_df['Cumulative Max'],
                     interpolate=True, color='red', alpha=0.3, label='Drawdown')

    # Plot cumulative returns on a secondary y-axis
    ax2 = ax1.twinx()
    sns.lineplot(data=balance_history_df, x='Date', y='Cumulative Return', ax=ax2, color='green', label='Cumulative Return (%)')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax2.set_ylabel('Cumulative Return (%)')

    # Format x-axis with date labels
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    # Set labels and title
    ax1.set_title('Equity Curve with Drawdowns and Cumulative Returns')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Account Balance ($)')

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1, labels_1, loc='upper left')

    # Add grid
    ax1.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)

def plot_model_results_with_trades(test_df, predictions_df, trade_log_df):
    """
    Plots the actual 'Auc Price', predicted 'Auc Price', and buy/sell signals from the trade log.

    Parameters:
    - test_df: DataFrame containing the actual test data with 'Auc Price' column.
    - predictions_df: DataFrame containing the predicted data with 'Auc Price' column.
    - trade_log_df: DataFrame containing the trade log with 'Entry Date', 'Exit Date', 'Signal', etc.

    Returns:
    - None (plots the figure)
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.figure(figsize=(12, 6))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the price data


    test_df.index = pd.to_datetime(test_df.index)
    predictions_df.index = pd.to_datetime(predictions_df.index)
    trade_log_df['Entry Date'] = pd.to_datetime(trade_log_df['Entry Date'])
    trade_log_df['Exit Date'] = pd.to_datetime(trade_log_df['Exit Date'])

    # Plot actual 'Auc Price'

    ax.plot(test_df.index, test_df['Auc Price'], label='Auc Price', color='blue')

    # Plot predicted 'Auc Price'
    ax.plot(predictions_df.index, predictions_df['Auc Price'], label='Predicted Auc Price', linestyle='dashed', color='orange')

    # Plot buy and sell signals from trade_log_df
    for idx, trade in trade_log_df.iterrows():
        entry_date = trade['Entry Date']
        exit_date = trade['Exit Date']
        signal = trade['Signal']
        entry_price = trade['Entry Price']
        exit_price = trade['Exit Price']

        # Ensure dates are in the index
        if entry_date not in test_df.index or exit_date not in test_df.index:
            continue

        if signal == 'Buy':
            # Entry marker
            ax.scatter(entry_date, entry_price, color='green', marker='^', s=100, label='Buy Signal' if idx == 0 else "")
            ax.text(entry_date, entry_price, "Buy", fontsize=8, verticalalignment='bottom', color='green')
        elif signal == 'Sell':
            # Entry marker
            ax.scatter(entry_date, entry_price, color='red', marker='v', s=100, label='Sell Signal' if idx == 0 else "")
            ax.text(entry_date, entry_price, "Sell", fontsize=8, verticalalignment='top', color='red')

        # Draw line from entry to exit
        # ax.plot([entry_date, exit_date], [entry_price, exit_price], linestyle='--', color='gray')

    ax.set_title('Auc Price with Trades')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    st.pyplot(fig)

def plot_recent_predictions(recent_preds, trend, test_df):

    plt.figure(figsize=(10, 4))
    fig, ax = plt.subplots(figsize=(10, 4))

    # test_df = test_df.iloc[-10:]
    ax.plot(test_df.index, test_df['Auc Price'], label='Auc Price', color='blue')

    pred_diff = np.mean(recent_preds.iloc[1:]['Auc Price'].values) - recent_preds.iloc[1]['Auc Price']

    grad = round(np.mean(np.gradient(recent_preds['Auc Price'].values[1:])), 4)
    if pred_diff > 0:
        ax.text(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], grad, fontsize=5, verticalalignment='bottom')
        ax.scatter(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], color='green', marker='^', s=100)
        ax.text(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], "Buy", fontsize=5, verticalalignment='top')
        ax.plot(recent_preds.index, recent_preds['Auc Price'], color='green', label="Prediction", alpha=0.4, linestyle='dashed')
    else: 
        ax.text(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], grad, fontsize=5, verticalalignment='bottom')
        ax.scatter(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], color='red', marker='v', s=100)
        ax.text(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], "Sell", fontsize=5, verticalalignment='top')
        ax.plot(recent_preds.index, recent_preds['Auc Price'], color='red', label="Prediction", alpha=0.4, linestyle='dashed')

    ax.set_title('Recent Predictions', fontsize=6)
    ax.set_xlabel('Date', fontsize=5)
    ax.set_ylabel('Price', fontsize=5)
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    st.pyplot(fig)
    

def plot_drawdown_curve(balance_history_df):
    # Calculate Drawdown
    balance_history_df['Cumulative Max'] = balance_history_df['Balance'].cummax()
    balance_history_df['Drawdown'] = balance_history_df['Balance'] / balance_history_df['Cumulative Max'] - 1

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=balance_history_df, x='Date', y='Drawdown', ax=ax, color='red')
    ax.set_title('Drawdown Curve')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True)
    st.pyplot(fig)

def generate_dummy_data():
    # This function generates dummy data for demonstration purposes
    dates = pd.date_range(start='2022-01-01', periods=50, freq='D')
    balance = np.linspace(10000, 12000, num=50) + np.random.normal(0, 100, size=50)
    balance_history_df = pd.DataFrame({'Date': dates, 'Balance': balance})

    trade_log_df = pd.DataFrame({
        'Entry Date': dates[::5],
        'Exit Date': dates[4::5],
        'Signal': ['Buy', 'Sell'] * 5,
        'Entry Price': np.random.uniform(100, 200, size=10),
        'Exit Price': np.random.uniform(100, 200, size=10),
        'Profit/Loss': np.random.uniform(-500, 500, size=10),
        'Return (%)': np.random.uniform(-5, 5, size=10),
        'Balance': np.linspace(10000, 12000, num=10)
    })

    performance_metrics = {
        'Total Return (%)': 15.0,
        'CAGR (%)': 14.5,
        'Maximum Drawdown (%)': -5.2,
        'Sharpe Ratio': 1.2,
        'Sortino Ratio': 1.8,
        'Win Rate (%)': 60.0,
        'Average Profit per Trade': 150.0,
        'Profit Factor': 1.5
    }

    return trade_log_df, performance_metrics, balance_history_df

if __name__ == "__main__":
    main()