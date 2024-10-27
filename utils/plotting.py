import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from utils.data_processing import reverse_normalize
import matplotlib.dates as mdates


def display_performance_metrics(performance_metrics):
    # Convert the metrics dictionary to a DataFrame for better display
    metrics_df = pd.DataFrame.from_dict(performance_metrics, orient='index', columns=['Value'])
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={'index': 'Metric'}, inplace=True)

    st.dataframe(metrics_df)

def display_trade_log(trade_log_df):
    trade_log_df['Entry Date'] = trade_log_df['Entry Date'].dt.date
    trade_log_df['Exit Date'] = trade_log_df['Exit Date'].dt.date
    st.dataframe(trade_log_df.round(2))

# def plot_equity_curve(balance_history_df):
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.lineplot(data=balance_history_df, x='Date', y='Balance', ax=ax)
#     ax.set_title('Equity Curve')
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Account Balance ($)')
#     ax.grid(True)
#     st.pyplot(fig)

def plot_equity_curve(balance_history_df):

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

def plot_model_results_with_trades(test_df_orig, predictions_df_orig, trade_log_df_orig, preprocessor):
    """
    Plots the actual 'Auc Price', predicted 'Auc Price', and buy/sell signals from the trade log.

    Parameters:
    - test_df: DataFrame containing the actual test data with 'Auc Price' column.
    - predictions_df: DataFrame containing the predicted data with 'Auc Price' column.
    - trade_log_df: DataFrame containing the trade log with 'Entry Date', 'Exit Date', 'Signal', etc.

    Returns:
    - None (plots the figure)
    """

    plt.figure(figsize=(12, 6))

    fig, ax = plt.subplots(figsize=(12, 6))

    test_df = test_df_orig.copy()
    predictions_df = predictions_df_orig.copy()
    trade_log_df = trade_log_df_orig.copy()
    # Plot the price data
    predictions_df = reverse_normalize(predictions_df, preprocessor.train_mean['Auc Price'], preprocessor.train_std['Auc Price'])
    test_df = reverse_normalize(test_df, preprocessor.train_mean['Auc Price'], preprocessor.train_std['Auc Price'])

    # predictions_df['Auc Price'] = (predictions_df['Auc Price'] * train_std) + train_mean≥÷

    # trade_log_df['Entry Price'] = (trade_log_df['Entry Price'] * preprocessor.train_std['Auc Price']) + preprocessor.train_mean['Auc Price']
    # trade_log_df['Exit Price'] = (trade_log_df['Exit Price'] * preprocessor.train_std['Auc Price']) + preprocessor.train_mean['Auc Price']

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
        return_pct = trade['Return (%)']

        # Ensure dates are in the index
        if entry_date not in test_df.index or exit_date not in test_df.index:
            continue

        # Determine if the trade was successful
        is_successful = return_pct > 0
        success_color = 'limegreen' if is_successful else 'red'
        
        if signal == 'Buy':
            # Entry marker
            ax.scatter(entry_date, entry_price, color='green', marker='^', s=100, label='Buy Signal' if idx == 0 else "")
            ax.text(entry_date, entry_price, "Buy", fontsize=8, verticalalignment='bottom', color='green')
        elif signal == 'Sell':
            # Entry marker
            ax.scatter(entry_date, entry_price, color='red', marker='v', s=100, label='Sell Signal' if idx == 0 else "")
            ax.text(entry_date, entry_price, "Sell", fontsize=8, verticalalignment='top', color='red')

        # Draw line from entry to exit
        # ax.plot([entry_date, exit_date], [entry_price, exit_price], linestyle='--', color=success_color, alpha=0.5)
        
        # Add return percentage annotation
        mid_date = entry_date + (exit_date - entry_date) / 2
        mid_price = (entry_price + exit_price) / 2
        ax.text(mid_date, mid_price, f"{return_pct:.2f}%", fontsize=8, 
                color=success_color, ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor=success_color, alpha=0.7, boxstyle='round,pad=0.3'))

    ax.set_title('Auc Price with Trades')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    # ax.legend()
    # Update legend
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([plt.Line2D([0], [0], color='limegreen', lw=2, linestyle='--'),
                    plt.Line2D([0], [0], color='red', lw=2, linestyle='--')])
    labels.extend(['Successful Trade', 'Unsuccessful Trade'])
    ax.legend(handles, labels)

    ax.grid(True)
    fig.autofmt_xdate()
    st.pyplot(fig)

def plot_recent_predictions(recent_preds_orig, trend, test_df_orig, preprocessor):

    plt.figure(figsize=(10, 4))
    fig, ax = plt.subplots(figsize=(10, 4))

    # test_df = test_df.iloc[-10:]
    recent_preds = recent_preds_orig.copy()
    test_df = test_df_orig.copy()
    test_df = reverse_normalize(test_df, preprocessor.train_mean['Auc Price'], preprocessor.train_std['Auc Price'])
    plot_df = test_df.copy().tail(90)

    ax.plot(plot_df.index, plot_df['Auc Price'], label='Auc Price', color='blue')

    pred_diff = np.mean(recent_preds.iloc[1:]['Auc Price'].values) - recent_preds.iloc[1]['Auc Price']

    prediction_price = recent_preds.iloc[0]['Auc Price']
    prediction_price = (prediction_price * preprocessor.train_std['Auc Price']) + preprocessor.train_mean['Auc Price'] 

    print("RECENT PREDICTIONS")
    print(recent_preds['Auc Price'])

    recent_preds['Auc Price'] = (recent_preds['Auc Price'] * preprocessor.train_std['Auc Price']) + preprocessor.train_mean['Auc Price']
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
