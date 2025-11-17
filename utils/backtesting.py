import pandas as pd
import numpy as np
from typing import Dict, Any
from utils.data_processing import reverse_normalize

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
            inputs = test_df[i - input_width:i].values.astype(np.float32)
            inputs_reshaped = np.array(inputs).reshape((1, input_width, num_features)).astype(np.float32)
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
        print(f"[DEBUG] entry_index: {entry_index}, entry_price: {entry_price}, type: {type(entry_price)}")
        if isinstance(entry_price, pd.Series):
            print(f"[DEBUG] entry_price is Series, values: {entry_price.values}")
            entry_price = entry_price.iloc[0]
        prev_index = entry_index - pd.Timedelta(days=1)
        prev_price = test_df.loc[prev_index, 'Auc Price'] if prev_index in test_df.index else entry_price
        print(f"[DEBUG] prev_index: {prev_index}, prev_price: {prev_price}, type: {type(prev_price)}")
        if isinstance(prev_price, pd.Series):
            print(f"[DEBUG] prev_price is Series, values: {prev_price.values}")
            prev_price = prev_price.iloc[0]

        # Determine signal based on predicted mean price
        pred_mean = predictions_df['Auc Price'][idx * out_steps:(idx + 1) * out_steps].mean()
        print(f"[DEBUG] pred_mean: {pred_mean}, type: {type(pred_mean)}")
        if isinstance(pred_mean, pd.Series):
            print(f"[DEBUG] pred_mean is Series, values: {pred_mean.values}")
            pred_mean = pred_mean.iloc[0]
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
            print(f"[DEBUG] current_index: {current_index}, current_price: {current_price}, type: {type(current_price)}")
            if isinstance(current_price, pd.Series):
                print(f"[DEBUG] current_price is Series, values: {current_price.values}")
                current_price = current_price.iloc[0]

            # Calculate return percentage
            if signal == 'Buy':
                return_pct = (current_price - entry_price) / entry_price
                print(f"[DEBUG] return_pct (Buy): {return_pct}, type: {type(return_pct)}")
                if isinstance(return_pct, pd.Series):
                    return_pct = return_pct.iloc[0]
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
                print(f"[DEBUG] return_pct (Sell): {return_pct}, type: {type(return_pct)}")
                if isinstance(return_pct, pd.Series):
                    return_pct = return_pct.iloc[0]
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
        if not bool(trade_closed):
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
            'Entry Date': pd.to_datetime(entry_index),
            'Exit Date': pd.to_datetime(exit_date),
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

def backtest_model_with_metrics(model, test_df, input_width, out_steps, initial_balance, take_profit, stop_loss, position_size_fraction=1, risk_free_rate=0.01, preprocessor=None):
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
            inputs = test_df[i - input_width:i].values.astype(np.float32)
            inputs_reshaped = np.array(inputs).reshape((1, input_width, num_features)).astype(np.float32)
            preds = model.predict(inputs_reshaped)
            predictions.append(preds[0])
        except Exception as e:
            print(f"Prediction error at index {i}: {e}")
            break

    # Create DataFrame for predictions
    predictions = np.concatenate(predictions, axis=0)
    pred_indices = test_df.index[input_width:input_width + len(predictions)]
    predictions_df = pd.DataFrame(predictions, columns=features, index=pred_indices)

    # Reverse normalize predictions and test data
    predictions_df = reverse_normalize(predictions_df, preprocessor.train_mean['Auc Price'], preprocessor.train_std['Auc Price'])
    test_df_denormalized = reverse_normalize(test_df.copy(), preprocessor.train_mean['Auc Price'], preprocessor.train_std['Auc Price'])

    print(f"PREDICTION DATAFRAME")
    print(predictions_df)
    
    # Simulate trades based on predictions
    for idx, i in enumerate(range(input_width, len(test_df) - out_steps + 1, out_steps)):
        entry_index = predictions_df.index[idx * out_steps]
        entry_price = test_df_denormalized.loc[entry_index, 'Auc Price']
        if isinstance(entry_price, pd.Series):
            entry_price = entry_price.iloc[0]
        prev_index = entry_index - pd.Timedelta(days=1)
        prev_price = test_df_denormalized.loc[prev_index, 'Auc Price'] if prev_index in test_df_denormalized.index else entry_price
        if isinstance(prev_price, pd.Series):
            prev_price = prev_price.iloc[0]

        print(f"PREV PRICE: {prev_price}")
        # Determine signal based on predicted mean price
        pred_mean = predictions_df['Auc Price'][idx * out_steps:(idx + 1) * out_steps].mean()
        if isinstance(pred_mean, pd.Series):
            pred_mean = pred_mean.iloc[0]
        signal = 'Buy' if pred_mean > prev_price else 'Sell'

        trade_closed = False
        exit_price = None
        exit_date = None
        # Position sizing
        position_size = balance * position_size_fraction

        # Simulate trade over the next 'out_steps' days
        for offset in range(1, out_steps + 1):
            current_index = entry_index + pd.Timedelta(days=offset)
            if current_index not in test_df_denormalized.index:
                continue
            current_price = test_df_denormalized.loc[current_index, 'Auc Price']
            if isinstance(current_price, pd.Series):
                current_price = current_price.iloc[0]

            # Calculate return percentage
            if signal == 'Buy':

                # print(f"Entry Price: {entry_price}, Current Price: {current_price}")
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
        if not bool(trade_closed):
            last_index = entry_index + pd.Timedelta(days=out_steps)
            while last_index not in test_df_denormalized.index and last_index > entry_index:
                last_index -= pd.Timedelta(days=1)
            if last_index in test_df_denormalized.index:
                exit_price = test_df_denormalized.loc[last_index, 'Auc Price']
                if isinstance(exit_price, pd.Series):
                    exit_price = exit_price.iloc[0]
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
        # Return percentage for this trade
        returns.append(profit_loss / position_size)
        equity_curve.append(balance)
        balance_history.append({'Date': exit_date, 'Balance': balance})

        trade_log.append({
            'Entry Date': pd.to_datetime(entry_index),
            'Exit Date': pd.to_datetime(exit_date),
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