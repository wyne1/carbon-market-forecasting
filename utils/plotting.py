import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from utils.data_processing import reverse_normalize
import matplotlib.dates as mdates
from utils.mongodb_utils import setup_mongodb_connection, get_stored_predictions

def plot_recent_predictions(recent_preds_orig, trend, test_df_orig, preprocessor):
    plt.figure(figsize=(10, 4))
    fig, ax = plt.subplots(figsize=(10, 4))

    recent_preds = recent_preds_orig.round(3).copy()
    test_df = test_df_orig.copy()
    test_df = reverse_normalize(test_df, preprocessor.train_mean['Auc Price'], preprocessor.train_std['Auc Price'])
    plot_df = test_df.copy().tail(60)

    # Plot historical data
    ax.plot(plot_df.index, plot_df['Auc Price'], label='Auc Price', color='tomato', marker='o', markersize=3)

    # Plot current prediction
    pred_diff = np.mean(recent_preds.iloc[1:]['Auc Price'].values) - recent_preds.iloc[1]['Auc Price']
    prediction_price = recent_preds.iloc[0]['Auc Price']
    prediction_price = (prediction_price * preprocessor.train_std['Auc Price']) + preprocessor.train_mean['Auc Price'] 
    recent_preds['Auc Price'] = (recent_preds['Auc Price'] * preprocessor.train_std['Auc Price']) + preprocessor.train_mean['Auc Price']
    
    # Plot stored predictions from MongoDB
    collection = setup_mongodb_connection()
    stored_predictions = get_stored_predictions(collection)[:-1]

    # Only plot the last 5 stored predictions
    if stored_predictions:
        last_5_predictions = stored_predictions[:5]
        for i, pred in enumerate(last_5_predictions):
            pred_dates = [pred['date']] + [pred['date'] + pd.Timedelta(days=j+1) for j in range(len(pred['predictions'])-1)]
            color = 'lightgreen' if pred['trade_direction'] == 'Buy' else 'lightcoral'

            colors = plt.cm.cool(np.linspace(0, 1, 20))
            ax.plot(pred_dates, pred['predictions'], 
                   color=colors[i % 20], 
                   alpha=0.8,
                   linestyle='dashed',
                   label=f"Stored Pred {pd.to_datetime(pred['date']).date()}")
            
            # Add marker for trade direction
            marker = '^' if pred['trade_direction'] == 'Buy' else 'v'
            ax.scatter(pred_dates[0], pred['predictions'][0], 
                      color=color, 
                      marker=marker, 
                      s=50,
                      alpha=0.3)

    # Plot current prediction on top
    grad = round(np.mean(np.gradient(recent_preds['Auc Price'].values[1:])), 4)
    if pred_diff > 0:
        ax.text(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], grad, fontsize=5, verticalalignment='bottom')
        ax.scatter(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], color='green', marker='^', s=100)
        ax.text(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], "Buy", fontsize=5, verticalalignment='top')
        ax.plot(recent_preds.index, recent_preds['Auc Price'], color='green', label="Current Prediction", alpha=0.8, linestyle='dashed')
    else: 
        ax.text(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], grad, fontsize=5, verticalalignment='bottom')
        ax.scatter(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], color='red', marker='v', s=100)
        ax.text(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], "Sell", fontsize=5, verticalalignment='top')
        ax.plot(recent_preds.index, recent_preds['Auc Price'], color='red', label="Current Prediction", alpha=0.8, linestyle='dashed')

    ax.set_title('Recent Predictions', fontsize=10)
    ax.set_xlabel('Date', fontsize=8)
    ax.set_ylabel('Price', fontsize=8)
    ax.xaxis.set_tick_params(labelsize=7, rotation=180)
    ax.yaxis.set_tick_params(labelsize=7)
    # ax.legend(fontsize=6)
    ax.grid(True)
    fig.autofmt_xdate()
    st.pyplot(fig)

def display_performance_metrics(performance_metrics):
    # Convert the metrics dictionary to a DataFrame for better display
    metrics_df = pd.DataFrame.from_dict(performance_metrics, orient='index', columns=['Value'])
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={'index': 'Metric'}, inplace=True)

    st.dataframe(metrics_df, width='stretch', hide_index=True)

def display_trade_log(trade_log_df):
    trade_log_df['Entry Date'] = trade_log_df['Entry Date'].dt.date
    trade_log_df['Exit Date'] = trade_log_df['Exit Date'].dt.date
    st.dataframe(trade_log_df.round(2), width='stretch')

def plot_ensemble_predictions(ensemble_predictions, test_df, preprocessor):
    plt.figure(figsize=(12, 6))
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    test_df = reverse_normalize(test_df.copy(), 
                              preprocessor.train_mean['Auc Price'], 
                              preprocessor.train_std['Auc Price'])
    plot_df = test_df.copy().tail(90)
    ax.plot(plot_df.index, plot_df['Auc Price'], 
            label='Actual Price', color='black', marker='o', markersize=3)
    
    # Create a DataFrame to store all predictions for averaging
    first_pred = ensemble_predictions[0][0]
    all_predictions = np.zeros((len(ensemble_predictions), len(first_pred)))
    
    # Plot predictions from each model and collect predictions for averaging
    colors = plt.cm.cool(np.linspace(0, 1, 50))
    for i, (preds, trend) in enumerate(ensemble_predictions):
        normalized_preds = (preds['Auc Price'] * preprocessor.train_std['Auc Price']) + preprocessor.train_mean['Auc Price']
        all_predictions[i] = normalized_preds
        ax.plot(preds.index, normalized_preds, 
                label=f'Model {i+1} ({trend})', 
                color=colors[i % len(colors)],
                linestyle='dashed',
                alpha=0.3)
    
    # Calculate and plot average prediction
    avg_predictions = np.mean(all_predictions, axis=0)
    ax.plot(first_pred.index, avg_predictions,
            label='Average Prediction',
            color='red',
            linewidth=1,
            linestyle='solid',
            alpha=0.8)
    
    ax.set_title('Ensemble Model Predictions')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    # ax.legend()
    # ax.grid(True)
    fig.autofmt_xdate()
    st.pyplot(fig)

def plot_ensemble_statistics(ensemble_predictions, test_df, preprocessor):
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[7, 3])
    
    # Box plot with enhanced styling (spans 70% of width)
    ax_box = fig.add_subplot(gs[0])
    first_pred = ensemble_predictions[0][0]
    all_predictions = np.zeros((len(ensemble_predictions), len(first_pred)))
    
    for i, (preds, trend) in enumerate(ensemble_predictions):
        normalized_preds = (preds['Auc Price'] * preprocessor.train_std['Auc Price']) + preprocessor.train_mean['Auc Price']
        all_predictions[i] = normalized_preds
    
    # Create violin plot with box plot inside
    parts = ax_box.violinplot(all_predictions, showmeans=True)
    bp = ax_box.boxplot(all_predictions, patch_artist=True)
    
    # Style violin plot
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.3)
    parts['cmeans'].set_color('red')
    
    # Style box plot
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_predictions[0])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax_box.set_title('Price Distribution by Day', pad=20)
    ax_box.set_ylabel('Price')
    ax_box.set_xlabel('Prediction Day')
    ax_box.grid(True, alpha=0.3)
    
    # Consensus Plot (30% of width)
    ax_consensus = fig.add_subplot(gs[1])
    buy_votes = sum(1 for _, trend in ensemble_predictions if trend == 'positive')
    sell_votes = len(ensemble_predictions) - buy_votes
    
    colors = ['lightgreen' if buy_votes > sell_votes else 'lightcoral', 
              'lightcoral' if buy_votes > sell_votes else 'lightgreen']
    sizes = [buy_votes, sell_votes]
    labels = [f'Buy\n{buy_votes} votes', f'Sell\n{sell_votes} votes']
    
    ax_consensus.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=90, pctdistance=0.85,
                    wedgeprops=dict(width=0.5))
    ax_consensus.set_title('Model Consensus', pad=20)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Calculate statistics for DataFrame
    mean_pred = np.mean(all_predictions, axis=0)
    std_pred = np.std(all_predictions, axis=0)
    cv = (std_pred/mean_pred) * 100
    
    # Create statistics DataFrame
    stats_dict = {
        'Metric': [
            'Number of Models',
            'Average Final Prediction',
            'Standard Deviation',
            'Coefficient of Variation',
            'Consensus Strength'
        ],
        'Value': [
            len(ensemble_predictions),
            f"{mean_pred[-1]:.2f}",
            f"{std_pred[-1]:.2f}",
            f"{cv[-1]:.1f}%",
            'Strong' if max(buy_votes, sell_votes)/len(ensemble_predictions) > 0.7 else 'Weak'
        ]
    }
    
    stats_df = pd.DataFrame(stats_dict)
    st.dataframe(stats_df, width='stretch', hide_index=True)

def plot_ensemble_predictions_realtime(predictions_list, test_df, preprocessor, container):
    with container:
        plt.figure(figsize=(12, 6))
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        test_df = reverse_normalize(test_df.copy(), 
                                  preprocessor.train_mean['Auc Price'], 
                                  preprocessor.train_std['Auc Price'])
        plot_df = test_df.copy().tail(90)
        ax.plot(plot_df.index, plot_df['Auc Price'], 
                label='Actual Price', color='black', marker='o', markersize=3)
        
        # Plot predictions from each model
        colors = plt.cm.cool(np.linspace(0, 1, 50))
        all_predictions = []
        
        for i, (preds, trend) in enumerate(predictions_list):
            normalized_preds = (preds['Auc Price'] * preprocessor.train_std['Auc Price']) + preprocessor.train_mean['Auc Price']
            all_predictions.append(normalized_preds)
            ax.plot(preds.index, normalized_preds, 
                    label=f'Model {i+1} ({trend})', 
                    color=colors[i % len(colors)],
                    linestyle='dashed',
                    linewidth=1,
                    alpha=0.3)
        
        # If we have more than one prediction, plot the average
        if len(all_predictions) > 1:
            avg_predictions = np.mean([pred.values for pred in all_predictions], axis=0)
            ax.plot(preds.index, avg_predictions,
                    label=f'Average Prediction (n={len(predictions_list)})',
                    color='red',
                    linewidth=1,
                    linestyle='solid')
        
        ax.set_title('Ensemble Model Predictions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        # ax.legend()
        # ax.grid(True)
        fig.autofmt_xdate()
        st.pyplot(fig)

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

    fig, ax1 = plt.subplots(figsize=(10, 5))

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

    plt.figure(figsize=(10, 5))

    fig, ax = plt.subplots(figsize=(10, 5))

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
    # predictions_df.loc[predictions_df['Auc Price'] < 0, 'Auc Price'] = predictions_df['Auc Price'].mean()
    # ax.plot(predictions_df.index, predictions_df['Auc Price'], label='Predicted Auc Price', linestyle='dashed', color='orange')

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

# def plot_recent_predictions(recent_preds_orig, trend, test_df_orig, preprocessor):

#     plt.figure(figsize=(10, 4))
#     fig, ax = plt.subplots(figsize=(10, 4))

#     # test_df = test_df.iloc[-10:]
#     recent_preds = recent_preds_orig.round(3).copy()

#     test_df = test_df_orig.copy()
#     test_df = reverse_normalize(test_df, preprocessor.train_mean['Auc Price'], preprocessor.train_std['Auc Price'])
#     plot_df = test_df.copy().tail(90)

#     ax.plot(plot_df.index, plot_df['Auc Price'], label='Auc Price', color='tomato', marker='o', markersize=3)

#     pred_diff = np.mean(recent_preds.iloc[1:]['Auc Price'].values) - recent_preds.iloc[1]['Auc Price']

#     prediction_price = recent_preds.iloc[0]['Auc Price']
#     prediction_price = (prediction_price * preprocessor.train_std['Auc Price']) + preprocessor.train_mean['Auc Price'] 

#     recent_preds['Auc Price'] = (recent_preds['Auc Price'] * preprocessor.train_std['Auc Price']) + preprocessor.train_mean['Auc Price']
#     grad = round(np.mean(np.gradient(recent_preds['Auc Price'].values[1:])), 4)
#     if pred_diff > 0:
#         ax.text(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], grad, fontsize=5, verticalalignment='bottom')
#         ax.scatter(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], color='green', marker='^', s=100)
#         ax.text(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], "Buy", fontsize=5, verticalalignment='top')
#         ax.plot(recent_preds.index, recent_preds['Auc Price'], color='green', label="Prediction", alpha=0.4, linestyle='dashed')
#     else: 
#         ax.text(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], grad, fontsize=5, verticalalignment='bottom')
#         ax.scatter(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], color='red', marker='v', s=100)
#         ax.text(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], "Sell", fontsize=5, verticalalignment='top')
#         ax.plot(recent_preds.index, recent_preds['Auc Price'], color='red', label="Prediction", alpha=0.4, linestyle='dashed')

#     ax.set_title('Recent Predictions', fontsize=10)
#     ax.set_xlabel('Date', fontsize=8)
#     ax.set_ylabel('Price', fontsize=8)
#     ax.xaxis.set_tick_params(labelsize=7, rotation=180)
#     ax.yaxis.set_tick_params(labelsize=7)
#     ax.legend()
#     ax.grid(True)
#     fig.autofmt_xdate()
#     st.pyplot(fig)
    
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

def plot_external_variables_correlation(df, preprocessor):
    """
    Plot correlation between external variables and carbon prices
    """
    external_cols = ['Brent_Oil', 'TTF_Gas', 'EU_Inflation']
    
    # Check if external variables exist
    if not all(col in df.columns for col in external_cols):
        st.info("⚠️ External variables not included in this dataset")
        return
    
    from utils.data_processing import reverse_normalize
    
    # Denormalize for visualization
    df_denorm = reverse_normalize(
        df.copy(), 
        preprocessor.train_mean['Auc Price'], 
        preprocessor.train_std['Auc Price']
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Brent Oil vs Carbon Price
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    ax1.plot(df_denorm.index, df_denorm['Auc Price'], 'b-', label='Carbon Price', alpha=0.7, linewidth=1.5)
    ax1_twin.plot(df_denorm.index, df_denorm['Brent_Oil'], 'r-', label='Brent Oil', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Carbon Price (€/tCO2)', color='b')
    ax1_twin.set_ylabel('Brent Oil ($/barrel)', color='r')
    ax1.set_title('Carbon Price vs Brent Oil')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Plot 2: TTF Gas vs Carbon Price
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    ax2.plot(df_denorm.index, df_denorm['Auc Price'], 'b-', label='Carbon Price', alpha=0.7, linewidth=1.5)
    ax2_twin.plot(df_denorm.index, df_denorm['TTF_Gas'], 'g-', label='TTF Gas', alpha=0.7, linewidth=1.5)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Carbon Price (€/tCO2)', color='b')
    ax2_twin.set_ylabel('TTF Gas (€/MWh)', color='g')
    ax2.set_title('Carbon Price vs TTF Gas')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='g')
    
    # Plot 3: EU Inflation vs Carbon Price
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    ax3.plot(df_denorm.index, df_denorm['Auc Price'], 'b-', label='Carbon Price', alpha=0.7, linewidth=1.5)
    ax3_twin.plot(df_denorm.index, df_denorm['EU_Inflation'], color='purple', label='EU Inflation', alpha=0.7, linewidth=1.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Carbon Price (€/tCO2)', color='b')
    ax3_twin.set_ylabel('EU Inflation (%)', color='purple')
    ax3.set_title('Carbon Price vs EU Inflation')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='y', labelcolor='b')
    ax3_twin.tick_params(axis='y', labelcolor='purple')
    
    # Plot 4: Correlation matrix
    ax4 = axes[1, 1]
    corr_cols = ['Auc Price', 'Brent_Oil', 'TTF_Gas', 'EU_Inflation']
    corr_matrix = df_denorm[corr_cols].corr()
    
    im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax4.set_xticks(range(len(corr_cols)))
    ax4.set_yticks(range(len(corr_cols)))
    ax4.set_xticklabels(['Carbon', 'Brent', 'TTF Gas', 'Inflation'], rotation=45, ha='right')
    ax4.set_yticklabels(['Carbon', 'Brent', 'TTF Gas', 'Inflation'])
    
    # Add correlation values
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
                           fontsize=10, fontweight='bold')
    
    ax4.set_title('Correlation Matrix')
    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
