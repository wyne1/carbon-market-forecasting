# Copy and paste this code into your Jupyter notebook cell

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create daily trading hours visualization
def plot_daily_trading_hours(df_cf, price_column='HIGH_1'):
    """
    Create a daily trading hours chart showing price vs time of day
    """
    
    # Create a copy to avoid modifying original data
    df = df_cf.copy()
    
    # Convert timestamp to datetime if not already
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    elif 'Date' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Date'])
    else:
        raise ValueError("No timestamp column found. Expected 'Timestamp' or 'Date'")
    
    # Extract time components
    df['time'] = df['Timestamp'].dt.time
    df['hour'] = df['Timestamp'].dt.hour
    df['minute'] = df['Timestamp'].dt.minute
    df['time_of_day'] = df['hour'] + df['minute'] / 60.0
    
    # Remove rows with missing price data
    df_clean = df.dropna(subset=[price_column])
    
    if len(df_clean) == 0:
        print("No valid price data found!")
        return None
    
    print(f"Found {len(df_clean)} valid data points")
    print(f"Time range: {df_clean['Timestamp'].min()} to {df_clean['Timestamp'].max()}")
    print(f"Price range: {df_clean[price_column].min():.2f} to {df_clean[price_column].max():.2f}")
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Price vs Time of Day (scatter plot with line)
    ax1.scatter(df_clean['time_of_day'], df_clean[price_column], 
               alpha=0.7, s=50, c='blue', edgecolors='black', linewidth=0.5)
    
    # Add a trend line
    z = np.polyfit(df_clean['time_of_day'], df_clean[price_column], 1)
    p = np.poly1d(z)
    ax1.plot(df_clean['time_of_day'], p(df_clean['time_of_day']), 
            "r--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.3f})')
    
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel(f'{price_column} Price')
    ax1.set_title('Carbon Market: Price vs Trading Hours')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 24)
    ax1.legend()
    
    # Add hour markers
    ax1.set_xticks(range(0, 25, 2))
    ax1.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 2)])
    
    # Plot 2: Trading Activity by Hour (bar chart)
    hourly_activity = df_clean.groupby('hour').size()
    bars = ax2.bar(hourly_activity.index, hourly_activity.values, 
                  alpha=0.7, color='green', edgecolor='black')
    
    # Color bars based on activity level
    max_activity = hourly_activity.max()
    for bar, count in zip(bars, hourly_activity.values):
        if count == max_activity:
            bar.set_color('red')
        elif count > max_activity * 0.7:
            bar.set_color('orange')
    
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Number of 10-min Intervals')
    ax2.set_title('Trading Activity by Hour (Peak hours in red/orange)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim(-0.5, 23.5)
    
    # Add value labels on bars
    for bar, count in zip(bars, hourly_activity.values):
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n=== Trading Hours Summary ===")
    print(f"Most active hour: {hourly_activity.idxmax()}:00 ({hourly_activity.max()} data points)")
    print(f"Least active hour: {hourly_activity.idxmin()}:00 ({hourly_activity.min()} data points)")
    print(f"Average price: {df_clean[price_column].mean():.2f}")
    print(f"Price volatility (std): {df_clean[price_column].std():.2f}")
    
    # Trading session analysis
    morning = df_clean[(df_clean['hour'] >= 8) & (df_clean['hour'] < 12)]
    afternoon = df_clean[(df_clean['hour'] >= 12) & (df_clean['hour'] < 16)]
    evening = df_clean[(df_clean['hour'] >= 16) & (df_clean['hour'] < 20)]
    
    print(f"\nTrading Session Activity:")
    print(f"  Morning (08:00-12:00): {len(morning)} intervals")
    print(f"  Afternoon (12:00-16:00): {len(afternoon)} intervals") 
    print(f"  Evening (16:00-20:00): {len(evening)} intervals")
    
    return fig

# Run the analysis with your df_cf data
plot_daily_trading_hours(df_cf, 'HIGH_1')

