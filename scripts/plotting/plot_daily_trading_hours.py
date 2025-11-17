#!/usr/bin/env python3
"""
Daily Trading Hours Visualization
================================

This script creates visualizations to show when the carbon market is most active
throughout the day using 10-minute interval data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

def plot_daily_trading_pattern(df_cf, price_column='HIGH_1', title="Carbon Market Trading Hours Pattern"):
    """
    Plot trading activity throughout the day showing price vs time of day
    
    Parameters:
    -----------
    df_cf : pandas.DataFrame
        DataFrame with 'Timestamp' and price column
    price_column : str
        Column name for price data (default: 'HIGH_1')
    title : str
        Title for the plot
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
    
    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Price vs Time of Day (scatter plot)
    ax1.scatter(df_clean['time_of_day'], df_clean[price_column], 
               alpha=0.6, s=30, c='blue', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel(f'{price_column} Price')
    ax1.set_title(f'{title} - Price Distribution by Hour')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 24)
    
    # Add hour markers
    ax1.set_xticks(range(0, 25, 2))
    ax1.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 2)])
    
    # Plot 2: Trading Volume/Activity by Hour (bar chart)
    hourly_activity = df_clean.groupby('hour').size()
    ax2.bar(hourly_activity.index, hourly_activity.values, 
           alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Number of 10-min Intervals')
    ax2.set_title('Trading Activity by Hour (Number of Data Points)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim(-0.5, 23.5)
    
    # Plot 3: Price statistics by hour (box plot)
    hourly_data = []
    hourly_labels = []
    for hour in sorted(df_clean['hour'].unique()):
        hour_data = df_clean[df_clean['hour'] == hour][price_column].dropna()
        if len(hour_data) > 0:
            hourly_data.append(hour_data)
            hourly_labels.append(f'{hour:02d}:00')
    
    if hourly_data:
        ax3.boxplot(hourly_data, labels=hourly_labels)
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel(f'{price_column} Price')
        ax3.set_title('Price Distribution by Hour (Box Plot)')
        ax3.grid(True, alpha=0.3, axis='y')
        plt.setp(ax3.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

def plot_trading_intensity_heatmap(df_cf, price_column='HIGH_1', days_to_show=7):
    """
    Create a heatmap showing trading intensity throughout the day across multiple days
    
    Parameters:
    -----------
    df_cf : pandas.DataFrame
        DataFrame with 'Timestamp' and price column
    price_column : str
        Column name for price data
    days_to_show : int
        Number of recent days to show
    """
    
    df = df_cf.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.dropna(subset=[price_column])
    
    # Get the most recent days
    df['date'] = df['Timestamp'].dt.date
    recent_dates = sorted(df['date'].unique())[-days_to_show:]
    df_recent = df[df['date'].isin(recent_dates)]
    
    # Create time bins (10-minute intervals)
    df_recent['time_bin'] = df_recent['Timestamp'].dt.floor('10min').dt.time
    df_recent['hour'] = df_recent['Timestamp'].dt.hour
    df_recent['minute'] = df_recent['Timestamp'].dt.minute
    df_recent['time_index'] = df_recent['hour'] * 6 + df_recent['minute'] // 10
    
    # Create pivot table for heatmap
    heatmap_data = df_recent.pivot_table(
        values=price_column, 
        index='date', 
        columns='time_index', 
        aggfunc='count',  # Count of data points
        fill_value=0
    )
    
    # Create time labels for x-axis
    time_labels = [f'{h:02d}:{m:02d}' for h in range(24) for m in range(0, 60, 10)]
    
    # Plot heatmap
    plt.figure(figsize=(20, 8))
    sns.heatmap(heatmap_data, 
                xticklabels=time_labels[::6],  # Show every hour
                yticklabels=[str(d) for d in recent_dates],
                cmap='YlOrRd', 
                cbar_kws={'label': 'Number of Data Points'},
                annot=False)
    
    plt.title(f'Trading Activity Heatmap - Last {days_to_show} Days')
    plt.xlabel('Time of Day')
    plt.ylabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt.gcf()

def analyze_trading_hours(df_cf, price_column='HIGH_1'):
    """
    Analyze and print trading hours statistics
    
    Parameters:
    -----------
    df_cf : pandas.DataFrame
        DataFrame with 'Timestamp' and price column
    price_column : str
        Column name for price data
    """
    
    df = df_cf.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.dropna(subset=[price_column])
    
    df['hour'] = df['Timestamp'].dt.hour
    df['minute'] = df['Timestamp'].dt.minute
    df['time_of_day'] = df['hour'] + df['minute'] / 60.0
    
    print("=== Trading Hours Analysis ===")
    print(f"Total data points: {len(df)}")
    print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"Price range: {df[price_column].min():.2f} to {df[price_column].max():.2f}")
    print()
    
    # Hourly statistics
    hourly_stats = df.groupby('hour').agg({
        price_column: ['count', 'mean', 'std', 'min', 'max'],
        'time_of_day': 'first'
    }).round(2)
    
    print("Hourly Trading Statistics:")
    print("Hour | Count | Mean Price | Std Dev | Min Price | Max Price")
    print("-" * 60)
    
    for hour in sorted(df['hour'].unique()):
        hour_data = df[df['hour'] == hour]
        count = len(hour_data)
        mean_price = hour_data[price_column].mean()
        std_price = hour_data[price_column].std()
        min_price = hour_data[price_column].min()
        max_price = hour_data[price_column].max()
        
        print(f"{hour:4d} | {count:5d} | {mean_price:10.2f} | {std_price:7.2f} | {min_price:9.2f} | {max_price:9.2f}")
    
    # Most active hours
    hourly_counts = df['hour'].value_counts().sort_index()
    most_active_hours = hourly_counts.nlargest(5)
    
    print(f"\nMost Active Trading Hours:")
    for hour, count in most_active_hours.items():
        print(f"  {hour:02d}:00 - {count} data points")
    
    # Trading session analysis
    morning_session = df[(df['hour'] >= 8) & (df['hour'] < 12)]
    afternoon_session = df[(df['hour'] >= 12) & (df['hour'] < 16)]
    evening_session = df[(df['hour'] >= 16) & (df['hour'] < 20)]
    
    print(f"\nTrading Session Analysis:")
    print(f"  Morning (08:00-12:00): {len(morning_session)} data points")
    print(f"  Afternoon (12:00-16:00): {len(afternoon_session)} data points")
    print(f"  Evening (16:00-20:00): {len(evening_session)} data points")

def create_comprehensive_daily_analysis(df_cf, price_column='HIGH_1', save_plots=True):
    """
    Create a comprehensive analysis of daily trading patterns
    
    Parameters:
    -----------
    df_cf : pandas.DataFrame
        DataFrame with 'Timestamp' and price column
    price_column : str
        Column name for price data
    save_plots : bool
        Whether to save plots to files
    """
    
    print("Creating comprehensive daily trading analysis...")
    
    # Analyze trading hours
    analyze_trading_hours(df_cf, price_column)
    
    # Create main trading pattern plot
    fig1 = plot_daily_trading_pattern(df_cf, price_column)
    if save_plots and fig1:
        fig1.savefig('daily_trading_pattern.png', dpi=300, bbox_inches='tight')
        print("Saved: daily_trading_pattern.png")
    
    # Create heatmap
    fig2 = plot_trading_intensity_heatmap(df_cf, price_column)
    if save_plots and fig2:
        fig2.savefig('trading_intensity_heatmap.png', dpi=300, bbox_inches='tight')
        print("Saved: trading_intensity_heatmap.png")
    
    plt.show()
    
    return fig1, fig2

# Example usage function
def example_usage():
    """
    Example of how to use the functions with your data
    """
    
    # Assuming you have df_cf loaded from your notebook
    # df_cf = your_data_here
    
    # Create the analysis
    # create_comprehensive_daily_analysis(df_cf, price_column='HIGH_1')
    
    print("To use this script with your data:")
    print("1. Load your df_cf data")
    print("2. Call: create_comprehensive_daily_analysis(df_cf, price_column='HIGH_1')")
    print("3. This will create plots and print analysis statistics")

if __name__ == "__main__":
    example_usage()

