"""
Chart generation utilities for trading automation system
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import io
import logging
from openpyxl.drawing.image import Image
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.chart.axis import DateAxis

logger = logging.getLogger(__name__)

def create_pnl_chart(df):
    """Create P&L chart"""
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot P&L
        ax.plot(df['Business Date (Start)'], df['Profit and Loss'], 
                linewidth=2, color='#2E8B57', label='Daily P&L')
        
        # Add zero line
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Line')
        
        # Formatting
        ax.set_title('Profit & Loss Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('P&L ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        # Convert to image
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        logger.error(f"Error creating P&L chart: {e}")
        return None

def create_open_trade_equity_chart(df):
    """Create Open Trade Equity chart"""
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot Open Trade Equity
        ax.plot(df['Business Date (Start)'], df['Open Trade Equity'], 
                linewidth=2, color='#4169E1', label='Open Trade Equity')
        
        # Add zero line
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Line')
        
        # Formatting
        ax.set_title('Open Trade Equity Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Open Trade Equity ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        # Convert to image
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        logger.error(f"Error creating Open Trade Equity chart: {e}")
        return None

def create_margin_utilisation_chart(df):
    """Create Margin Utilisation chart"""
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot Margin Utilisation as area chart
        ax.fill_between(df['Business Date (Start)'], df['Margin Utilisation'], 
                       alpha=0.7, color='#FF6347', label='Margin Utilisation')
        
        # No reference line needed for margin utilisation
        
        # Formatting
        ax.set_title('Margin Utilisation Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Margin Utilisation (%)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}%'))
        
        plt.tight_layout()
        
        # Convert to image
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        logger.error(f"Error creating Margin Utilisation chart: {e}")
        return None

def create_drawdown_chart(df):
    """Create Drawdown chart"""
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot Drawdown
        ax.plot(df['Business Date (Start)'], df['Drawdown (Total FUM)'], 
                linewidth=2, color='#DC143C', label='Drawdown')
        
        # Add zero line
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Line')
        
        # Formatting
        ax.set_title('Drawdown (Total FUM) Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}%'))
        
        plt.tight_layout()
        
        # Convert to image
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        logger.error(f"Error creating Drawdown chart: {e}")
        return None

def add_charts_to_excel(writer, df):
    """Add charts to Excel file"""
    try:
        # Create charts
        charts = {
            'P&L': create_pnl_chart(df),
            'Open Trade Equity': create_open_trade_equity_chart(df),
            'Margin Utilization': create_margin_utilisation_chart(df),
            'Drawdown': create_drawdown_chart(df)
        }
        
        # Add each chart as a separate sheet
        for chart_name, chart_buffer in charts.items():
            if chart_buffer:
                # Create a new sheet for the chart
                chart_sheet = writer.book.create_sheet(chart_name)
                
                # Add the chart image to the sheet
                img = Image(chart_buffer)
                img.width = 800
                img.height = 400
                chart_sheet.add_image(img, 'A1')
                
                logger.info(f"Added {chart_name} chart to Excel file")
            else:
                logger.warning(f"Failed to create {chart_name} chart")
        
        return True
        
    except Exception as e:
        logger.error(f"Error adding charts to Excel: {e}")
        return False