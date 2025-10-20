"""
Excel generator for trading automation system
"""

import pandas as pd
import psycopg2
import logging
from datetime import datetime
from config import DB_CONFIG, EQUITY_BASE
from chart_generator import add_charts_to_excel

logger = logging.getLogger(__name__)

def generate_excel_from_database(output_file=None):
    """Generate Excel file from database data with all columns including calculated ones"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Query all data from database including calculated columns
        query = """
        SELECT 
            business_date, starting_cash, daily_charges, profit_and_loss, 
            daily_transfers, daily_received_cash, daily_cash_paid, cash, 
            open_trade_equity, total_equity, net_liquidation_value, 
            initial_margin, maintenance_margin, excess_deficit, 
            mtd_profit_and_loss, ytd_profit_and_loss, wpac_bizone, 
            wpac_cash_reserve, fum, margin_utilisation, drawdown_total_fum
        FROM equity_data 
        ORDER BY business_date DESC;
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Rename columns to match original Excel format exactly
        column_mapping = {
            'business_date': 'Business Date (Start)',
            'starting_cash': 'Starting Cash',
            'daily_charges': 'Daily Charges',
            'profit_and_loss': 'Profit and Loss',
            'daily_transfers': 'Daily Transfers',
            'daily_received_cash': 'Daily Received Cash',
            'daily_cash_paid': 'Daily Cash Paid',
            'cash': 'Cash',
            'open_trade_equity': 'Open Trade Equity',
            'total_equity': 'Total Equity',
            'net_liquidation_value': 'Net Liquidation Value',
            'initial_margin': 'Initial Margin',
            'maintenance_margin': 'Maintenance Margin',
            'excess_deficit': 'Excess Deficit',
            'mtd_profit_and_loss': 'MTD Profit and Loss',
            'ytd_profit_and_loss': 'YTD Profit and Loss',
            'wpac_bizone': 'Wpac BizOne',
            'wpac_cash_reserve': 'Wpac Cash Reserve',
            'fum': 'FUM',
            'margin_utilisation': 'Margin Utilisation',
            'drawdown_total_fum': 'Drawdown (Total FUM)'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Add the additional columns that are in the original Excel but not in our database
        # These will be calculated from existing data
        additional_columns = {
            'Gross Trading P&L': df['Profit and Loss'],  # Same as Profit and Loss
            'Net Trading P&L': df['Profit and Loss'],    # Same as Profit and Loss  
            'Open Trade Equity ': df['Open Trade Equity'],  # Note the trailing space
            'Daily Futures P&L': df['Profit and Loss']   # Same as Profit and Loss
        }
        
        # Add the additional columns
        for col_name, col_data in additional_columns.items():
            df[col_name] = col_data
        
        # Reorder columns to match original Excel format
        column_order = [
            'Business Date (Start)', 'Starting Cash', 'Daily Charges', 'Profit and Loss',
            'Daily Transfers', 'Daily Received Cash', 'Daily Cash Paid', 'Cash',
            'Open Trade Equity', 'Total Equity', 'Net Liquidation Value', 'Initial Margin',
            'Maintenance Margin', 'Excess Deficit', 'MTD Profit and Loss', 'YTD Profit and Loss',
            'Gross Trading P&L', 'Net Trading P&L', 'Open Trade Equity ', 'Daily Futures P&L',
            'Drawdown (Total FUM)', 'Margin Utilisation', 'FUM', 'Wpac BizOne', 'Wpac Cash Reserve'
        ]
        
        df = df[column_order]
        
        # Generate output filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'data/Wentworth_Stock_Trading_Futures_Account_{timestamp}.xlsx'
        
        # Create Excel file with multiple sheets
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Main equity data sheet
            df.to_excel(writer, sheet_name=EQUITY_BASE, index=False)
            
            # Summary sheet
            summary_data = {
                'Total Records': [len(df)],
                'Date Range': [f"{df['Business Date (Start)'].min()} to {df['Business Date (Start)'].max()}"],
                'Last Updated': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'Account ID': ['S3121200'],
                'Legal Account Name': ['WENTWORTH STOCK AND TRADING PTY LTD']
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Add charts
            logger.info("Generating charts...")
            add_charts_to_excel(writer, df)
        
        logger.info(f"Excel file generated successfully: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error generating Excel file: {e}")
        return None