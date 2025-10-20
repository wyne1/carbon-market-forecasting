"""
Data processing utilities for trading automation system
"""

import pandas as pd
import psycopg2
import logging
from io import StringIO
from datetime import datetime
from config import DB_CONFIG
from database_utils import calculate_derived_columns

logger = logging.getLogger(__name__)

def process_csv_data(csv_content):
    """Process CSV content and return DataFrame"""
    try:
        df = pd.read_csv(StringIO(csv_content))
        
        # Clean column names (remove spaces, convert to lowercase)
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        
        # Convert Business Date to datetime
        if 'business_date' in df.columns:
            df['business_date'] = pd.to_datetime(df['business_date'])
        
        logger.info(f"Processed CSV with {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"Error processing CSV data: {e}")
        return None

def insert_csv_to_database(df, wpac_bizone, wpac_cash_reserve):
    """Insert CSV data into database with calculated columns"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Prepare data for insertion with calculated columns
        insert_query = """
        INSERT INTO equity_data (
            business_date, managing_location, account_id, family_group_code_1, 
            legal_account_name, currency, starting_cash, daily_charges, daily_tax,
            daily_option_premiums, profit_and_loss, daily_transfers, daily_received_cash,
            daily_cash_paid, cash, open_trade_equity, total_equity, net_option_value,
            net_liquidation_value, initial_margin, maintenance_margin, excess_deficit,
            mtd_profit_and_loss, ytd_profit_and_loss, forward_cash_entries,
            forward_futures_pl, forward_charges, wpac_bizone, wpac_cash_reserve,
            fum, margin_utilisation, drawdown_total_fum
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON CONFLICT (business_date, account_id) DO UPDATE SET
            managing_location = EXCLUDED.managing_location,
            family_group_code_1 = EXCLUDED.family_group_code_1,
            legal_account_name = EXCLUDED.legal_account_name,
            currency = EXCLUDED.currency,
            starting_cash = EXCLUDED.starting_cash,
            daily_charges = EXCLUDED.daily_charges,
            daily_tax = EXCLUDED.daily_tax,
            daily_option_premiums = EXCLUDED.daily_option_premiums,
            profit_and_loss = EXCLUDED.profit_and_loss,
            daily_transfers = EXCLUDED.daily_transfers,
            daily_received_cash = EXCLUDED.daily_received_cash,
            daily_cash_paid = EXCLUDED.daily_cash_paid,
            cash = EXCLUDED.cash,
            open_trade_equity = EXCLUDED.open_trade_equity,
            total_equity = EXCLUDED.total_equity,
            net_option_value = EXCLUDED.net_option_value,
            net_liquidation_value = EXCLUDED.net_liquidation_value,
            initial_margin = EXCLUDED.initial_margin,
            maintenance_margin = EXCLUDED.maintenance_margin,
            excess_deficit = EXCLUDED.excess_deficit,
            mtd_profit_and_loss = EXCLUDED.mtd_profit_and_loss,
            ytd_profit_and_loss = EXCLUDED.ytd_profit_and_loss,
            forward_cash_entries = EXCLUDED.forward_cash_entries,
            forward_futures_pl = EXCLUDED.forward_futures_pl,
            forward_charges = EXCLUDED.forward_charges,
            wpac_bizone = EXCLUDED.wpac_bizone,
            wpac_cash_reserve = EXCLUDED.wpac_cash_reserve,
            fum = EXCLUDED.fum,
            margin_utilisation = EXCLUDED.margin_utilisation,
            drawdown_total_fum = EXCLUDED.drawdown_total_fum,
            created_at = CURRENT_TIMESTAMP;
        """
        
        # Convert DataFrame to list of tuples with calculated columns
        data_tuples = []
        for _, row in df.iterrows():
            # Calculate derived columns
            derived = calculate_derived_columns(row, wpac_bizone, wpac_cash_reserve)
            
            data_tuple = (
                row.get('business_date'),
                row.get('managing_location'),
                row.get('account_id'),
                row.get('family_group_code_1'),
                row.get('legal_account_name'),
                row.get('currency'),
                row.get('starting_cash'),
                row.get('daily_charges'),
                row.get('daily_tax'),
                row.get('daily_option_premiums'),
                row.get('profit_and_loss'),
                row.get('daily_transfers'),
                row.get('daily_received_cash'),
                row.get('daily_cash_paid'),
                row.get('cash'),
                row.get('open_trade_equity'),
                row.get('total_equity'),
                row.get('net_option_value'),
                row.get('net_liquidation_value'),
                row.get('initial_margin'),
                row.get('maintenance_margin'),
                row.get('excess_deficit'),
                row.get('mtd_profit_and_loss'),
                row.get('ytd_profit_and_loss'),
                row.get('forward_cash_entries'),
                row.get('forward_futures_pl'),
                row.get('forward_charges'),
                derived['wpac_bizone'],
                derived['wpac_cash_reserve'],
                derived['fum'],
                derived['margin_utilisation'],
                derived['drawdown_total_fum']
            )
            data_tuples.append(data_tuple)
        
        # Insert data
        cursor.executemany(insert_query, data_tuples)
        conn.commit()
        
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully inserted {len(data_tuples)} records to database")
        return True
        
    except Exception as e:
        logger.error(f"Error inserting CSV data: {e}")
        return False

def migrate_excel_to_database(excel_file_path, wpac_bizone, wpac_cash_reserve):
    """Migrate existing Excel data to PostgreSQL database"""
    try:
        # Read the existing Excel file
        df = pd.read_excel(excel_file_path, sheet_name='S3121200 - Equity Base')
        
        logger.info(f"Loaded {len(df)} rows from Excel file")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Clean the data - remove rows with null business dates
        df_clean = df.dropna(subset=['Business Date (Start)'])
        logger.info(f"After cleaning: {len(df_clean)} rows")
        
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Prepare data for insertion - mapping Excel columns to database columns
        insert_query = """
        INSERT INTO equity_data (
            business_date, managing_location, account_id, family_group_code_1, 
            legal_account_name, currency, starting_cash, daily_charges, daily_tax,
            daily_option_premiums, profit_and_loss, daily_transfers, daily_received_cash,
            daily_cash_paid, cash, open_trade_equity, total_equity, net_option_value,
            net_liquidation_value, initial_margin, maintenance_margin, excess_deficit,
            mtd_profit_and_loss, ytd_profit_and_loss, forward_cash_entries,
            forward_futures_pl, forward_charges, wpac_bizone, wpac_cash_reserve,
            fum, margin_utilisation, drawdown_total_fum
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON CONFLICT (business_date, account_id) DO NOTHING;
        """
        
        # Convert DataFrame to list of tuples with proper mapping
        data_tuples = []
        for _, row in df_clean.iterrows():
            # Calculate derived columns
            derived = calculate_derived_columns(row, wpac_bizone, wpac_cash_reserve)
            
            # Map Excel columns to database columns
            data_tuple = (
                row.get('Business Date (Start)'),  # business_date
                'SINSFT',  # managing_location (default value)
                'S3121200',  # account_id (default value)
                None,  # family_group_code_1
                'WENTWORTH STOCK AND TRADING PTY LTD',  # legal_account_name (default value)
                'AUD',  # currency (default value)
                row.get('Starting Cash'),  # starting_cash
                row.get('Daily Charges'),  # daily_charges
                None,  # daily_tax
                None,  # daily_option_premiums
                row.get('Profit and Loss'),  # profit_and_loss
                row.get('Daily Transfers'),  # daily_transfers
                row.get('Daily Received Cash'),  # daily_received_cash
                row.get('Daily Cash Paid'),  # daily_cash_paid
                row.get('Cash'),  # cash
                row.get('Open Trade Equity'),  # open_trade_equity
                row.get('Total Equity'),  # total_equity
                None,  # net_option_value
                row.get('Net Liquidation Value'),  # net_liquidation_value
                row.get('Initial Margin'),  # initial_margin
                row.get('Maintenance Margin'),  # maintenance_margin
                row.get('Excess Deficit'),  # excess_deficit
                row.get('MTD Profit and Loss'),  # mtd_profit_and_loss
                row.get('YTD Profit and Loss'),  # ytd_profit_and_loss
                None,  # forward_cash_entries
                None,  # forward_futures_pl
                None,  # forward_charges
                derived['wpac_bizone'],
                derived['wpac_cash_reserve'],
                derived['fum'],
                derived['margin_utilisation'],
                derived['drawdown_total_fum']
            )
            data_tuples.append(data_tuple)
        
        # Insert data
        cursor.executemany(insert_query, data_tuples)
        conn.commit()
        
        # Get count of inserted records
        cursor.execute("SELECT COUNT(*) FROM equity_data")
        count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully migrated {count} records to database")
        return True
        
    except Exception as e:
        logger.error(f"Error migrating Excel data: {e}")
        return False