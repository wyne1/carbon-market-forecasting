"""
Database utilities for trading automation system
"""

import psycopg2
import pandas as pd
import logging
from datetime import datetime
from config import DB_CONFIG

logger = logging.getLogger(__name__)

def create_database():
    """Create the wentworth database if it doesn't exist"""
    try:
        # Connect to default postgres database to create wentworth
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database='postgres'  # Connect to default postgres db
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'wentworth'")
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute("CREATE DATABASE wentworth")
            logger.info("Database 'wentworth' created successfully")
        else:
            logger.info("Database 'wentworth' already exists")
            
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False

def create_equity_table():
    """Create the equity_data table with proper schema including new calculated columns"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Create table with all columns including new calculated ones
        create_table_query = """
        CREATE TABLE IF NOT EXISTS equity_data (
            id SERIAL PRIMARY KEY,
            business_date DATE NOT NULL,
            managing_location VARCHAR(50),
            account_id VARCHAR(50),
            family_group_code_1 VARCHAR(50),
            legal_account_name VARCHAR(255),
            currency VARCHAR(10),
            starting_cash DECIMAL(15,2),
            daily_charges DECIMAL(15,2),
            daily_tax DECIMAL(15,2),
            daily_option_premiums DECIMAL(15,2),
            profit_and_loss DECIMAL(15,2),
            daily_transfers DECIMAL(15,2),
            daily_received_cash DECIMAL(15,2),
            daily_cash_paid DECIMAL(15,2),
            cash DECIMAL(15,2),
            open_trade_equity DECIMAL(15,2),
            total_equity DECIMAL(15,2),
            net_option_value DECIMAL(15,2),
            net_liquidation_value DECIMAL(15,2),
            initial_margin DECIMAL(15,2),
            maintenance_margin DECIMAL(15,2),
            excess_deficit DECIMAL(15,2),
            mtd_profit_and_loss DECIMAL(15,2),
            ytd_profit_and_loss DECIMAL(15,2),
            forward_cash_entries DECIMAL(15,2),
            forward_futures_pl DECIMAL(15,2),
            forward_charges DECIMAL(15,2),
            -- New calculated columns
            wpac_bizone DECIMAL(15,2),
            wpac_cash_reserve DECIMAL(15,2),
            fum DECIMAL(15,2),
            margin_utilisation DECIMAL(15,6),
            drawdown_total_fum DECIMAL(15,6),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(business_date, account_id)
        );
        """
        
        cursor.execute(create_table_query)
        
        # Create indexes for efficient querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_business_date ON equity_data(business_date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_account_id ON equity_data(account_id);")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("Equity data table created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error creating equity table: {e}")
        return False

def update_equity_table_schema():
    """Update existing equity_data table to include new calculated columns"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Check if new columns exist
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'equity_data' 
            AND column_name IN ('wpac_bizone', 'wpac_cash_reserve', 'fum', 'margin_utilisation', 'drawdown_total_fum')
        """)
        existing_columns = [row[0] for row in cursor.fetchall()]
        
        # Add missing columns
        new_columns = [
            ('wpac_bizone', 'DECIMAL(15,2)'),
            ('wpac_cash_reserve', 'DECIMAL(15,2)'),
            ('fum', 'DECIMAL(15,2)'),
            ('margin_utilisation', 'DECIMAL(15,6)'),
            ('drawdown_total_fum', 'DECIMAL(15,6)')
        ]
        
        for column_name, column_type in new_columns:
            if column_name not in existing_columns:
                alter_query = f"ALTER TABLE equity_data ADD COLUMN {column_name} {column_type}"
                cursor.execute(alter_query)
                logger.info(f"Added column: {column_name}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("Equity data table schema updated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error updating equity table schema: {e}")
        return False

def get_latest_date():
    """Get the latest business date from the database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute("SELECT MAX(business_date) FROM equity_data")
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return result[0] if result[0] else None
        
    except Exception as e:
        logger.error(f"Error getting latest date: {e}")
        return None

def get_database_stats():
    """Get statistics about the database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM equity_data")
        total_count = cursor.fetchone()[0]
        
        # Get date range
        cursor.execute("SELECT MIN(business_date), MAX(business_date) FROM equity_data")
        date_range = cursor.fetchone()
        
        # Get latest record
        cursor.execute("SELECT business_date, account_id, total_equity FROM equity_data ORDER BY business_date DESC LIMIT 1")
        latest_record = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        stats = {
            'total_records': total_count,
            'date_range': date_range,
            'latest_record': latest_record
        }
        
        logger.info(f"Database Stats: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return None

def calculate_derived_columns(row, wpac_bizone, wpac_cash_reserve):
    """Calculate the derived columns based on the formulas"""
    try:
        # Handle both Excel column names and lowercase CSV column names
        # Convert to float to handle Decimal types from database
        cash = float(row.get('cash') or row.get('Cash', 0))
        maintenance_margin = float(row.get('maintenance_margin') or row.get('Maintenance Margin', 0))
        open_trade_equity = float(row.get('open_trade_equity') or row.get('Open Trade Equity', 0))
        
        # FUM = Cash + Wpac BizOne + Wpac Cash Reserve
        fum = cash + float(wpac_bizone) + float(wpac_cash_reserve)
        
        # Margin Utilisation = Maintenance Margin / FUM
        margin_utilisation = maintenance_margin / fum if fum != 0 else 0
        
        # Drawdown (Total FUM) = |Open Trade Equity| / FUM (absolute value to ensure positive)
        drawdown_total_fum = abs(open_trade_equity) / fum if fum != 0 else 0
        
        return {
            'wpac_bizone': wpac_bizone,
            'wpac_cash_reserve': wpac_cash_reserve,
            'fum': fum,
            'margin_utilisation': margin_utilisation,
            'drawdown_total_fum': drawdown_total_fum
        }
        
    except Exception as e:
        logger.error(f"Error calculating derived columns: {e}")
        return {
            'wpac_bizone': wpac_bizone,
            'wpac_cash_reserve': wpac_cash_reserve,
            'fum': 0,
            'margin_utilisation': 0,
            'drawdown_total_fum': 0
        }