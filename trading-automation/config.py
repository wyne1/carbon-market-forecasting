"""
Configuration file for trading automation system
"""

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'zeerakwyne',
    'password': '',
    'database': 'wentworth'
}

# Gmail Configuration
GMAIL_CREDENTIALS = {
    'credentials_file': '../credentials.json',  # Path to Gmail API credentials
    'token_file': '../token.json'  # Path to Gmail API token
}

# Excel Configuration
EQUITY_BASE = 'S3121200 - Equity Base'
WPAC_STATEMENT = 'Wpac Statement data'

# Default values for Wpac columns
DEFAULT_WPAC_BIZONE = 939
DEFAULT_WPAC_CASH_RESERVE = 26789.05

# Email search patterns
EMAIL_SUBJECT_PATTERN = "TIDS3121200 Equity"