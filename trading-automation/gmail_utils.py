"""
Gmail utilities for trading automation system
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from config import GMAIL_CREDENTIALS, EMAIL_SUBJECT_PATTERN

# Add current directory to path to import gmail_api
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

def setup_gmail_api():
    """Initialize Gmail API connection"""
    try:
        from gmail_api import GmailAPI
        gmail = GmailAPI(
            credentials_file=GMAIL_CREDENTIALS['credentials_file'],
            token_file=GMAIL_CREDENTIALS['token_file']
        )
        logger.info("Gmail API setup successful")
        return gmail
    except Exception as e:
        logger.error(f"Error setting up Gmail API: {e}")
        return None

def get_emails_since_date(start_date):
    """Get all emails since a specific date"""
    try:
        gmail = setup_gmail_api()
        if not gmail:
            logger.error("Failed to setup Gmail API")
            return []
        
        # Convert start_date to datetime if it's a string or date
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        elif hasattr(start_date, 'date'):  # datetime.date object
            start_date = datetime.combine(start_date, datetime.min.time())
        
        # Get emails for date range from start_date to today
        end_date = datetime.now()
        emails = gmail.get_emails_by_date_range(start_date, end_date)
        
        if not emails:
            logger.warning(f"No emails found since {start_date}")
            return []
        
        logger.info(f"Found {len(emails)} emails since {start_date}")
        return emails
        
    except Exception as e:
        logger.error(f"Error getting emails since date: {e}")
        return []

def process_email_data(email_data):
    """Process email data and return CSV content"""
    try:
        gmail = setup_gmail_api()
        if not gmail:
            return None
        
        date_str, message = email_data
        csv_data = gmail.download_csv_attachment(message['id'])
        
        if csv_data:
            return csv_data.decode('utf-8')
        else:
            logger.error(f"No CSV data found for {date_str}")
            return None
            
    except Exception as e:
        logger.error(f"Error processing email data: {e}")
        return None