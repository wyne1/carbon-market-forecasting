"""
Gmail API integration for downloading equity data emails
"""
import os
import pickle
import base64
import email
from datetime import datetime, timedelta
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import logging

logger = logging.getLogger(__name__)

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class GmailAPI:
    def __init__(self, credentials_file='credentials.json', token_file='token.json'):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self.authenticate()
    
    def authenticate(self):
        """Authenticate with Gmail API"""
        creds = None
        
        # Load existing token
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(f"Credentials file {self.credentials_file} not found. Please download it from Google Cloud Console.")
                
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('gmail', 'v1', credentials=creds)
        logger.info("Gmail API authenticated successfully")
    
    def search_emails(self, query, max_results=10):
        """Search for emails with given query"""
        try:
            results = self.service.users().messages().list(
                userId='me', 
                q=query, 
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            logger.info(f"Found {len(messages)} emails matching query: {query}")
            return messages
            
        except Exception as e:
            logger.error(f"Error searching emails: {e}")
            return []
    
    def get_email_by_date(self, date_str):
        """Get email for specific date (YYYYMMDD format)"""
        query = f'subject:"TIDS3121200 Equity {date_str}"'
        messages = self.search_emails(query, max_results=1)
        
        if messages:
            return messages[0]
        return None
    
    def get_emails_by_date_range(self, start_date, end_date):
        """Get emails for date range"""
        emails = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            message = self.get_email_by_date(date_str)
            if message:
                emails.append((date_str, message))
            current_date += timedelta(days=1)
        
        return emails
    
    def get_message_details(self, message_id):
        """Get full message details"""
        try:
            message = self.service.users().messages().get(
                userId='me', 
                id=message_id,
                format='full'
            ).execute()
            return message
        except Exception as e:
            logger.error(f"Error getting message details: {e}")
            return None
    
    def download_csv_attachment(self, message_id, filename_pattern="Equity"):
        """Download CSV attachment from email"""
        try:
            message = self.get_message_details(message_id)
            if not message:
                return None
            
            # Get message payload
            payload = message['payload']
            
            # Find CSV attachment
            if 'parts' in payload:
                for part in payload['parts']:
                    if part['filename'] and part['filename'].startswith(filename_pattern) and part['filename'].endswith('.csv'):
                        # Get attachment data
                        attachment_id = part['body']['attachmentId']
                        attachment = self.service.users().messages().attachments().get(
                            userId='me',
                            messageId=message_id,
                            id=attachment_id
                        ).execute()
                        
                        # Decode attachment data
                        data = base64.urlsafe_b64decode(attachment['data'])
                        return data
            
            logger.warning(f"No CSV attachment found in message {message_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error downloading CSV attachment: {e}")
            return None
    
    def get_daily_equity_data(self, date_str):
        """Get equity data for specific date"""
        message = self.get_email_by_date(date_str)
        if not message:
            logger.warning(f"No email found for date {date_str}")
            return None
        
        csv_data = self.download_csv_attachment(message['id'])
        if csv_data:
            return csv_data.decode('utf-8')
        
        return None
