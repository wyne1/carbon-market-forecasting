import pandas as pd
import numpy as np
from datetime import timedelta
import warnings

class SmartAuctionPreprocessor:
    """
    Better preprocessing that respects the auction schedule and handles missing values intelligently
    """
    
    def __init__(self):
        self.auction_schedule_pattern = None
        self.market_regime_changes = []
        
    def analyze_auction_pattern(self, df):
        """Analyze the actual auction schedule pattern"""
        df_clean = df.dropna(subset=['Auction Price']).copy()
        
        if len(df_clean) == 0:
            print("⚠️ No valid auction data found!")
            return None, None
            
        # Analyze day of week patterns
        day_counts = df_clean['Day of Week'].value_counts()
        print("📅 Auction Day Patterns:")
        print(day_counts)
        
        # Analyze gaps between auctions
        df_clean = df_clean.sort_values('Date')
        df_clean['Days_Since_Last'] = df_clean['Date'].diff().dt.days
        gap_analysis = df_clean['Days_Since_Last'].describe()
        print("\n⏰ Days Between Auctions:")
        print(gap_analysis)
        
        return day_counts, gap_analysis
    
    def identify_auction_types(self, df):
        """Identify different types of auctions based on missing data patterns"""
        df = df.copy()
        df['Auction_Type'] = 'Regular'
        
        # Type 1: Missing market data (BID/ASK missing)
        if 'BID' in df.columns and 'ASK' in df.columns:
            market_missing = df['BID'].isna() | df['ASK'].isna()
            df.loc[market_missing, 'Auction_Type'] = 'Limited_Market_Data'
        
        # Type 2: Special auctions (very low volume)
        if 'VOL_DEC' in df.columns:
            special_volume = df['VOL_DEC'] < 1000000  # Less than 1M
            df.loc[special_volume, 'Auction_Type'] = 'Special_Auction'
        
        # Type 3: Price-only auctions (missing median price)
        price_only = df['Median Price'].isna() & df['Auction Price'].notna()
        df.loc[price_only, 'Auction_Type'] = 'Price_Only'
        
        print("🏷️ Auction Types Identified:")
        print(df['Auction_Type'].value_counts())
        
        return df
    
    def smart_missing_value_handling(self, df):
        """Handle missing values based on auction context, not just forward fill"""
        df = df.copy()
        
        # For each auction, handle missing values contextually
        for idx, row in df.iterrows():
            auction_type = row.get('Auction_Type', 'Regular')
            
            # Handle missing median price
            if pd.isna(row['Median Price']) and pd.notna(row['Auction Price']):
                if auction_type == 'Special_Auction':
                    # For special auctions, median might equal auction price
                    df.loc[idx, 'Median Price'] = row['Auction Price']
                else:
                    # Use recent relationship between auction and median
                    recent_data = df.loc[:idx].dropna(subset=['Auction Price', 'Median Price']).tail(5)
                    if len(recent_data) > 0:
                        avg_ratio = (recent_data['Median Price'] / recent_data['Auction Price']).mean()
                        if not np.isnan(avg_ratio):
                            df.loc[idx, 'Median Price'] = row['Auction Price'] * avg_ratio
                        else:
                            df.loc[idx, 'Median Price'] = row['Auction Price']  # Fallback
                    else:
                        df.loc[idx, 'Median Price'] = row['Auction Price']  # Fallback
            
            # Handle missing market data (BID/ASK)
            if 'BID' in df.columns and 'ASK' in df.columns and 'AVG_PRC1' in df.columns:
                if pd.isna(row['BID']) or pd.isna(row['ASK']):
                    if pd.notna(row['AVG_PRC1']):
                        # Use AVG_PRC1 to estimate BID/ASK spread
                        recent_spreads = df.loc[:idx].dropna(subset=['BID', 'ASK', 'AVG_PRC1']).tail(10)
                        if len(recent_spreads) > 0:
                            avg_bid_ratio = (recent_spreads['BID'] / recent_spreads['AVG_PRC1']).mean()
                            avg_ask_ratio = (recent_spreads['ASK'] / recent_spreads['AVG_PRC1']).mean()
                            
                            if pd.isna(row['BID']) and not np.isnan(avg_bid_ratio):
                                df.loc[idx, 'BID'] = row['AVG_PRC1'] * avg_bid_ratio
                            if pd.isna(row['ASK']) and not np.isnan(avg_ask_ratio):
                                df.loc[idx, 'ASK'] = row['AVG_PRC1'] * avg_ask_ratio
                        else:
                            # Fallback: use simple spread
                            if pd.isna(row['BID']):
                                df.loc[idx, 'BID'] = row['AVG_PRC1'] * 0.98  # 2% below avg
                            if pd.isna(row['ASK']):
                                df.loc[idx, 'ASK'] = row['AVG_PRC1'] * 1.02  # 2% above avg
        
        return df
    
    def create_auction_features(self, df):
        """Create features that capture the auction schedule irregularity"""
        df = df.copy()
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Days since last auction (actual feature, not artifact)
        df['Days_Since_Last_Auction'] = df['Date'].diff().dt.days
        df['Days_Since_Last_Auction'].fillna(0, inplace=True)
        
        # Is this a regular auction day?
        regular_days = ['Tuesday', 'Thursday']  # Based on pattern analysis
        df['Is_Regular_Auction_Day'] = df['Day of Week'].isin(regular_days).astype(int)
        
        # Auction frequency in the last month (rolling count)
        df['Recent_Auction_Frequency'] = df.rolling(window=30, min_periods=1)['Auction Price'].count()
        
        # Volume regime (detect structural changes)
        # if 'VOL_DEC' in df.columns:
        #     df['Volume_Regime'] = pd.cut(df['VOL_DEC'], 
        #                                bins=[0, 1000000, 2000000, 3000000, float('inf')],
        #                                labels=['Very_Low', 'Low', 'Medium', 'High'])
        
        return df
    
    def handle_outliers_contextually(self, df):
        """Handle outliers based on auction context"""
        df = df.copy()
        
        # Handle extreme BID_LOW_1 values (like 0.01, 1.12)
        for col in ['BID_LOW_1', 'BID_HIGH_1']:
            if col in df.columns:
                # Calculate reasonable bounds based on auction prices
                median_auction = df['Auction Price'].median()
                
                # Flag extreme values (more than 50% away from typical auction prices)
                extreme_low = df[col] < (median_auction * 0.5)
                extreme_high = df[col] > (median_auction * 2.0)
                
                # For extreme values, use more conservative estimates
                df.loc[extreme_low, col] = df.loc[extreme_low, 'Auction Price'] * 0.9
                df.loc[extreme_high, col] = df.loc[extreme_high, 'Auction Price'] * 1.1
        
        return df
    
    def preprocess_auction_data(self, df):
        """Main preprocessing pipeline"""
        print("🔍 Starting Smart Auction Preprocessing...")
        
        # Step 1: Basic data preparation
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Add Day of Week if not present
        if 'Day of Week' not in df.columns:
            df['Day of Week'] = df['Date'].dt.day_name()
        
        # Step 2: Analyze patterns (don't artificially create daily data!)
        day_patterns, gap_analysis = self.analyze_auction_pattern(df)
        
        # Step 3: Identify auction types
        df = self.identify_auction_types(df)
        
        # Step 4: Smart missing value handling
        df = self.smart_missing_value_handling(df)
        
        # Step 5: Handle outliers contextually
        df = self.handle_outliers_contextually(df)
        
        # Step 6: Create meaningful features
        df = self.create_auction_features(df)
        
        # Step 7: Keep only actual auction dates (no artificial daily data!)
        df_auctions_only = df.dropna(subset=['Auction Price']).copy()
        
        print(f"✅ Preprocessing complete!")
        print(f"📊 Original rows: {len(df)}")
        print(f"📊 Valid auction rows: {len(df_auctions_only)}")
        print(f"📊 Data range: {df_auctions_only['Date'].min()} to {df_auctions_only['Date'].max()}")
        
        return df_auctions_only



# MODIFIED: WindowGenerator that works with irregular time series


# Integration instructions for your existing code
"""
TO INTEGRATE INTO YOUR EXISTING APP:

1. Replace your load_and_preprocess_data function in app.py:
   
   @st.cache_data
   def load_and_preprocess_data():
       return load_and_preprocess_data_smart()

2. If you get windowing errors, replace WindowGenerator import:
   
   from utils.windowgenerator import WindowGenerator
   # Replace with:
   # WindowGenerator = IrregularTimeSeriesWindowGenerator

3. Update your train/test split dates in dataset.py to match your actual data range:
   
   # Check your data range first:
   print(f"Data range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
   
   # Then adjust the dates in train_test_data method accordingly
"""