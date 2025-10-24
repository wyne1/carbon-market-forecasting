import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime, timedelta
import streamlit as st

from utils.smart_preprocessing import SmartAuctionPreprocessor

class LSEGDataLoader:
    """
    Handles data loading from LSEG Data Library for carbon market data
    """
    
    def __init__(self, config_path: str = "../../Configuration"):
        """
        Initialize LSEG data loader
        
        Args:
            config_path: Path to LSEG configuration directory
        """
        self.config_path = config_path
        self.session = None
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize LSEG session"""
        try:
            os.environ["LD_LIB_CONFIG_PATH"] = self.config_path
            import lseg.data as ld
            self.ld = ld
            
            # Open session
            self.session = self.ld.open_session(
                name='desktop.workspace',
                config_name=f"/Users/zeerakwyne/Documents/freelance/Example.DataLibrary.Python/Configuration/lseg-data.config.json"
            )
            st.success("LSEG session initialized successfully")
        except Exception as e:
            st.error(f"Failed to initialize LSEG session: {str(e)}")
            raise

    def _fetch_latest_spot_price(self, target_date: str) -> Optional[Dict]:
        """
        Fetch spot price for a specific date using the settlement value extraction logic
        
        Args:
            target_date: Date in 'YYYY-MM-DD' format
            
        Returns:
            Dict with Date and Spot Value, or None if no data found
        """
        try:
            # Convert to datetime to get the next day for end_time
            date_obj = pd.to_datetime(target_date)
            next_day = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Fetch spot data for the target date
            df_cf = self.ld.get_history(
                universe="ECEFDc1",
                interval="1min",
                start=target_date,
                end=next_day
            ).reset_index()
            
            if df_cf.empty:
                st.warning(f"No spot data found for {target_date}")
                return None
            
            # Rename column to match expected format
            df_cf = df_cf.rename(columns={
                'Timestamp' : 'Date',
                'TRDPRC_1': 'Spot Value'
                },
            )
            
            print(f"DF CF NAMES: {df_cf.columns}")
            
            # Use the settlement extraction logic
            from utils.dataset import extract_settlement_values  # Import your function
            settlement_df = extract_settlement_values(df_cf)
            
            if settlement_df.empty:
                st.warning(f"No settlement value found for {target_date}")
                return None
            

            print(f"SETTLEMENT DF NAMES: {settlement_df.columns}")
            # Get the settlement value for this date
            settlement_row = settlement_df[settlement_df['Date'] == date_obj.date()]
            
            if settlement_row.empty:
                st.warning(f"No settlement value found for {target_date}")
                return None
            
            return {
                'Date': date_obj,
                'Spot Value': settlement_row.iloc[0]['Settlement_Value']
            }
            
        except Exception as e:
            st.error(f"Failed to fetch spot price for {target_date}: {str(e)}")
            return None

    def _update_spot_file(self, spot_file_path: str, new_spot_data: list) -> pd.DataFrame:
        """
        Update the spot price file with new data
        
        Args:
            spot_file_path: Path to the spot price Excel file
            new_spot_data: List of dicts with Date and Spot Value
            
        Returns:
            Updated spot DataFrame
        """
        try:
            # Load existing spot data
            if os.path.exists(spot_file_path):
                existing_spot_df = pd.read_excel(spot_file_path)
                existing_spot_df['Date'] = pd.to_datetime(existing_spot_df['Date'])
            else:
                existing_spot_df = pd.DataFrame(columns=['Date', 'Spot Value'])
            
            # Create DataFrame from new data
            new_spot_df = pd.DataFrame(new_spot_data)
            if not new_spot_df.empty:
                new_spot_df['Date'] = pd.to_datetime(new_spot_df['Date'])
            
            # Combine and remove duplicates (keep latest)
            combined_df = pd.concat([existing_spot_df, new_spot_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
            
            # Save updated file
            combined_df.to_excel(spot_file_path, index=False)
            st.info(f"Updated spot price file with {len(new_spot_data)} new records")
            
            return combined_df
            
        except Exception as e:
            st.error(f"Failed to update spot file: {str(e)}")
            raise

    def _check_and_update_spot_data(self, auction_df: pd.DataFrame, spot_file_path: str) -> pd.DataFrame:
        """
        Check if spot data is up to date with auction data and update if necessary
        
        Args:
            auction_df: DataFrame with auction data
            spot_file_path: Path to the spot price Excel file
            
        Returns:
            Updated spot DataFrame
        """
        try:
            # Get the latest auction date
            auction_df['Date'] = pd.to_datetime(auction_df['Date'])
            latest_auction_date = auction_df['Date'].max()
            
            # Load existing spot data
            if os.path.exists(spot_file_path):
                existing_spot_df = pd.read_excel(spot_file_path)
                existing_spot_df['Date'] = pd.to_datetime(existing_spot_df['Date'])
                latest_spot_date = existing_spot_df['Date'].max()
            else:
                existing_spot_df = pd.DataFrame(columns=['Date', 'Spot Value'])
                latest_spot_date = pd.Timestamp('1900-01-01')  # Very old date
            
            st.info(f"Latest auction date: {latest_auction_date.date()}")
            st.info(f"Latest spot date: {latest_spot_date.date()}")
            
            # Check if spot data needs updating
            if latest_spot_date >= latest_auction_date:
                st.success("Spot data is up to date")
                return existing_spot_df
            
            st.info("Spot data is outdated. Fetching missing data...")
            
            # Find missing dates
            if existing_spot_df.empty:
                # If no existing data, get all unique auction dates
                missing_dates = auction_df['Date'].dt.date.unique()
            else:
                # Get dates that are in auction but not in spot (or newer than latest spot)
                auction_dates = set(auction_df['Date'].dt.date)
                existing_dates = set(existing_spot_df['Date'].dt.date)
                missing_dates = auction_dates - existing_dates
                
                # Also add any dates newer than latest spot date
                newer_dates = auction_df[auction_df['Date'] > latest_spot_date]['Date'].dt.date.unique()
                missing_dates.update(newer_dates)
            
            # Fetch missing spot data
            new_spot_data = []
            for date in sorted(missing_dates):
                date_str = date.strftime('%Y-%m-%d')
                spot_data = self._fetch_latest_spot_price(date_str)
                if spot_data:
                    new_spot_data.append(spot_data)
                    st.info(f"Fetched spot price for {date}: {spot_data['Spot Value']:.2f}")

            
            # Update the spot file
            if new_spot_data:
                updated_spot_df = self._update_spot_file(spot_file_path, new_spot_data)
                return updated_spot_df
            else:
                st.warning("No new spot data could be fetched")
                return existing_spot_df
                
        except Exception as e:
            st.error(f"Failed to check/update spot data: {str(e)}")
            # Return existing data if update fails
            if os.path.exists(spot_file_path):
                return pd.read_excel(spot_file_path)
            else:
                return pd.DataFrame(columns=['Date', 'Spot Value'])

    def load_external_variables(self, start_date: str = None, end_date: str = None, count: int = 5000):
        """
        Load external market variables that influence carbon prices
        
        Returns:
            DataFrame with Date index and columns: Brent_Oil, TTF_Gas, EU_Inflation
        """
        try:
            # 1. Brent Crude Oil (daily)
            st.info("📊 Loading Brent Crude Oil prices...")
            brent_crude = self.ld.get_history(
                universe="LCOc1",
                interval="1D",
                count=count
            ).reset_index()
            
            # 2. TTF Gas (daily)
            st.info("📊 Loading TTF Natural Gas prices...")
            ttf_gas = self.ld.get_history(
                universe="TFMBMc1",
                interval="1D",
                count=count
            ).reset_index()
            
            # 3. EU Inflation (monthly - will need to forward fill)
            st.info("📊 Loading EU Inflation data...")
            eu_inflation = self.ld.get_history(
                universe="EUHICY=ECI",
                interval="1M",
                count=count
            ).reset_index()
            
            # Process and clean the data
            external_df = self._process_external_variables(
                brent_crude, ttf_gas, eu_inflation
            )
            
            st.success(f"✅ Loaded {len(external_df.columns)-1} external variables")
            return external_df
            
        except Exception as e:
            st.error(f"❌ Failed to load external variables: {str(e)}")
            st.warning("⚠️ Continuing without external variables...")
            return pd.DataFrame()
    
    def _process_external_variables(self, brent_df, ttf_df, inflation_df):
        """
        Process and align external variables to daily frequency
        
        Args:
            brent_df: Brent crude oil DataFrame
            ttf_df: TTF gas DataFrame  
            inflation_df: EU inflation DataFrame (monthly)
            
        Returns:
            Cleaned and aligned DataFrame
        """
        # Extract prices using TRDPRC_1 for commodities, VALUE for inflation
        brent_clean = brent_df[['Date', 'TRDPRC_1']].copy()
        brent_clean.columns = ['Date', 'Brent_Oil']
        brent_clean['Date'] = pd.to_datetime(brent_clean['Date'])
        
        ttf_clean = ttf_df[['Date', 'TRDPRC_1']].copy()
        ttf_clean.columns = ['Date', 'TTF_Gas']
        ttf_clean['Date'] = pd.to_datetime(ttf_clean['Date'])
        
        inflation_clean = inflation_df[['Date', 'VALUE']].copy()
        inflation_clean.columns = ['Date', 'EU_Inflation']
        inflation_clean['Date'] = pd.to_datetime(inflation_clean['Date'])
        
        # Merge daily data (Brent + TTF)
        external_df = pd.merge(
            brent_clean, 
            ttf_clean, 
            on='Date', 
            how='outer'
        ).sort_values('Date')
        
        # Add monthly inflation - forward fill to daily
        external_df = pd.merge(
            external_df,
            inflation_clean,
            on='Date',
            how='left'
        ).sort_values('Date')
        
        # Forward fill inflation (monthly → daily)
        external_df['EU_Inflation'] = external_df['EU_Inflation'].fillna(method='ffill')
        
        # Forward fill any remaining missing values in commodities
        external_df['Brent_Oil'] = external_df['Brent_Oil'].fillna(method='ffill').fillna(method='bfill')
        external_df['TTF_Gas'] = external_df['TTF_Gas'].fillna(method='ffill').fillna(method='bfill')
        
        # Log statistics
        st.info(f"""
        External Variables Summary:
        - Date range: {external_df['Date'].min().date()} to {external_df['Date'].max().date()}
        - Total days: {len(external_df)}
        - Brent Oil: ${external_df['Brent_Oil'].mean():.2f} ± {external_df['Brent_Oil'].std():.2f}
        - TTF Gas: €{external_df['TTF_Gas'].mean():.2f} ± {external_df['TTF_Gas'].std():.2f}
        - EU Inflation: {external_df['EU_Inflation'].mean():.2f}% ± {external_df['EU_Inflation'].std():.2f}%
        - Missing values: Brent={external_df['Brent_Oil'].isnull().sum()}, TTF={external_df['TTF_Gas'].isnull().sum()}, Inflation={external_df['EU_Inflation'].isnull().sum()}
        """)
        
        return external_df

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_auction_data(_self, start_date: Optional[str] = None, 
                         end_date: Optional[str] = None, 
                         count: int = 5000,
                         include_external: bool = True) -> pd.DataFrame:
        """
        Load auction data from LSEG, preprocess, and engineer features.
        Automatically updates spot price data if needed.
        
        NEW: Optionally includes external variables (Brent, Gas, Inflation)
        """
        try:
            # 1. Load data from different exchanges
            params = {"interval": "1D", "count": count}
            if start_date and end_date:
                params.update({"start": start_date, "end": end_date})
                params.pop("count", None)

            df_eu = _self.ld.get_history(universe="EEX-EUA4EU-AUC", **params).reset_index()
            df_de = _self.ld.get_history(universe="EEX-EUA4DE-AUC", **params).reset_index()
            df_pl = _self.ld.get_history(universe="EEX-EUA4PL-AUC", **params).reset_index()

            # 2. Merge and clean the raw data
            merged_df = _self._merge_timeseries_concat(df_eu, df_de, df_pl)

            # 3. Rename columns to match the legacy format expected by the preprocessor
            merged_df = merged_df.rename(columns={
                'TRDPRC_1': 'Auction Price',
                'MID_PRICE': 'Median Price',
                'MARGIN_RTO': 'Cover Ratio'
            })

            FILE_NAME = "data/latest_auction.xlsx"
            SPOT_FILE_NAME = "data/historic_spots.xlsx"
            
            # Save auction data
            merged_df.to_excel(FILE_NAME)
            merged_df = pd.read_excel(FILE_NAME).set_index('Unnamed: 0')
            
            # 4. Check and update spot data if necessary
            spot_df = _self._check_and_update_spot_data(merged_df.reset_index(), SPOT_FILE_NAME)
            print(f"SPOT DATAFRAME: {spot_df.tail(10)}")
            # 5. Apply the same smart preprocessing as the original pipeline
            preprocessor = SmartAuctionPreprocessor()
            auction_df = preprocessor.preprocess_auction_data(merged_df)

            # 6. Merge with spot data and calculate differences
            auction_df = auction_df.merge(spot_df, on='Date', how='left')
            auction_df['Auction Spot Diff'] = auction_df['Auction Price'] - auction_df['Spot Value']
            auction_df['Median Spot Diff'] = auction_df['Median Price'] - auction_df['Spot Value']

            auction_df['Premium/discount-settle'] = auction_df['Auction Spot Diff'] / auction_df['Spot Value']
            auction_df = auction_df.rename(columns={'Auction Price': 'Auc Price'})
            
            # 7. NEW: Add external variables if requested
            if include_external:
                st.info("🌍 Loading external market variables...")
                external_df = _self.load_external_variables(count=count)
                
                if not external_df.empty:
                    # Merge external variables with auction data
                    auction_df = auction_df.merge(
                        external_df,
                        on='Date',
                        how='left'
                    )
                    
                    # Forward fill any missing external data
                    external_cols = ['Brent_Oil', 'TTF_Gas', 'EU_Inflation']
                    for col in external_cols:
                        if col in auction_df.columns:
                            auction_df[col] = auction_df[col].fillna(method='ffill').fillna(method='bfill')
                    
                    st.success(f"✅ Added external variables: {', '.join(external_cols)}")
                else:
                    st.warning("⚠️ Proceeding without external variables")
            
            # 8. Apply the final feature engineering step
            from utils.dataset import DataPreprocessor
            final_df = DataPreprocessor.engineer_auction_features(auction_df)

            return final_df
            
        except Exception as e:
            st.error(f"Failed to load auction data: {str(e)}")
            raise
    
    def _merge_timeseries_concat(self, df: pd.DataFrame, 
                                german_df: pd.DataFrame, 
                                pl_df: pd.DataFrame) -> pd.DataFrame:
        """Merge multiple dataframes with same columns by concatenating and sorting by date"""
        # Ensure Date column is datetime in all dataframes
        df['Date'] = pd.to_datetime(df['Date'])
        german_df['Date'] = pd.to_datetime(german_df['Date'])
        pl_df['Date'] = pd.to_datetime(pl_df['Date'])
        
        # Concatenate all dataframes
        merged_df = pd.concat([df, german_df, pl_df], ignore_index=True)
        
        # Sort by Date and remove duplicates
        merged_df = merged_df.sort_values('Date')
        
        # Reset index for clean output
        merged_df = merged_df.reset_index(drop=True)
        
        return merged_df
    
    def close_session(self):
        """Close LSEG session"""
        if self.session:
            self.ld.close_session()