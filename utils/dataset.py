from __future__ import annotations
from pathlib import Path
from typing import Optional, AnyStr, List, Tuple
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.windowgenerator import WindowGenerator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import talib
from statsmodels.tsa.seasonal import seasonal_decompose
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator


def extract_settlement_values(df):
    """
    Extract settlement values based on daylight saving adjusted times
    
    Args:
        df: DataFrame with 'Date' (datetime) and 'Spot Value' columns
    
    Returns:
        DataFrame with daily settlement values
    """
    
    # Ensure Date is datetime
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract date components
    df['DateOnly'] = df['Date'].dt.date
    df['Hour'] = df['Date'].dt.hour
    df['Month'] = df['Date'].dt.month
    
    # Define settlement hours by month groups
    def get_settle_hour(month):
        if month in [11, 12, 1, 2, 3]:  # Nov-Mar
            return 11
        else:  # Apr-Oct
            return 10
    
    settlement_results = []
    
    # Process each trading day
    for date in df['DateOnly'].unique():
        day_data = df[df['DateOnly'] == date].copy()
        
        if day_data.empty:
            continue
            
        # Get settlement hour for this month
        month = day_data['Month'].iloc[0]
        settle_hour = get_settle_hour(month)
        
        # Find the closest time >= settlement hour
        settle_or_after = day_data[day_data['Hour'] >= settle_hour]
        
        if settle_or_after.empty:
            # No data at or after settlement time, use last available value
            last_valid_idx = day_data['Spot Value'].last_valid_index()
            if last_valid_idx is not None:
                settlement_value = day_data.loc[last_valid_idx, 'Spot Value']
                settlement_time = day_data.loc[last_valid_idx, 'Date']
            else:
                # No valid data for this day
                continue
        else:
            # Find the earliest time >= settlement hour
            target_time = settle_or_after['Date'].min()
            target_idx = day_data[day_data['Date'] == target_time].index[0]
            
            # Go backwards from target time to find last non-null value
            before_target = day_data[day_data.index < target_idx]
            
            if before_target.empty:
                # No data before target time, use target time value if not null
                if pd.notna(day_data.loc[target_idx, 'Spot Value']):
                    settlement_value = day_data.loc[target_idx, 'Spot Value']
                    settlement_time = day_data.loc[target_idx, 'Date']
                else:
                    continue
            else:
                # Find last non-null value before target time
                last_valid_idx = before_target['Spot Value'].last_valid_index()
                if last_valid_idx is not None:
                    settlement_value = day_data.loc[last_valid_idx, 'Spot Value']
                    settlement_time = day_data.loc[last_valid_idx, 'Date']
                else:
                    # No valid data before target, try target itself
                    if pd.notna(day_data.loc[target_idx, 'Spot Value']):
                        settlement_value = day_data.loc[target_idx, 'Spot Value']
                        settlement_time = day_data.loc[target_idx, 'Date']
                    else:
                        continue
        
        settlement_results.append({
            'Date': date,
            'Settlement_Time': settlement_time,
            'Settlement_Value': settlement_value,
            'Settlement_Hour_Target': settle_hour,
            'Month': month
        })
    
    # Convert to DataFrame
    settlement_df = pd.DataFrame(settlement_results)
    
    if not settlement_df.empty:
        settlement_df = settlement_df.sort_values('Date').reset_index(drop=True)
        
        # Add some useful info
        settlement_df['Actual_Hour'] = pd.to_datetime(settlement_df['Settlement_Time']).dt.hour
        settlement_df['Actual_Minute'] = pd.to_datetime(settlement_df['Settlement_Time']).dt.minute
        settlement_df['Time_Used'] = pd.to_datetime(settlement_df['Settlement_Time']).dt.strftime('%H:%M')
    
    return settlement_df

class MarketData:
    COT_SHEET_NAME: str = "COT-G362"
    AUCTION_SHEET_NAME: str = "Auction"
    OPTIONS_SHEET_NAME: str = "EUA option-G363"
    TA_SHEET_NAME: str = "TA"
    FUNDAMENTALS_SHEET_NAME: str = "power generation-G355"
    ICE_SHEET_NAME: str = "ICE Value"

    @staticmethod
    def analyze_price_relationships(df):
        # Replace inf values with NaN first
        df['Spot Value'] = df['Spot Value'].replace([np.inf, -np.inf], np.nan)
        df['Close'] = df['Close'].replace([np.inf, -np.inf], np.nan)
        df['Auc Price'] = df['Auc Price'].replace([np.inf, -np.inf], np.nan)


        mask = ~(np.isinf(df['Spot Value']) | np.isinf(df['Close']) | df['Spot Value'].isna() | df['Close'].isna())
        filtered_df = df.loc[mask]

        # Additional check to ensure no zeros in Close column to avoid division by zero
        mask2 = filtered_df['Close'] != 0
        ratio = filtered_df.loc[mask2, 'Spot Value'] / filtered_df.loc[mask2, 'Close']

        # Remove any remaining inf values that may have occurred during division
        ratio = ratio[~np.isinf(ratio)]

        
        # Calculate average differences and ratios
        df['Spot_Close_Diff'] = df['Spot Value'] - df['Close']
        # df['Spot_Close_Ratio'] = df['Spot Value'] / df['Close']
        df['Auc_Spot_Diff'] = df['Auc Price'] - df['Spot Value']
        df['Auc_Spot_Ratio'] = df['Auc Price'] / df['Spot Value']

        
        # Calculate statistics for non-null pairs
        stats = {
            'Spot_Close_Diff_Mean': df['Spot_Close_Diff'].mean(),
            'Spot_Close_Diff_Median': df['Spot_Close_Diff'].median(),
            'Spot_Close_Ratio_Mean': ratio.mean(),
            'Auc_Spot_Diff_Mean': df['Auc_Spot_Diff'].mean(),
            'Auc_Spot_Diff_Median': df['Auc_Spot_Diff'].median(),
            'Auc_Spot_Ratio_Mean': df['Auc_Spot_Ratio'].mean(),
        }
    
        return stats
    
    @classmethod
    def latest(cls, directory: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return cls.version(directory, "latest")

    @classmethod
    def version(cls, directory: Path, version: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        path: Path = directory / f"data_sheet_{version}.xlsx"
        return cls.load_dataset(path)

    @classmethod
    def load_dataset(cls, path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cot_df = cls.load_cot_data(path)
        auction_df = cls.load_auction_data(path)
        options_df = cls.load_options_data(path)
        ta_df = cls.load_ta_data(path)
        fundamentals_df = cls.load_fundamentals_data(path)
        return cot_df, auction_df, options_df, ta_df, fundamentals_df

    @staticmethod
    def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the dataset using historical relationships."""
        stats = MarketData.analyze_price_relationships(df)

        print(f"\n\nCalculated Stats: {stats}\n\n")
        
        filled_df = df.copy()
        
        inf_counts = filled_df.isin([np.inf, -np.inf]).sum()
        print("Number of inf values in each column FILLED DF:")
        print(inf_counts.head(20))

        # Fill Spot Value using Close price
        mask = filled_df['Spot Value'].isna() & filled_df['Close'].notna()
        filled_df.loc[mask, 'Spot Value'] = (
            filled_df.loc[mask, 'Close'] * stats['Spot_Close_Ratio_Mean']
        )

        inf_counts = filled_df.isin([np.inf, -np.inf]).sum()
        print("Number of inf values in each column FILLED DF 1:")
        print(inf_counts.head(20))
        
        # Fill Auc Price using Spot Value
        mask = filled_df['Auc Price'].isna() & filled_df['Spot Value'].notna()
        filled_df.loc[mask, 'Auc Price'] = (
            filled_df.loc[mask, 'Spot Value'] + stats['Auc_Spot_Diff_Mean']
        )

        inf_counts = filled_df.isin([np.inf, -np.inf]).sum()
        print("Number of inf values in each column FILLED DF2:")
        print(inf_counts.head(20))
        
        # Fill Median Price using High and Low
        mask = filled_df['Median Price'].isna()
        filled_df.loc[mask, 'Median Price'] = (
            (filled_df.loc[mask, 'High'] + filled_df.loc[mask, 'Low']) / 2
        )

        inf_counts = filled_df.isin([np.inf, -np.inf]).sum()
        print("Number of inf values in each column FILLED DF3:")
        print(inf_counts.head(20))

        # Calculate derived columns
        filled_df['Median Spot Diff'] = filled_df['Median Price'] - filled_df['Spot Value']
        filled_df['Auction Spot Diff'] = filled_df['Auc Price'] - filled_df['Spot Value'] 
        filled_df['Premium/discount-settle'] = np.where(
            filled_df['Spot Value'] != 0,
            filled_df['Auction Spot Diff'] / filled_df['Spot Value'],
            0  # or another default value like np.nan
        )
        
        inf_counts = filled_df.isin([np.inf, -np.inf]).sum()
        print("Number of inf values in each column FILLED DF4:")
        print(inf_counts.head(20))
        # Fill Cover Ratio
        # First fill NaN values with rolling median
        filled_df['Cover Ratio'] = filled_df['Cover Ratio'].fillna(
            filled_df['Cover Ratio'].rolling(window=30, min_periods=1).median()
        )
        
        return filled_df
    
    @classmethod
    def load_cot_data(cls, path: Path) -> pd.DataFrame:
        cot_df = pd.read_excel(path, sheet_name=cls.COT_SHEET_NAME)
        cot_df['Date'] = pd.to_datetime(cot_df['Date'])
        cot_df = cot_df.sort_values(by='Date').reset_index(drop=True)
        cot_df.columns = ['Date', 'net_speculators', 'spec_long_%', 'spec_short_%','Unnamed: 4', 'average OI', 'Z net', 'Z long', 'Z short', 'Z ave OI','Unnamed: 10', 'Long/Short', 'longshort Z', '1M Trend', '3M Trend','6M Trend', '1M Trend.1', '3M Trend.1', '6M Trend.1', '1M Trend.2','3M Trend.2', '6M Trend.2', 'Unnamed: 22', 'Unnamed: 23', 'C1', 'C2','C3']
        
        COT_COLUMNS = ['Date', 'net_speculators', 'spec_long_%', 'spec_short_%']
        cot_df = cot_df[COT_COLUMNS]
        cot_df.loc[:, 'Long/Short'] = cot_df['spec_long_%'] / cot_df['spec_short_%']

        return cot_df
    

    @classmethod
    def load_auction_data(cls, path: Path) -> pd.DataFrame:
        """Load and process auction data with ICE data for filling missing values."""

        print("\n\nLOADING AUCTIONS DATA\n")
        auction_df = pd.read_excel(path, sheet_name=cls.AUCTION_SHEET_NAME)
        cols = ['date', 'auction price', 'median price', 'cover ratio', 'Spot.value', 
                'Auction.Spot.diff', 'Median.Spot.diff', 'Premium/discount-settle']
        auction_df = auction_df[cols]
        auction_df.columns = ['Date', 'Auc Price', 'Median Price', 'Cover Ratio', 
                            'Spot Value', 'Auction Spot Diff', 'Median Spot Diff', 
                            'Premium/discount-settle']
        
        # Process dates
        auction_df['Date'] = pd.to_datetime(auction_df['Date'])
        auction_df = auction_df[~auction_df['Date'].isna()]
        auction_df = auction_df.sort_values(by='Date').reset_index(drop=True)
        auction_df = auction_df.set_index('Date').resample('D').mean().reset_index()

        auction_df = auction_df[auction_df['Date'].dt.year >= 2020]
        # Merge with ICE data and fill missing values
        # ice_df = cls.load_ice_data(path)

        print(f"Auction DF: {auction_df.head()} | SHAPE: {auction_df.shape}")
        ice_df = pd.read_excel(path, sheet_name=cls.ICE_SHEET_NAME, skiprows=4)
        ice_df = ice_df[['Unnamed: 11', 'High', 'Open', 'Low', 'Close']][1:]
        ice_df.columns = ['Date', 'High', 'Open', 'Low', 'Close']
        
        ice_df['Date'] = pd.to_datetime(ice_df['Date'])
        ice_df = ice_df[ice_df['Date'].dt.year >= 2000]
        ice_df['Date'] = ice_df['Date'].dt.date
        ice_df['Date'] = pd.to_datetime(ice_df['Date'])
        ice_df = ice_df[1:]


        merged_df = pd.merge(auction_df, ice_df, on='Date', how='outer')
        filled_df = cls.fill_missing_values(merged_df)
        # print(f"Merged DF: {merged_df.head()} | SHAPE: {merged_df.shape}")
        
        # Return only the auction columns after filling
        filled_df = filled_df[auction_df.columns]
        
        # print(f"Filled DF: {filled_df.head()} | SHAPE: {filled_df.shape}")
        # Replace negative Auc Price values with rolling mean
        mask = filled_df['Auc Price'] < 0
        filled_df.loc[mask, 'Auc Price'] = filled_df['Auc Price'].rolling(window=30, min_periods=1).mean()
        

        # print(f"Filled DF: {filled_df.head()} | SHAPE: {filled_df.shape}")
        return filled_df
    
    @classmethod
    def load_ice_data(cls, path: Path) -> pd.DataFrame:
        """Load ICE market data from Excel file."""
        ice_df = pd.read_excel(path, sheet_name=cls.ICE_SHEET_NAME, skiprows=4)
        ice_df = ice_df[['Unnamed: 11', 'High', 'Open', 'Low', 'Close']][1:]
        ice_df.columns = ['Date', 'High', 'Open', 'Low', 'Close']
        
        ice_df['Date'] = pd.to_datetime(ice_df['Date'])
        ice_df = ice_df[ice_df['Date'].dt.year >= 2000]
        ice_df['Date'] = ice_df['Date'].dt.date
        ice_df['Date'] = pd.to_datetime(ice_df['Date'])
        return ice_df[1:]
    @classmethod
    def load_options_data(cls, path: Path) -> pd.DataFrame:
        # eua_df = pd.read_excel(path, sheet_name=cls.OPTIONS_SHEET_NAME)
        # eua_df['Date'] = pd.to_datetime(eua_df['Date'])
        # cols = ['Date', 'Aggregate Put Open Interest  (R1)', 'Aggregate Call Open Interest  (R1)', 'Aggregate Open Interest  (L1)', 'OPTION OI%', 'PUT/CALL OI',
        #         '1M Trend', '3M Trend', '6M Trend', '1M Trend.1', '3M Trend.1', '6M Trend.1']
        # eua_df = eua_df[cols]
        # eua_df.columns = ['Date', 'Put OI', 'Call OI', 'Agg OI', 'Option OI%', 'Put/Call OI', '1M Trend', '3M Trend', '6M Trend', '1M Trend.1', '3M Trend.1', '6M Trend.1']
        # eua_df = eua_df[~eua_df['Date'].isna()]
        # eua_df = eua_df.sort_values(by='Date').reset_index(drop=True)


        eua_options_cols = ['Date', 'Aggregate Put Open Interest  (R1)', 
                       'Aggregate Call Open Interest  (R1)', 
                       'Aggregate Open Interest  (L1)', 'OPTION OI%', 'PUT/CALL OI']
        eua_options = pd.read_excel(path, 
                                sheet_name='EUA option-G363')
        eua_options['Date'] = pd.to_datetime(pd.to_datetime(eua_options['Date']).dt.date)
        eua_options = eua_options[eua_options_cols][eua_options['Date'].dt.year >= 2018].dropna()

        # Load and process pachis delta data
        pachis_delta = pd.read_excel(path, 
                                sheet_name='25Delta')
        
        # Process December data
        dec_cols = ["Date", "Hist Vol", "50D-Hist Vol", "50D", "25D Spread", "butterfly"]
        pachis_delta_dec = pachis_delta[dec_cols].copy()
        pachis_delta_dec.columns = ["Date", "Hist Vol - 1Y", 
                        "iVol/Hist Vol Spread - Dec", 
                        "50 Delta iVol - Dec", 
                        "25Δ Risk Reversal (Call - Put) - Dec", 
                        "Butterfly - Dec"]
        pachis_delta_dec['Date'] = pd.to_datetime(pachis_delta_dec['Date'])
        pachis_delta_dec = pachis_delta_dec[pachis_delta_dec['Date'].dt.year > 2018] 
        for col in ['Hist Vol - 1Y', 'iVol/Hist Vol Spread - Dec', '50 Delta iVol - Dec']:
            pachis_delta_dec.iloc[:, pachis_delta_dec.columns.get_loc(col)] = pachis_delta_dec[col].replace(' ', np.nan).astype(float)
        pachis_delta_dec.iloc[:, pachis_delta_dec.columns.get_loc('iVol/Hist Vol Spread - Dec')] = pachis_delta_dec['iVol/Hist Vol Spread - Dec'].astype(float)
        pachis_delta_dec.iloc[:, pachis_delta_dec.columns.get_loc('50 Delta iVol - Dec')] = pachis_delta_dec['50 Delta iVol - Dec'].astype(float)

        # Process Prompt data
        prompt_cols = ["Date.1", "50D.1", "25D Spread.1", "butterfly.1"]
        pachis_delta_prompt = pachis_delta[prompt_cols].copy()
        pachis_delta_prompt.columns = ["Date", "50 Delta iVol - Prompt",
                               "25Δ Risk Reversal (Call - Put) - Prompt",
                               "Butterfly - Prompt"]
        pachis_delta_prompt['Date'] = pd.to_datetime(pachis_delta_prompt['Date'])
        pachis_delta_prompt = pachis_delta_prompt[pachis_delta_prompt['Date'].dt.year > 2018]
        # First convert any string values to numeric, handling spaces
        pachis_delta_prompt['50 Delta iVol - Prompt'] = pd.to_numeric(pachis_delta_prompt['50 Delta iVol - Prompt'].replace(' ', np.nan), errors='coerce')

        
        # Load and process option time series
        option_ts = pd.read_excel(path, 
                                sheet_name='Option Time series')
        # Create prompt dataframe
        cols = ['Date', 'Call_OI-Prompt', 'Put_OI-Prompt', 'Dec_OI', 'Call/Put', 'Option%']
        option_ts_prompt = option_ts[cols].copy()
        option_ts_prompt.columns = ['Date', 'Call_OI-Prompt', 'Put_OI-Prompt', 'Dec_OI', 'Call/Put-Prompt', 'Option%-Prompt']

        # Create december dataframe
        cols = ['Date', 'Call/Put.1', 'Option%.1']
        option_ts_dec = option_ts[cols].copy()
        option_ts_dec.columns = ['Date', 'Call/Put-Dec', 'Option%-Dec']

        # Merge the two dataframes on Date
        option_ts_combined = pd.merge(option_ts_prompt, option_ts_dec, on='Date', how='outer')
        option_ts_combined = option_ts_combined.sort_values(by='Date', ascending=False)
        option_ts_combined = option_ts_combined[:-74]

        combined_df = eua_options.merge(pachis_delta_dec, on='Date', how='left')\
            .merge(pachis_delta_prompt, on='Date', how='left')\
            .merge(option_ts_combined, on='Date', how='left')
        
        combined_df = combined_df[combined_df['Date'].dt.year>=2023][:-70]

        return combined_df
        # return eua_df

    @classmethod
    def load_ta_data(cls, path: Path) -> pd.DataFrame:
        ta_df = pd.read_excel(path, sheet_name=cls.TA_SHEET_NAME)
        ta_df = ta_df.iloc[2:]
        return ta_df

    @classmethod
    def load_fundamentals_data(cls, path: Path) -> pd.DataFrame:
        fundamentals_df = pd.read_excel(path, sheet_name=cls.FUNDAMENTALS_SHEET_NAME)
        return fundamentals_df

    @classmethod
    def load_ice_data(cls, path: Path) -> pd.DataFrame:
        ice_df = pd.read_excel(path, sheet_name=cls.ICE_SHEET_NAME)
        return ice_df

    @staticmethod
    def merge_auc_cot(auction_df: pd.DataFrame, cot_df: pd.DataFrame) -> pd.DataFrame:
        merged_df = pd.merge_asof(auction_df, cot_df, on='Date', direction='backward')
        merged_df.ffill(inplace=True)
        merged_df.bfill(inplace=True)
        merged_df.drop(["Spot Value", "Median Price"], axis=1, inplace=True)
        merged_df['Pct_Change_Auc_Price'] = merged_df['Auc Price'].pct_change() * 100
        merged_df = merged_df.dropna(subset=['Pct_Change_Auc_Price'])
        merged_df.dropna(inplace=True)
        return merged_df

class DataPreprocessor:
    def __init__(self, features: List[str], label_columns: List[str], input_width: int, label_width: int, shift: int):
        self.features = features
        self.label_columns = label_columns
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.train_mean = None
        self.train_std = None

    def train_test_data(self, merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        FEATURES = [feature for feature in self.features if feature != 'Date']
        # train_df = merged_df[merged_df['Date'] < "2022-06-01"].copy().set_index('Date')[FEATURES].copy()
        # val_df = merged_df[(merged_df['Date'].dt.year == 2022) & (merged_df['Date'].dt.month >= 6)].copy().set_index('Date')[FEATURES].copy()
        # test_df = merged_df[merged_df['Date'].dt.year >= 2023].copy().set_index('Date')[FEATURES].copy()

        # train_df = merged_df[merged_df['Date'] <
        #                 "2024-04-01"].copy().set_index('Date')[FEATURES].copy()
    
        # val_df = merged_df[(merged_df['Date'] >= "2024-01-01") & (merged_df['Date'] < "2024-04-01")].copy().set_index('Date')[FEATURES].copy()
        # # val_df = train_df
        # test_df = merged_df[merged_df['Date'] >= "2024-04-01"].copy().set_index('Date')[FEATURES].copy()

        train_df = merged_df[merged_df['Date'] < "2025-01-01"].copy().set_index('Date')[FEATURES].copy()
        val_df = merged_df[(merged_df['Date'] >= "2025-02-01") & (merged_df['Date'] < "2025-03-01")].copy().set_index('Date')[FEATURES].copy()
        test_df = merged_df[merged_df['Date'] >= "2025-03-01"].copy().set_index('Date')[FEATURES].copy()
        return train_df, test_df, val_df

    def normalize(self, train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.train_mean = train_df.mean()
        self.train_std = train_df.std()

        train_df = (train_df - self.train_mean) / self.train_std
        val_df = (val_df - self.train_mean) / self.train_std
        test_df = (test_df - self.train_mean) / self.train_std
        return train_df, test_df, val_df
    
    def compile_and_fit(self, model: tf.keras.Model, 
                    window: WindowGenerator, 
                    patience: int = 2, 
                    max_epochs: int = 20,
                    use_early_stopping: bool = True):
        callbacks = []
        if use_early_stopping:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
            callbacks.append(early_stopping)
        
        model.compile(loss=tf.keras.losses.MeanSquaredError(), 
                    optimizer=tf.keras.optimizers.Adam(), 
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])
        
        history = model.fit(window.train, epochs=max_epochs, validation_data=window.val, callbacks=callbacks)
        return history

    @staticmethod
    def calculate_rolling_mean(df: pd.DataFrame, column_name: str, window: int) -> pd.Series:
        return df[column_name].rolling(window=window).mean()

    @staticmethod
    def calculate_ema(df: pd.DataFrame, column_name: str, window: int) -> pd.Series:
        return talib.EMA(df[column_name], timeperiod=window)

    @staticmethod
    def MA_features(merged_df: pd.DataFrame) -> pd.DataFrame:
        merged_df['7 MA Premium/discount-settle'] = talib.SMA(merged_df['Premium/discount-settle'], timeperiod=7)
        merged_df['20 MA Premium/discount-settle'] = talib.SMA(merged_df['Premium/discount-settle'], timeperiod=20)
        merged_df['7 EMA Premium/discount-settle'] = talib.EMA(merged_df['Premium/discount-settle'], timeperiod=7)
        merged_df['20 EMA Premium/discount-settle'] = talib.EMA(merged_df['Premium/discount-settle'], timeperiod=20)
        merged_df['7 MA net_speculators'] = talib.SMA(merged_df['net_speculators'], timeperiod=7)
        merged_df['20 MA net_speculators'] = talib.SMA(merged_df['net_speculators'], timeperiod=20)
        merged_df['7 EMA net_speculators'] = talib.EMA(merged_df['net_speculators'], timeperiod=7)
        merged_df['20 EMA net_speculators'] = talib.EMA(merged_df['net_speculators'], timeperiod=20)
        merged_df['7 MA spec_long_%'] = talib.SMA(merged_df['spec_long_%'], timeperiod=7)
        merged_df['20 MA spec_long_%'] = talib.SMA(merged_df['spec_long_%'], timeperiod=20)
        merged_df['7 EMA spec_long_%'] = talib.EMA(merged_df['spec_long_%'], timeperiod=7)
        merged_df['20 EMA spec_long_%'] = talib.EMA(merged_df['spec_long_%'], timeperiod=20)
        return merged_df

    @staticmethod
    def LAG_features(df: pd.DataFrame, column_name: str, lags: List[int]) -> pd.DataFrame:
        for lag in lags:
            df[f'{column_name}_lag_{lag}'] = df[column_name].shift(lag)
        return df

    @staticmethod
    def momentum_and_volatility_features(df: pd.DataFrame, column_name: str, window: int) -> pd.DataFrame:
        df[f'{column_name}_momentum'] = talib.MOM(df[column_name], timeperiod=window)
        df[f'{column_name}_volatility'] = talib.STDDEV(df[column_name], timeperiod=window)
        return df
    

    # def engineer_auction_features(df):
    #     """
    #     Engineer features for auction data, including interactions with 
    #     exogenous macro energy and FX variables relevant to 'Auc Price'.
        
    #     Parameters:
    #         df (pd.DataFrame): Input DataFrame with columns including:
    #             - 'Date', 'Auc Price', 'Median Price', 'Cover Ratio', 'Spot Value', 
    #               'Auction Spot Diff', 'Median Spot Diff', 'Premium/discount-settle',
    #               and the following new features:
    #             - 'TTF Gas Price', 'Brent Crude Price', 'Coal Futures Price', 'EURUSD Mid Price'
                  
    #     Returns:
    #         pd.DataFrame: DataFrame with original and engineered features.
    #     """
    #     merged_df = df.copy()
    #     merged_df['Date'] = pd.to_datetime(merged_df['Date'])
        
    #     # 1. Time-based features
    #     merged_df['DayOfWeek'] = merged_df['Date'].dt.dayofweek
    #     merged_df['Month'] = merged_df['Date'].dt.month
    #     merged_df['Quarter'] = merged_df['Date'].dt.quarter

    #     # 2. Rolling statistics - focus on main auction columns for signal stability
    #     for column in ['Auc Price', 'Median Price', 'Spot Value']:
    #         merged_df[f'{column}_7d_MA'] = merged_df[column].rolling(window=7).mean()
    #         merged_df[f'{column}_30d_MA'] = merged_df[column].rolling(window=30).mean()
    #         merged_df[f'{column}_7d_EMA'] = merged_df[column].ewm(span=7, adjust=False).mean()
    #         merged_df[f'{column}_30d_EMA'] = merged_df[column].ewm(span=30, adjust=False).mean()
        
    #     # 3. Price change features
    #     for column in ['Auc Price', 'Median Price']:
    #         merged_df[f'{column}_pct_change'] = merged_df[column].pct_change()
    #     merged_df['Spot_Value_ROC'] = merged_df['Spot Value'].pct_change()

    #     # 4. Ratio-based features
    #     merged_df['Auc_to_Median_Ratio'] = merged_df['Auc Price'] / merged_df['Median Price']
    #     merged_df['Auc_to_Spot_Ratio'] = merged_df['Auc Price'] / merged_df['Spot Value']

    #     # 5. Volatility indicators
    #     bb_indicator = BollingerBands(close=merged_df['Auc Price'], window=20, window_dev=2)
    #     merged_df['BB_high'] = bb_indicator.bollinger_hband()
    #     merged_df['BB_low'] = bb_indicator.bollinger_lband()

    #     # 6. Trend indicators
    #     merged_df['SMA_5'] = merged_df['Auc Price'].rolling(window=5).mean()
    #     merged_df['SMA_20'] = merged_df['Auc Price'].rolling(window=20).mean()
    #     merged_df['SMA_cross'] = np.where(merged_df['SMA_5'] > merged_df['SMA_20'], 1, 0)

    #     macd = MACD(close=merged_df['Auc Price'])
    #     merged_df['MACD'] = macd.macd()
    #     merged_df['MACD_signal'] = macd.macd_signal()

    #     # 7. Seasonal decomposition (optional, does not overload features)
    #     try:
    #         decomposition = seasonal_decompose(merged_df['Auc Price'], model='additive', period=30)
    #         merged_df['Seasonal'] = decomposition.seasonal
    #         merged_df['Trend'] = decomposition.trend
    #         merged_df['Residual'] = decomposition.resid
    #     except Exception:
    #         print("Warning: Seasonal decomposition failed. Skipping this feature.")

    #     # 8. Lagged features (short and medium horizon)
    #     for column in ['Auc Price', 'Median Price', 'Cover Ratio', 'Spot Value']:
    #         merged_df[f'{column}_lag1'] = merged_df[column].shift(1)
    #         merged_df[f'{column}_lag7'] = merged_df[column].shift(7)
        
    #     # --- New exogenous feature interactions ---
    #     # Focus on physically and financially motivated relationships,
    #     # but do not overload with all possible combinations.

    #     macro_vars = [
    #         'TTF Gas Price', 'Brent Crude Price', 'Coal Futures Price', 'EURUSD Mid Price'
    #     ]
    #     for var in macro_vars:
    #         # Multiplicative interaction (relative pricing pressure)
    #         merged_df[f'Auc_{var}_Interaction'] = merged_df['Auc Price'] * merged_df[var]

    #         # Ratio interaction (pricing relationship, e.g. per-unit cost/hedge/arb)
    #         merged_df[f'Auc_to_{var}_Ratio'] = merged_df['Auc Price'] / merged_df[var]
            
    #         # 7-day rolling correlation (captures shifting short-term relationship, but not lagged)
    #         merged_df[f'Auc_{var}_7d_corr'] = (
    #             merged_df['Auc Price'].rolling(7).corr(merged_df[var])
    #         )
        
    #     # Interactions with key energy price trends
    #     for var in ['TTF Gas Price', 'Brent Crude Price']:
    #         # Difference (spread) feature: important for fundamental arbitrage
    #         merged_df[f'Auc_minus_{var}'] = merged_df['Auc Price'] - merged_df[var]

    #     # Interaction of FX (EURUSD) - log return feature to normalize scale and reflect market response
    #     merged_df['EURUSD_log_return'] = np.log(merged_df['EURUSD Mid Price']).diff()
    #     merged_df['Auc_Price_fx_return_interaction'] = merged_df['Auc Price'] * merged_df['EURUSD_log_return']

    #     # 9. Main domain interactions (retaining only the important ones)
    #     merged_df['Cover_Premium_Interaction'] = merged_df['Cover Ratio'] * merged_df['Premium/discount-settle']
    #     merged_df['Spot_Diff_Interaction'] = merged_df['Auction Spot Diff'] * merged_df['Median Spot Diff']

    #     # 10. Technical indicator
    #     rsi = RSIIndicator(close=merged_df['Auc Price'], window=14)
    #     merged_df['RSI'] = rsi.rsi()

    #     # 11. Difference features
    #     merged_df['Auc_Price_diff1'] = merged_df['Auc Price'].diff()
    #     merged_df['Auc_Price_diff2'] = merged_df['Auc Price'].diff().diff()

    #     # Minimal additive and squared terms to keep feature count lean
    #     merged_df['Auc_Median_Sum'] = merged_df['Auc Price'] + merged_df['Median Price']
    #     merged_df['Auc_Spot_Sum'] = merged_df['Auc Price'] + merged_df['Spot Value']
    #     merged_df['Auc_Price_Squared'] = merged_df['Auc Price'] ** 2

    #     # Handle NaN values from rolling and shift ops
    #     merged_df = merged_df.bfill().ffill()

    #     # (No normalization here; assume done later in pipeline)
    #     return merged_df

    def engineer_auction_features(df):
        """
        Engineer features for auction data.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame with columns ['Date', 'Auc Price', 'Median Price', 
                        'Cover Ratio', 'Spot Value', 'Auction Spot Diff', 'Median Spot Diff', 
                        'Premium/discount-settle']
        
        Returns:
        pd.DataFrame: DataFrame with original and engineered features
        """
        # Create a copy of the input DataFrame to avoid modifying the original
        merged_df = df.copy()
        

        """
        - Missing initial data - need 
        """
        # Ensure 'Date' is in datetime format
        merged_df['Date'] = pd.to_datetime(merged_df['Date'])
        
        # 1. Time-based features
        merged_df['DayOfWeek'] = merged_df['Date'].dt.dayofweek
        merged_df['Month'] = merged_df['Date'].dt.month
        merged_df['Quarter'] = merged_df['Date'].dt.quarter

        # 2. Rolling statistics
        for column in ['Auc Price', 'Median Price', 'Spot Value']:
            merged_df.loc[:, f'{column}_7d_MA'] = merged_df[column].rolling(window=7).mean()
            merged_df.loc[:, f'{column}_30d_MA'] = merged_df[column].rolling(window=30).mean()
            merged_df.loc[:, f'{column}_7d_std'] = merged_df[column].rolling(window=7).std()
            merged_df.loc[:, f'{column}_30d_std'] = merged_df[column].rolling(window=30).std()
            merged_df.loc[:, f'{column}_7d_EMA'] = merged_df[column].ewm(span=7, adjust=False).mean()
            merged_df.loc[:, f'{column}_30d_EMA'] = merged_df[column].ewm(span=30, adjust=False).mean()

        # 3. Price change features
        for column in ['Auc Price', 'Median Price']:
            merged_df[f'{column}_pct_change'] = merged_df[column].pct_change()
        merged_df['Spot_Value_ROC'] = merged_df['Spot Value'].diff() / merged_df['Spot Value'].shift(1)

        # 4. Ratio-based features
        merged_df['Auc_to_Median_Ratio'] = merged_df['Auc Price'] / merged_df['Median Price']
        merged_df['Auc_to_Spot_Ratio'] = merged_df['Auc Price'] / merged_df['Spot Value']
        # 5. Volatility indicators
        bb_indicator = BollingerBands(close=merged_df['Auc Price'], window=20, window_dev=2)
        merged_df['BB_high'] = bb_indicator.bollinger_hband()
        merged_df['BB_low'] = bb_indicator.bollinger_lband()

        # 6. Trend indicators
        merged_df['SMA_5'] = merged_df['Auc Price'].rolling(window=5).mean()
        merged_df['SMA_20'] = merged_df['Auc Price'].rolling(window=20).mean()
        merged_df['SMA_cross'] = np.where(merged_df['SMA_5'] > merged_df['SMA_20'], 1, 0)

        macd = MACD(close=merged_df['Auc Price'])
        merged_df['MACD'] = macd.macd()
        merged_df['MACD_signal'] = macd.macd_signal()

        # 7. Seasonal decomposition (assuming daily data, adjust freq if different)
        try:
            decomposition = seasonal_decompose(merged_df['Auc Price'], model='additive', period=30)
            merged_df['Seasonal'] = decomposition.seasonal
            merged_df['Trend'] = decomposition.trend
            merged_df['Residual'] = decomposition.resid
        except:
            print("Warning: Seasonal decomposition failed. Skipping this feature.")

        # 8. Lagged features
        for column in ['Auc Price', 'Median Price', 'Cover Ratio', 'Spot Value']:
            merged_df[f'{column}_lag1'] = merged_df[column].shift(1)
            merged_df[f'{column}_lag7'] = merged_df[column].shift(7)

        # 9. Interaction features
        merged_df['Cover_Premium_Interaction'] = merged_df['Cover Ratio'] * merged_df['Premium/discount-settle']
        merged_df['Spot_Diff_Interaction'] = merged_df['Auction Spot Diff'] * merged_df['Median Spot Diff']

        # 10. Categorical encodings (assuming no categorical variables in this case)

        # 11. Technical indicators
        rsi = RSIIndicator(close=merged_df['Auc Price'], window=14)
        merged_df['RSI'] = rsi.rsi()

        # 12. External factors (not included as we don't have this data)

        # 13. Frequency-domain features (simplified FFT)
        def fft_feature(series, num_components=3):
            fft_complex = np.fft.fft(series)
            fft_magnitudes = np.abs(fft_complex)[:len(series)//2]
            return fft_magnitudes[:num_components]

        # merged_df['FFT_1'], merged_df['FFT_2'], merged_df['FFT_3'] = zip(*merged_df['Auc Price'].rolling(window=30).apply(fft_feature))

        # 14. Difference features
        merged_df['Auc_Price_diff1'] = merged_df['Auc Price'].diff()
        merged_df['Auc_Price_diff2'] = merged_df['Auc Price'].diff().diff()

        # Multiplicative interactions
        merged_df['Auc_Median_Interaction'] = merged_df['Auc Price'] * merged_df['Median Price']
        merged_df['Auc_Spot_Interaction'] = merged_df['Auc Price'] * merged_df['Spot Value']
        merged_df['Cover_Spot_Interaction'] = merged_df['Cover Ratio'] * merged_df['Spot Value']
        merged_df['AucSpotDiff_MedianSpotDiff_Interaction'] = merged_df['Auction Spot Diff'] * merged_df['Median Spot Diff']
        
        # Additive interactions
        merged_df['Auc_Median_Sum'] = merged_df['Auc Price'] + merged_df['Median Price']
        merged_df['Auc_Spot_Sum'] = merged_df['Auc Price'] + merged_df['Spot Value']
        
        # Ratio interactions
        merged_df['Auc_to_Median_Spot_Ratio'] = merged_df['Auc Price'] / (merged_df['Median Price'] * merged_df['Spot Value'])
        
        # Squared terms (to capture non-linear relationships)
        merged_df['Auc_Price_Squared'] = merged_df['Auc Price'] ** 2
        merged_df['Cover_Ratio_Squared'] = merged_df['Cover Ratio'] ** 2
        
        # Interaction with time-based features
        merged_df['Auc_Price_DayOfWeek_Interaction'] = merged_df['Auc Price'] * merged_df['DayOfWeek']
        merged_df['Auc_Price_Month_Interaction'] = merged_df['Auc Price'] * merged_df['Month']
        
        # Interaction with lagged features
        merged_df['Auc_Price_Lag_Interaction'] = merged_df['Auc Price'] * merged_df['Auc Price_lag1']
        
        # Interaction with technical indicators
        merged_df['Auc_Price_RSI_Interaction'] = merged_df['Auc Price'] * merged_df['RSI']
        
        # Complex interactions
        merged_df['Complex_Interaction_1'] = merged_df['Auc Price'] * merged_df['Cover Ratio'] / merged_df['Spot Value']
        merged_df['Complex_Interaction_2'] = (merged_df['Auc Price'] - merged_df['Median Price']) * merged_df['Cover Ratio']

        # Handle NaN values created by rolling windows and diff operations
        merged_df = merged_df.bfill().ffill()

        # Normalize numerical features
        numerical_columns = merged_df.select_dtypes(include=[np.number]).columns
        # merged_df[numerical_columns] = (merged_df[numerical_columns] - merged_df[numerical_columns].mean()) / merged_df[numerical_columns].std()

        return merged_df


class Plotting:
    @staticmethod
    def plot_history(history: tf.keras.callbacks.History):
        """
        Plots the training and validation loss over epochs.

        Parameters:
        history (tf.keras.callbacks.History): The history object returned by the fit method of a Keras model.
        """
        history_df = pd.DataFrame(history.history)
        plt.figure(figsize=(10, 6))
        plt.plot(history_df.index, history_df['loss'], label='Training Loss')
        plt.plot(history_df.index, history_df['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()
# Example usage
if __name__ == "__main__":
    from pathlib import Path
    from utils.dataset import MarketData

    # Load data
    cot_df, auction_df, eua_df, ta_df, fundamentals_df = MarketData.latest(Path('data'))
    cot_df = cot_df.set_index('Date').resample('W', origin='end').mean().reset_index()
    auction_df = auction_df.set_index('Date').resample('D').mean().reset_index()

    # Merge and preprocess data
    merged_df = MarketData.merge_auc_cot(auction_df, cot_df)
    merged_df = DataPreprocessor.MA_features(merged_df)
    merged_df = merged_df.dropna()

    # Define features and labels
    FEATURES = merged_df.columns.tolist()
    LABEL_COLS = ['Auc Price']

    # Initialize DataPreprocessor
    preprocessor = DataPreprocessor(features=FEATURES, label_columns=LABEL_COLS, input_width=7, label_width=7, shift=1)

    # Prepare train, test, and validation data
    train_df, test_df, val_df = preprocessor.train_test_data(merged_df)
    train_df, test_df, val_df = preprocessor.normalize(train_df, test_df, val_df)

    # Define window generator
    wide_window = WindowGenerator(
        input_width=7, label_width=7, shift=1,
        label_columns=LABEL_COLS,
        train_df=train_df,
        test_df=test_df,
        val_df=val_df
    )

    # Define Residual LSTM model
    class ResidualWrapper(tf.keras.Model):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def call(self, inputs, *args, **kwargs):
            delta = self.model(inputs, *args, **kwargs)
            return inputs + delta

    residual_lstm = ResidualWrapper(
        tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dense(len(FEATURES), kernel_initializer=tf.initializers.zeros())
        ])
    )

    # Compile and fit the model
    history = preprocessor.compile_and_fit(residual_lstm, wide_window)
