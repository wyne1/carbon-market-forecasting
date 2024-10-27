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


def prepare_data(merged_df):
    FEATURES = merged_df.columns.tolist()
    LABEL_COLS = ['Auc Price']

    preprocessor = DataPreprocessor(features=FEATURES, label_columns=LABEL_COLS, input_width=7, label_width=7, shift=1)
    train_df, test_df, val_df = preprocessor.train_test_data(merged_df)
    train_df, test_df, val_df = preprocessor.normalize(train_df, test_df, val_df)
    return train_df, test_df, val_df, preprocessor

class MarketData:
    COT_SHEET_NAME: str = "COT-G362"
    AUCTION_SHEET_NAME: str = "Auction"
    OPTIONS_SHEET_NAME: str = "EUA option-G363"
    TA_SHEET_NAME: str = "TA"
    FUNDAMENTALS_SHEET_NAME: str = "power generation-G355"
    ICE_SHEET_NAME: str = "ICE"

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
        eua_df = cls.load_options_data(path)
        ta_df = cls.load_ta_data(path)
        fundamentals_df = cls.load_fundamentals_data(path)
        return cot_df, auction_df, eua_df, ta_df, fundamentals_df

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
        auction_df = pd.read_excel(path, sheet_name=cls.AUCTION_SHEET_NAME)
        cols = ['date', 'auction price', 'median price', 'cover ratio', 'Spot.value', 'Auction.Spot.diff', 'Median.Spot.diff', 'Premium/discount-settle']
        auction_df = auction_df[cols]
        auction_df.columns = ['Date', 'Auc Price', 'Median Price', 'Cover Ratio', 'Spot Value', 'Auction Spot Diff', 'Median Spot Diff', 'Premium/discount-settle']
        auction_df['Date'] = pd.to_datetime(auction_df['Date'])
        auction_df = auction_df[~auction_df['Date'].isna()]
        auction_df = auction_df.sort_values(by='Date').reset_index(drop=True)
        return auction_df

    @classmethod
    def load_options_data(cls, path: Path) -> pd.DataFrame:
        eua_df = pd.read_excel(path, sheet_name=cls.OPTIONS_SHEET_NAME)
        eua_df['Date'] = pd.to_datetime(eua_df['Date'])
        cols = ['Date', 'Aggregate Put Open Interest  (R1)', 'Aggregate Call Open Interest  (R1)', 'Aggregate Open Interest  (L1)', 'OPTION OI%', 'PUT/CALL OI',
                '1M Trend', '3M Trend', '6M Trend', '1M Trend.1', '3M Trend.1', '6M Trend.1']
        eua_df = eua_df[cols]
        eua_df.columns = ['Date', 'Put OI', 'Call OI', 'Agg OI', 'Option OI%', 'Put/Call OI', '1M Trend', '3M Trend', '6M Trend', '1M Trend.1', '3M Trend.1', '6M Trend.1']
        eua_df = eua_df[~eua_df['Date'].isna()]
        eua_df = eua_df.sort_values(by='Date').reset_index(drop=True)
        return eua_df

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

        train_df = merged_df[merged_df['Date'] <
                        "2024-04-01"].copy().set_index('Date')[FEATURES].copy()
    
        val_df = merged_df[(merged_df['Date'] >= "2024-01-01") & (merged_df['Date'] < "2024-04-01")].copy().set_index('Date')[FEATURES].copy()
        # val_df = train_df
        test_df = merged_df[merged_df['Date'] >= "2024-04-01"].copy().set_index('Date')[FEATURES].copy()
        return train_df, test_df, val_df

    def normalize(self, train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.train_mean = train_df.mean()
        self.train_std = train_df.std()

        print(f"Train DF: {train_df.head()}")
        print(f"SELF.Train_std: {self.train_std['Auc Price']} | SELF.train_mean: {self.train_mean['Auc Price']}")
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

    # Example usage:
    # engineered_df = engineer_auction_features(original_df)
    # print(engineered_df.columns)

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
