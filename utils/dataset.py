from __future__ import annotations

from pathlib import Path
from typing import Optional, AnyStr, List
import pandas as pd


class MarketData:
    COT_SHEET_NAME: str = "COT-G362"
    AUCTION_SHEET_NAME: str = "Auction"
    OPTIONS_SHEET_NAME: str = "EUA option-G363"
    TA_SHEET_NAME: str = "TA"
    FUNDAMENTALS_SHEET_NAME: str = "power generation-G355"
    ICE_SHEET_NAME: str = "ICE"

    @classmethod
    def latest(cls, directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return cls.version(directory, "2")

    @classmethod
    def version(cls, directory: Path, version: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        path: Path = directory / f"data_sheet{version}.xlsx"
        return cls.load_dataset(path)

    @classmethod
    def load_dataset(cls, path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

# Example usage:
# directory = Path("/path/to/directory")
# cot_df = MarketData.load_cot_data(directory / "MarketData - latest.xlsx")
# auction_df = MarketData.load_auction_data(directory / "MarketData - latest.xlsx")
# eua_df = MarketData.load_options_data(directory / "MarketData - latest.xlsx")
# ta_df = MarketData.load_ta_data(directory / "MarketData - latest.xlsx")
# fundamentals_df = MarketData.load_fundamentals_data(directory / "MarketData - latest.xlsx")
# ice_df = MarketData.load_ice_data(directory / "MarketData - latest.xlsx")