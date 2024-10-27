import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path

from utils.dataset import MarketData, DataPreprocessor
from utils.windowgenerator import WindowGenerator, compile_and_fit

def load_and_preprocess_data():
    cot_df, auction_df, eua_df, ta_df, fundamentals_df = MarketData.latest(Path('data'))
    cot_df = cot_df.set_index('Date').resample('W', origin='end').mean().reset_index()
    auction_df = auction_df.set_index('Date').resample('D').mean().reset_index()

    auction_df = auction_df[7:]
    auction_df.loc[:, 'Premium/discount-settle'] = auction_df['Premium/discount-settle'].ffill()
    auction_df.loc[:, ['Auc Price', 'Median Price', 'Cover Ratio', 'Spot Value',
    'Auction Spot Diff', 'Median Spot Diff', 'Premium/discount-settle']] = auction_df[['Auc Price', 'Median Price', 'Cover Ratio', 'Spot Value', 
                                                                                              'Auction Spot Diff', 'Median Spot Diff', 'Premium/discount-settle']].ffill()

    merged_df = DataPreprocessor.engineer_auction_features(auction_df)
    return merged_df

def create_model(num_features, out_steps):
    CONV_WIDTH = 3
    multi_conv_model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
        tf.keras.layers.Dense(out_steps*num_features,
                              kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([out_steps, num_features])
    ])
    return multi_conv_model

def train_model(model, train_df, val_df, test_df, preprocessor):
    OUT_STEPS = 7
    INPUT_STEPS = 7
    multi_window = WindowGenerator(input_width=INPUT_STEPS,
                                   train_df=train_df, val_df=val_df, test_df=test_df,
                                   label_width=OUT_STEPS,
                                   shift=OUT_STEPS)

    history = preprocessor.compile_and_fit(model, multi_window, use_early_stopping=True, max_epochs=40)
    return history

if __name__ == "__main__":
    model, preprocessor, test_df = train_model()
    print("Model training completed.")