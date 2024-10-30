import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
import datetime
from utils.dataset import MarketData, DataPreprocessor
from utils.windowgenerator import WindowGenerator, compile_and_fit
from utils.data_processing import prepare_data

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

def train_model(model, train_df, val_df, test_df, preprocessor, max_epochs=40):
    OUT_STEPS = 7
    INPUT_STEPS = 7
    multi_window = WindowGenerator(input_width=INPUT_STEPS,
                                   train_df=train_df, val_df=val_df, test_df=test_df,
                                   label_width=OUT_STEPS,
                                   shift=OUT_STEPS)

    history = preprocessor.compile_and_fit(model, multi_window, use_early_stopping=True, max_epochs=max_epochs)
    return history

# def train_ensemble_models(merged_df, num_models=3):
#     ensemble_predictions = []
    
#     for i in range(num_models):
#         train_df, test_df, val_df, preprocessor = prepare_data(merged_df)
#         num_features = len(test_df.columns)
#         model = create_model(num_features, out_steps=7)
#         history = train_model(model, train_df, val_df, test_df, preprocessor, max_epochs=10)
#         _, recent_preds, trend = generate_model_predictions(model, test_df)
#         ensemble_predictions.append((recent_preds, trend))
    
#     return ensemble_predictions, preprocessor, test_df

def train_ensemble_models(merged_df, num_models=3, max_epochs=40):
    for i in range(num_models):
        train_df, test_df, val_df, preprocessor = prepare_data(merged_df)
        num_features = len(test_df.columns)
        model = create_model(num_features, out_steps=7)
        history = train_model(model, train_df, val_df, test_df, preprocessor, max_epochs=max_epochs)
        _, recent_preds, trend = generate_model_predictions(model, test_df)
        yield recent_preds, trend, preprocessor, test_df

def generate_predictions(model, test_df, input_width, out_steps):
    features = test_df.columns
    num_features = len(features)
    predictions = []

    for idx, i in enumerate(range(input_width, len(test_df) - out_steps + 1, out_steps)):
        try:
            inputs = test_df[i - input_width:i].values
            inputs_reshaped = inputs.reshape((1, input_width, num_features))
            preds = model.predict(inputs_reshaped)
            predictions.append(preds[0])
        except Exception as e:
            print(f"Prediction error at index {i}: {e}")
            break

    predictions = np.concatenate(predictions, axis=0)
    pred_indices = test_df.index[input_width:input_width + len(predictions)]
    predictions_df = pd.DataFrame(predictions, columns=features, index=pred_indices)

    return predictions_df

def generate_recent_predictions(model, test_df, input_width, out_steps):
    features = test_df.columns
    num_features = len(features)
    inputs = test_df[-input_width:].values
    inputs_reshaped = inputs.reshape((1, input_width, num_features))
    preds = model.predict(inputs_reshaped)
    
    predictions = []
    predictions.append(preds[0])
    predictions = np.concatenate(predictions, axis=0)

    start_date = test_df.index[-1] + datetime.timedelta(days=1)
    date_range = pd.date_range(start=start_date, periods=out_steps)
    recent_preds = pd.DataFrame(predictions, index=date_range, columns=test_df.columns)
    
    trend = check_gradient(recent_preds['Auc Price'])
    recent_preds = pd.concat([test_df.iloc[[-1]], recent_preds])

    return recent_preds, trend

def check_gradient(values):
    gradient = np.gradient(values)
    return 'positive' if np.all(gradient > 0) else 'negative'


def generate_model_predictions(model, test_df):
    INPUT_STEPS = 7
    OUT_STEPS = 7
    predictions_df = generate_predictions(model, test_df, INPUT_STEPS, OUT_STEPS)
    recent_preds, trend = generate_recent_predictions(model, test_df, INPUT_STEPS, OUT_STEPS)
    return predictions_df, recent_preds, trend

if __name__ == "__main__":
    model, preprocessor, test_df = train_model()
    print("Model training completed.")