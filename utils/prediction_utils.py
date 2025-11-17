import numpy as np
import pandas as pd
import datetime

def generate_predictions(model, test_df, input_width, out_steps):
    features = test_df.columns
    num_features = len(features)
    predictions = []

    for idx, i in enumerate(range(input_width, len(test_df) - out_steps + 1, out_steps)):
        try:
            inputs = test_df[i - input_width:i].values.astype(np.float32)
            inputs_reshaped = np.array(inputs).reshape((1, input_width, num_features)).astype(np.float32)
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
    inputs = test_df[-input_width:].values.astype(np.float32)
    inputs_reshaped = np.array(inputs).reshape((1, input_width, num_features)).astype(np.float32)
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