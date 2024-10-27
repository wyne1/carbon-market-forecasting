from pymongo import MongoClient
import numpy as np

def get_stored_predictions(collection):
    stored_predictions = list(collection.find().sort("date", -1).limit(10))
    return stored_predictions

def setup_mongodb_connection():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['carbon_market_predictions']
    collection = db['recent_predictions']
    return collection

def save_recent_predictions(collection, recent_preds_orig, preprocessor):
    recent_preds = recent_preds_orig.copy()
    prediction_date = recent_preds.index[0]
    
    auc_price_predictions = (recent_preds['Auc Price'] * preprocessor.train_std['Auc Price']) + preprocessor.train_mean['Auc Price']
    
    pred_diff = np.mean(auc_price_predictions.iloc[1:].values) - auc_price_predictions.iloc[1]
    trade_direction = 'Buy' if pred_diff > 0 else 'Sell'
    
    document = {
        'date': prediction_date,
        'predictions': auc_price_predictions.tolist(),
        'trade_direction': trade_direction,
        'pred_diff': float(pred_diff)
    }
    
    collection.update_one({'date': prediction_date}, {'$set': document}, upsert=True)
    return f"Predictions for {prediction_date} saved successfully. Trade direction: {trade_direction}"