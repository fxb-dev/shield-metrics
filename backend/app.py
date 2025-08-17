from flask import Flask, jsonify, request
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import logging
import os
import pandas as pd
from ml_service import MLPredictor

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ML predictor
ml_predictor = MLPredictor()

# Initialize scheduler for data pipeline
scheduler = BackgroundScheduler()

def update_model_task():
    """Task to update model with latest data"""
    logger.info("Running scheduled model update...")
    success = ml_predictor.update_model()
    if success:
        ml_predictor.save_model("model.pkl")
        logger.info("Model update completed successfully")
    else:
        logger.error("Model update failed")

# Schedule model updates
scheduler.add_job(
    func=update_model_task,
    trigger="cron",
    hour=6,  # Run at 6 AM daily
    minute=0,
    id="daily_model_update",
    replace_existing=True
)

# Also schedule a more frequent update during market hours (optional)
scheduler.add_job(
    func=update_model_task,
    trigger="cron",
    day_of_week="mon-fri",
    hour="9-16",
    minute="*/30",  # Every 30 minutes during market hours
    id="market_hours_update",
    replace_existing=True
)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "shield-metrics-api"})

@app.route('/api/info', methods=['GET'])
def get_info():
    return jsonify({
        "name": "Shield Metrics",
        "version": "0.1.0",
        "description": "Tool for wise DCA investing on SP500 - AI boosted"
    })

@app.route('/api/prediction/current', methods=['GET'])
def get_current_prediction():
    """Get the current 5-day prediction"""
    try:
        prediction = ml_predictor.get_current_prediction()
        if prediction:
            return jsonify(prediction)
        else:
            return jsonify({
                "error": "No prediction available",
                "message": "Model may be updating or data unavailable"
            }), 503
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/prediction/update', methods=['POST'])
def trigger_model_update():
    """Manually trigger model update"""
    try:
        success = ml_predictor.update_model()
        if success:
            ml_predictor.save_model("model.pkl")
            return jsonify({
                "status": "success",
                "message": "Model updated successfully",
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "status": "failed",
                "message": "Model update failed"
            }), 500
    except Exception as e:
        logger.error(f"Error updating model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/model/status', methods=['GET'])
def get_model_status():
    """Get model status and metadata"""
    try:
        has_model = ml_predictor.model is not None
        return jsonify({
            "model_loaded": has_model,
            "last_update": ml_predictor.last_update.isoformat() if ml_predictor.last_update else None,
            "confidence_threshold": ml_predictor.confidence_threshold,
            "horizon_days": ml_predictor.horizon,
            "features_count": len(ml_predictor.feature_cols)
        })
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/historical', methods=['GET'])
def get_historical_data():
    """Get historical predictions and prices"""
    try:
        if ml_predictor.cached_data is None:
            return jsonify({"error": "No data available"}), 503
        
        # Get last 30 days of data
        df = ml_predictor.cached_data.tail(30)
        
        historical = []
        for idx, row in df.iterrows():
            historical.append({
                "date": idx.strftime("%Y-%m-%d"),
                "price": float(row["adj_close"]) if not pd.isna(row["adj_close"]) else None,
                "rsi": float(row["RSI_14"]) if "RSI_14" in row and not pd.isna(row["RSI_14"]) else None,
                "volatility": float(row["rv_21d"]) if "rv_21d" in row and not pd.isna(row["rv_21d"]) else None
            })
        
        return jsonify({"data": historical})
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load existing model if available
    if ml_predictor.load_model("model.pkl"):
        logger.info("Loaded existing model")
    
    # Start scheduler
    scheduler.start()
    
    # Run initial model update
    logger.info("Running initial model update...")
    update_model_task()
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False)  # debug=False with APScheduler
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()