"""
ML Service for Shield Metrics
Triple-Barrier Strategy with XGBoost for 5-day predictions
"""

import os
import json
import pickle
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sklearn.metrics import balanced_accuracy_score
from collections import Counter
from xgboost import XGBClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class MLPredictor:
    def __init__(self):
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            logger.warning("Alpha Vantage API key not found")
        
        self.api_url = "https://www.alphavantage.co/query?"
        self.symbol_vuaa = "VUAA.FRK"
        self.symbol_vixy = "VIXY"
        self.symbol_uup = "UUP"
        
        # Model parameters
        self.horizon = 5
        self.k_up = 2.0
        self.k_dn = 2.0
        self.neutral_coef = 0.35
        self.confidence_threshold = 0.60
        
        self.model = None
        self.last_update = None
        self.cached_data = None
        self.cached_predictions = None
        self.trained_features = None
        
        # Feature columns
        self.feature_cols = [
            "ret_5d", "ret_20d", "ret_60d",
            "ma10_ratio", "ma50_ratio", "ma200_ratio",
            "RSI_14", "MACD_12_26_9", "MACD_signal", "MACD_hist",
            "rv_5d", "rv_21d",
            "lvl_VIX_proxy", "chg_VIX_5d_proxy",
            "UST10Y", "UST3M", "term_spread_10Y_3M", "FEDFUNDS", "CPI_yoy", "UNRATE",
            "WTI_close", "WTI_ret_20d",
            "UUP_adj", "DXY_proxy_ret_20d",
        ]
    
    def align_asof(self, series, target_index, allow_exact_matches=False):
        """Align series to target_index using last observation"""
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        right = (series.rename("value").sort_index().reset_index()
                .rename(columns={"index": "date"}))
        left = pd.DataFrame({"date": pd.to_datetime(target_index)}).sort_values("date")
        aligned = pd.merge_asof(left, right, on="date", direction="backward",
                               allow_exact_matches=allow_exact_matches)
        return pd.Series(aligned["value"].values, index=target_index, name=series.name)
    
    def fetch_data(self, symbol, function="TIME_SERIES_DAILY_ADJUSTED"):
        """Fetch data from Alpha Vantage"""
        try:
            url = f"{self.api_url}function={function}&symbol={symbol}&apikey={self.api_key}&outputsize=full"
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def get_ts_daily_adjusted(self, symbol):
        """Get daily adjusted time series"""
        js = self.fetch_data(symbol)
        if not js or "Time Series (Daily)" not in js:
            return None
        
        df = pd.DataFrame.from_dict(js["Time Series (Daily)"], orient="index")
        df.rename(columns={
            "1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close",
            "5. adjusted close": "adj_close", "6. volume": "volume",
            "7. dividend amount": "dividend", "8. split coefficient": "split_coefficient"
        }, inplace=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    
    def prepare_features(self, df, df_vix, macro_data):
        """Calculate all features for the model"""
        # Adjust OHLC
        adj_factor = df["adj_close"] / df["close"]
        df["adj_high"] = df["high"] * adj_factor
        df["adj_low"] = df["low"] * adj_factor
        
        p = df["adj_close"]
        delta = p.diff()
        
        # Returns
        df["ret_5d"] = p / p.shift(5) - 1
        df["ret_20d"] = p / p.shift(20) - 1
        df["ret_60d"] = p / p.shift(60) - 1
        
        # Moving averages
        df["ma10_ratio"] = p / p.rolling(10).mean().shift(1)
        df["ma50_ratio"] = p / p.rolling(50).mean().shift(1)
        df["ma200_ratio"] = p / p.rolling(200).mean().shift(1)
        
        # RSI
        up = delta.clip(lower=0)
        down = (-delta).clip(lower=0)
        avg_up = up.rolling(14, min_periods=14).mean()
        avg_down = down.rolling(14, min_periods=14).mean()
        rs = avg_up / avg_down
        df["RSI_14"] = (100 - 100 / (1 + rs)).shift(1)
        
        # MACD
        ema12 = p.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = p.ewm(span=26, adjust=False, min_periods=26).mean()
        macd_raw = ema12 - ema26
        macd_sig = macd_raw.ewm(span=9, adjust=False, min_periods=9).mean()
        df["MACD_12_26_9"] = macd_raw.shift(1)
        df["MACD_signal"] = macd_sig.shift(1)
        df["MACD_hist"] = (macd_raw - macd_sig).shift(1)
        
        # Volatility
        df["ret"] = np.log(p / p.shift(1))
        df["rv_5d"] = df["ret"].rolling(5).std().shift(1)
        df["rv_21d"] = df["ret"].rolling(21).std().shift(1)
        
        # VIX proxy
        if df_vix is not None:
            df["lvl_VIX_proxy"] = self.align_asof(df_vix["adj_close"], df.index, allow_exact_matches=False)
            df["chg_VIX_5d_proxy"] = df["lvl_VIX_proxy"].pct_change(5)
        
        # Macro features
        for key, series in macro_data.items():
            if series is not None:
                df[key] = self.align_asof(series, df.index, allow_exact_matches=False)
        
        # Additional calculations
        if "UST10Y" in df.columns and "UST3M" in df.columns:
            df["term_spread_10Y_3M"] = df["UST10Y"] - df["UST3M"]
        
        if "WTI_close" in df.columns:
            df["WTI_ret_20d"] = np.log(df["WTI_close"] / df["WTI_close"].shift(20))
        
        if "UUP_adj" in df.columns:
            df["DXY_proxy_ret_20d"] = np.log(df["UUP_adj"] / df["UUP_adj"].shift(20))
        
        return df
    
    def triple_barrier_labels(self, df):
        """Generate triple-barrier labels"""
        idx = df.index.to_numpy(dtype='datetime64[ns]')
        p = df["adj_close"].to_numpy()
        hi = df["adj_high"].to_numpy()
        lo = df["adj_low"].to_numpy()
        sig = df["rv_5d"].to_numpy()
        
        n = len(df)
        labels = np.full(n, np.nan, dtype=float)
        sqrtH = np.sqrt(self.horizon)
        
        for i in range(n):
            if np.isnan(p[i]) or np.isnan(sig[i]):
                continue
            end = min(i + self.horizon, n - 1)
            
            up_log = self.k_up * sig[i] * sqrtH
            dn_log = self.k_dn * sig[i] * sqrtH
            upper_price = p[i] * np.exp(up_log)
            lower_price = p[i] * np.exp(-dn_log)
            
            touched = False
            for j in range(i + 1, end + 1):
                hit_up = (not np.isnan(hi[j])) and (hi[j] >= upper_price)
                hit_dn = (not np.isnan(lo[j])) and (lo[j] <= lower_price)
                
                if hit_up and hit_dn:
                    ret_log = np.log(df["adj_close"].iloc[j] / p[i])
                    labels[i] = 1.0 if ret_log > 0 else (-1.0 if ret_log < 0 else 0.0)
                    touched = True
                    break
                elif hit_up:
                    labels[i] = 1.0
                    touched = True
                    break
                elif hit_dn:
                    labels[i] = -1.0
                    touched = True
                    break
            
            if not touched:
                ret_log = np.log(df["adj_close"].iloc[end] / p[i])
                thr = self.neutral_coef * sig[i] * sqrtH
                if np.isnan(thr):
                    labels[i] = 0.0
                else:
                    labels[i] = 0.0 if abs(ret_log) <= thr else (1.0 if ret_log > 0 else -1.0)
        
        return pd.Series(labels, index=df.index, name="tb_label")
    
    def train_model(self, df):
        """Train XGBoost model"""
        # Prepare labels
        df["tb_label"] = self.triple_barrier_labels(df)
        tb_label_map = {-1: 0, 0: 1, 1: 2}
        df["tb_enc"] = df["tb_label"].map(tb_label_map).astype("Int64")
        
        # Filter feature columns to only those that exist in df
        available_features = [col for col in self.feature_cols if col in df.columns]
        logger.info(f"Using {len(available_features)} features out of {len(self.feature_cols)}")
        
        # Store the features used for training
        self.trained_features = available_features
        
        # Prepare features
        X_full = df[available_features].copy()
        y_full = df["tb_enc"].copy()
        
        mask_nonan = X_full.notna().all(axis=1) & y_full.notna()
        X_full, y_full = X_full[mask_nonan], y_full[mask_nonan]
        
        # Split
        split_pos = int(len(X_full) * 0.70)
        X_train, y_train = X_full.iloc[:split_pos], y_full.iloc[:split_pos]
        X_test, y_test = X_full.iloc[split_pos:], y_full.iloc[split_pos:]
        
        # Validation split
        val_pos = int(len(X_train) * 0.80)
        X_tr, y_tr = X_train.iloc[:val_pos], y_train.iloc[:val_pos]
        X_val, y_val = X_train.iloc[val_pos:], y_train.iloc[val_pos:]
        
        # Class weights
        cnt = Counter(y_tr)
        n_cls = len(cnt)
        total = sum(cnt.values())
        cls_w = {c: total / (n_cls * cnt[c]) for c in cnt}
        w_tr = y_tr.map(cls_w).values
        
        # Train model
        self.model = XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            random_state=42,
            tree_method="hist",
            n_jobs=0,
            eval_metric="mlogloss"
        )
        
        self.model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = balanced_accuracy_score(y_test, y_pred)
        
        return {
            "accuracy": accuracy,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "class_distribution": dict(Counter(y_train))
        }
    
    def update_model(self):
        """Fetch latest data and retrain model"""
        try:
            logger.info("Starting model update...")
            
            # Fetch main data
            df = self.get_ts_daily_adjusted(self.symbol_vuaa)
            if df is None:
                logger.error("Failed to fetch VUAA data")
                return False
            
            df_vix = self.get_ts_daily_adjusted(self.symbol_vixy)
            df_uup = self.get_ts_daily_adjusted(self.symbol_uup)
            
            # Fetch macro data (simplified for now)
            macro_data = {}
            if df_uup is not None:
                macro_data["UUP_adj"] = df_uup["adj_close"].rename("UUP_adj")
            
            # Add placeholder macro data
            for col in ["UST10Y", "UST3M", "FEDFUNDS", "CPI_yoy", "UNRATE", "WTI_close"]:
                macro_data[col] = None
            
            # Prepare features
            df = self.prepare_features(df, df_vix, macro_data)
            
            # Train model
            metrics = self.train_model(df)
            
            # Cache data
            self.cached_data = df
            self.last_update = datetime.now()
            
            logger.info(f"Model updated successfully. Accuracy: {metrics['accuracy']:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            return False
    
    def get_current_prediction(self):
        """Get prediction for current date"""
        try:
            if self.model is None or self.cached_data is None:
                if not self.update_model():
                    return None
            
            # Use the same features that were used during training
            if self.trained_features is None:
                available_features = [col for col in self.feature_cols if col in self.cached_data.columns]
            else:
                available_features = self.trained_features
            
            # Get latest features
            latest_idx = self.cached_data.index[-1]
            X_current = self.cached_data[available_features].iloc[-1:].copy()
            
            # Check for NaN
            if X_current.isna().any().any():
                logger.warning("Current features contain NaN values")
                return None
            
            # Get prediction
            proba = self.model.predict_proba(X_current)[0]
            pred_class = np.argmax(proba)
            confidence = proba[pred_class]
            
            # Map back to original labels
            tb_inv_map = {0: -1, 1: 0, 2: 1}
            pred_label = tb_inv_map[pred_class]
            
            # Decision based on confidence threshold
            if confidence < self.confidence_threshold or pred_label == 0:
                action = "HOLD"
            elif pred_label == 1:
                action = "BUY"
            else:
                action = "SELL"
            
            return {
                "date": latest_idx.strftime("%Y-%m-%d"),
                "prediction_class": int(pred_label),
                "confidence": float(confidence),
                "probabilities": {
                    "down": float(proba[0]),
                    "neutral": float(proba[1]),
                    "up": float(proba[2])
                },
                "action": action,
                "horizon_days": self.horizon,
                "last_update": self.last_update.isoformat() if self.last_update else None,
                "current_price": float(self.cached_data["adj_close"].iloc[-1]),
                "features": {
                    "RSI_14": float(X_current["RSI_14"].iloc[0]) if not np.isnan(X_current["RSI_14"].iloc[0]) else None,
                    "ma50_ratio": float(X_current["ma50_ratio"].iloc[0]) if not np.isnan(X_current["ma50_ratio"].iloc[0]) else None,
                    "rv_21d": float(X_current["rv_21d"].iloc[0]) if not np.isnan(X_current["rv_21d"].iloc[0]) else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            return None
    
    def save_model(self, path="model.pkl"):
        """Save trained model"""
        if self.model:
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
    
    def load_model(self, path="model.pkl"):
        """Load saved model"""
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.model = pickle.load(f)
                return True
        return False