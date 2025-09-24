#!/usr/bin/env python3
"""
BTCUSDT Multi-Model Training Script
Reverse-engineered from prediction.py architecture

Training 10 neural network models with different sequence lengths:
- 6 short-term models (20h): model_a, b, c, d, h, j  
- 1 daily cycle (24h): model_e
- 1 weekly pattern (32h): model_i
- 2 extended trend (40h): model_f, g

Target: 0.5% profit in 5-hour horizon
Features: 30 technical indicators (identical to prediction.py)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import ta
import warnings
warnings.filterwarnings('ignore')
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

# Identik dengan CONFIG prediction.py
TRAINING_CONFIG = {
    'DATA_FILE': "BTCUSDT_BINANCE15.csv",
    'TARGET_THRESHOLD': 0.005,  # 0.5% profit target
    'PREDICTION_HORIZON': 5,    # 5 hours
    'M15_PER_HOUR': 4,         # 4 M15 candles per hour
    'TEST_SIZE': 0.2,          # 20% for testing
    'VALIDATION_SIZE': 0.2,    # 20% of remaining for validation
    'RANDOM_STATE': 42,
    'EPOCHS': 100,
    'BATCH_SIZE': 32,
    'PATIENCE': 15
}

# Konfigurasi model sesuai reverse engineering
MODEL_CONFIGS = {
    'model_a': {'sequence_length': 20, 'description': 'Baseline model - pattern dasar 20 jam'},
    'model_b': {'sequence_length': 20, 'description': 'Short-term momentum - momentum jangka pendek'},
    'model_c': {'sequence_length': 20, 'description': 'Short-term volatility - volatilitas jangka pendek'},
    'model_d': {'sequence_length': 20, 'description': 'Short-term patterns - pola teknikal pendek'},
    'model_e': {'sequence_length': 24, 'description': 'Daily cycle patterns - siklus harian penuh'},
    'model_f': {'sequence_length': 40, 'description': 'Extended trend analysis - analisis trend panjang'},
    'model_g': {'sequence_length': 40, 'description': 'Extended pattern recognition - pengenalan pola panjang'},
    'model_h': {'sequence_length': 20, 'description': 'Short-term reversal - reversal pattern'},
    'model_i': {'sequence_length': 32, 'description': 'Weekly patterns - pola mingguan (4 hari)'},
    'model_j': {'sequence_length': 20, 'description': 'Short-term consolidation - konsolidasi pendek'}
}

class BTCUSDTModelTrainer:
    def __init__(self, data_file: str = None):
        self.data_file = data_file or TRAINING_CONFIG['DATA_FILE']
        self.target_threshold = TRAINING_CONFIG['TARGET_THRESHOLD']
        self.prediction_horizon = TRAINING_CONFIG['PREDICTION_HORIZON']
        
        print("="*80)
        print("BTCUSDT MULTI-MODEL TRAINING SYSTEM")
        print("="*80)
        print(f"Data File: {self.data_file}")
        print(f"Target: {self.target_threshold*100:.1f}% profit in {self.prediction_horizon}h")
        print(f"Models to train: {len(MODEL_CONFIGS)}")
        print("="*80)

    def load_m15_data(self, file_path: str) -> pd.DataFrame:
        """Load M15 data identik dengan prediction.py"""
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            print(f"✅ Data loaded: {len(df):,} M15 candles")
            print(f"   Period: {df.index[0]} to {df.index[-1]}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering IDENTIK dengan prediction.py
        Menghasilkan 30 features yang sama persis
        """
        print("🔧 Extracting features (30 technical indicators)...")
        features = df.copy()

        # 1. ATR 14
        features['atr_14'] = ta.volatility.average_true_range(
            features['high'].shift(1),
            features['low'].shift(1),
            features['close'].shift(1),
            window=14
        )
        
        # 2. Day of week
        features['day_of_week'] = df.index.dayofweek

        # 3. Donchian Channel
        donchian_h = ta.volatility.DonchianChannel(
            features['high'].shift(1),
            features['low'].shift(1),
            features['close'].shift(1),
            window=20
        ).donchian_channel_hband()
        donchian_l = ta.volatility.DonchianChannel(
            features['high'].shift(1),
            features['low'].shift(1),
            features['close'].shift(1),
            window=20
        ).donchian_channel_lband()
        features['donchian_channel_width'] = (donchian_h - donchian_l) / features['close'].shift(1) * 100

        # 4. True Range
        features['true_range'] = ta.volatility.average_true_range(
            features['high'].shift(1),
            features['low'].shift(1),
            features['close'].shift(1),
            window=1
        )
        
        # 5. Range expansion potential
        features['range_expansion_potential'] = features['atr_14'] / features['close'].shift(1) * 100 * 6
        
        # 6. Volume mean 20
        features['volume_mean_20'] = features['volume'].shift(1).rolling(20).mean()

        # 7. Hourly volatility ratio
        hourly_means = {}
        for hour in range(24):
            mask = features.index.hour == hour
            if mask.any():
                hourly_means[hour] = features.loc[mask, 'true_range'].shift(1).rolling(window=20, min_periods=1).mean()

        hourly_vol = pd.Series(index=features.index, dtype=float)
        for idx in features.index:
            hour = idx.hour
            if hour in hourly_means and idx in hourly_means[hour].index:
                hourly_vol.loc[idx] = hourly_means[hour].loc[idx]
            else:
                hourly_vol.loc[idx] = 0

        features['hour_volatility_ratio'] = features['true_range'] / (hourly_vol + 1e-10)
        
        # 8. Range compression
        features['range_compression'] = features['atr_14'] / features['atr_14'].rolling(20).mean()

        # 9. Money flow raw
        typical_price = (features['high'].shift(1) + features['low'].shift(1) + features['close'].shift(1)) / 3
        features['money_flow_raw'] = typical_price * features['volume'].shift(1)

        # 10. TR percentile
        features['tr_percentile'] = features['true_range'].rolling(100, min_periods=1).apply(
            lambda x: np.sum(x.values < x.values[-1]) / len(x) if len(x) > 0 else 0.5
        )

        # 11. Exit urgency
        momentum_3h = features['close'].shift(1) / features['close'].shift(4) - 1
        volume_momentum = (features['volume'].shift(1) / features['volume_mean_20'] - 1) * 100
        features['hours_to_ny_close'] = (21 - features.index.hour) % 24
        time_pressure = 1 - (features['hours_to_ny_close'] / 24)

        features['exit_urgency'] = (
            momentum_3h.abs() * 0.3 +
            (features['true_range'] / features['atr_14']) * 0.3 +
            volume_momentum / 100 * 0.2 +
            time_pressure * 0.2
        )

        # 12-13. Volume stats
        features['volume_std_20'] = features['volume'].shift(1).rolling(20).std()
        volume_zscore = (features['volume'].shift(1) - features['volume_mean_20']) / (features['volume_std_20'] + 1e-10)

        # 14. ADX
        adx = ta.trend.adx(
            features['high'].shift(1),
            features['low'].shift(1),
            features['close'].shift(1)
        )

        # 15. Risk score
        features['risk_score'] = (
            features['tr_percentile'] * 0.3 +
            (1 - features['range_compression']) * 0.3 +
            volume_zscore.clip(0, 3) / 3 * 0.2 +
            (adx / 100) * 0.2
        )

        # 16. Chaikin signal
        chaikin_oscillator = ta.volume.ChaikinMoneyFlowIndicator(
            features['high'].shift(1),
            features['low'].shift(1),
            features['close'].shift(1),
            features['volume'].shift(1),
            window=20
        ).chaikin_money_flow()
        features['chaikin_signal'] = chaikin_oscillator.rolling(5).mean()

        # 17. Volatility squeeze
        bb = ta.volatility.BollingerBands(features['close'].shift(1), window=20, window_dev=2)
        kc = ta.volatility.KeltnerChannel(
            features['high'].shift(1),
            features['low'].shift(1),
            features['close'].shift(1),
            window=20
        )
        features['volatility_squeeze'] = (bb.bollinger_hband() - bb.bollinger_lband()) / (kc.keltner_channel_hband() - kc.keltner_channel_lband())

        # 18-19. Donchian & Keltner width
        features['donchian_width'] = ta.volatility.DonchianChannel(
            features['high'].shift(1),
            features['low'].shift(1),
            features['close'].shift(1)
        ).donchian_channel_wband()

        features['keltner_width'] = ta.volatility.KeltnerChannel(
            features['high'].shift(1),
            features['low'].shift(1),
            features['close'].shift(1)
        ).keltner_channel_wband()

        # 20-23. TP/SL levels Donchian
        donchian_tp_ratio = 0.5
        donchian_sl_ratio = 0.2

        features['tp_level_donchian_long'] = features['close'].shift(1) + (features['donchian_channel_width'] / 100 * features['close'].shift(1) * donchian_tp_ratio)
        features['sl_level_donchian_long'] = features['close'].shift(1) - (features['donchian_channel_width'] / 100 * features['close'].shift(1) * donchian_sl_ratio)
        features['tp_level_donchian_short'] = features['close'].shift(1) - (features['donchian_channel_width'] / 100 * features['close'].shift(1) * donchian_tp_ratio)
        features['sl_level_donchian_short'] = features['close'].shift(1) + (features['donchian_channel_width'] / 100 * features['close'].shift(1) * donchian_sl_ratio)

        # 24-26. Psychological indicators
        features['fear_candle'] = (
            (features['high'].shift(1) - features['close'].shift(1)) > 2 * (features['close'].shift(1) - features['low'].shift(1))
        ).astype(int)

        features['panic_volume'] = (
            (features['volume'].shift(1) > features['volume'].shift(1).rolling(20).mean() * 2) &
            (features['close'].shift(1) < features['open'].shift(1))
        ).astype(int)

        features['greed_candle'] = (
            (features['close'].shift(1) - features['low'].shift(1)) > 2 * (features['high'].shift(1) - features['close'].shift(1))
        ).astype(int)

        # Clean data
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)

        print(f"✅ Features extracted: {features.shape[1]} columns")
        return features

    def create_target(self, df_h1: pd.DataFrame) -> pd.Series:
        """
        Create binary target: 1 if price rises ≥ 0.5% within 5 hours
        Identik dengan logika prediction.py
        """
        print(f"🎯 Creating targets (≥{self.target_threshold*100:.1f}% in {self.prediction_horizon}h)...")
        
        targets = []
        for i in range(len(df_h1) - self.prediction_horizon):
            entry_price = df_h1['close'].iloc[i]
            target_price = entry_price * (1 + self.target_threshold)
            
            # Check future 5 hours
            future_highs = df_h1['high'].iloc[i+1:i+1+self.prediction_horizon]
            max_high = future_highs.max() if len(future_highs) > 0 else 0
            
            hit_target = 1 if max_high >= target_price else 0
            targets.append(hit_target)
        
        # Pad with zeros for last prediction_horizon rows
        targets.extend([0] * self.prediction_horizon)
        
        target_series = pd.Series(targets, index=df_h1.index)
        positive_rate = target_series.mean()
        
        print(f"✅ Targets created: {positive_rate:.1%} positive samples")
        return target_series

    def create_sequences(self, features: pd.DataFrame, targets: pd.Series, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences identik dengan prediction.py
        M15 → H1 conversion dengan sampling every 4 candles
        """
        print(f"📊 Creating sequences (length: {sequence_length}h)...")
        
        # Convert to H1
        df_h1 = features.resample('1H', label='right', closed='right').agg({
            col: 'last' if col not in ['volume'] else 'sum' 
            for col in features.columns
        }).dropna()
        
        targets_h1 = self.create_target(df_h1[['open', 'high', 'low', 'close', 'volume']])
        
        # Align with features M15 for sequence sampling
        X_sequences = []
        y_sequences = []
        
        for i in range(len(df_h1) - self.prediction_horizon):
            h1_time = df_h1.index[i]
            
            try:
                # Find corresponding M15 index
                m15_end_idx = features.index.get_loc(h1_time)
            except KeyError:
                # Find nearest
                time_diffs = abs(features.index - h1_time)
                nearest_idx = time_diffs.argmin()
                if time_diffs[nearest_idx] > pd.Timedelta(hours=1):
                    continue
                m15_end_idx = nearest_idx
            
            # Sample every 4 M15 candles (hourly sampling)
            m15_start_idx = m15_end_idx - (sequence_length * TRAINING_CONFIG['M15_PER_HOUR']) + 1
            
            if m15_start_idx < 0:
                continue
            
            seq = features.values[m15_start_idx:m15_end_idx+1:TRAINING_CONFIG['M15_PER_HOUR']]
            
            if len(seq) != sequence_length:
                continue
            
            X_sequences.append(seq)
            y_sequences.append(targets_h1.iloc[i])
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        print(f"✅ Sequences created: {X.shape[0]} samples, {X.shape[1]} timesteps, {X.shape[2]} features")
        print(f"   Positive samples: {y.mean():.1%}")
        
        return X, y

    def create_model_architecture(self, sequence_length: int, feature_count: int, model_name: str) -> tf.keras.Model:
        """
        Create neural network architecture
        Different architectures based on model type for variety
        """
        model = Sequential()
        
        if model_name in ['model_a', 'model_b']:  # Baseline models
            model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, feature_count)))
            model.add(Dropout(0.2))
            model.add(LSTM(32, return_sequences=False))
            model.add(Dropout(0.2))
            
        elif model_name in ['model_c', 'model_d']:  # Volatility/pattern models
            model.add(LSTM(128, return_sequences=True, input_shape=(sequence_length, feature_count)))
            model.add(Dropout(0.3))
            model.add(LSTM(64, return_sequences=True))
            model.add(Dropout(0.3))
            model.add(LSTM(32, return_sequences=False))
            model.add(Dropout(0.2))
            
        elif model_name == 'model_e':  # Daily cycle (24h)
            model.add(LSTM(96, return_sequences=True, input_shape=(sequence_length, feature_count)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(LSTM(48, return_sequences=False))
            model.add(Dropout(0.2))
            
        elif model_name in ['model_f', 'model_g']:  # Extended trend (40h)
            model.add(LSTM(128, return_sequences=True, input_shape=(sequence_length, feature_count)))
            model.add(Dropout(0.3))
            model.add(LSTM(96, return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(LSTM(48, return_sequences=False))
            model.add(Dropout(0.2))
            
        elif model_name in ['model_h', 'model_j']:  # Reversal/consolidation
            model.add(LSTM(80, return_sequences=True, input_shape=(sequence_length, feature_count)))
            model.add(Dropout(0.25))
            model.add(LSTM(40, return_sequences=False))
            model.add(Dropout(0.2))
            
        elif model_name == 'model_i':  # Weekly patterns (32h)
            model.add(LSTM(112, return_sequences=True, input_shape=(sequence_length, feature_count)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(LSTM(56, return_sequences=False))
            model.add(Dropout(0.2))
        
        # Final layers (same for all)
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model

    def train_single_model(self, model_name: str) -> Dict[str, Any]:
        """Train single model and save all required files"""
        
        print(f"\n{'='*80}")
        print(f"🚀 TRAINING {model_name.upper()}")
        print(f"{'='*80}")
        
        config = MODEL_CONFIGS[model_name]
        sequence_length = config['sequence_length']
        
        print(f"📋 Description: {config['description']}")
        print(f"📏 Sequence Length: {sequence_length} hours")
        
        # Load and prepare data
        df_m15 = self.load_m15_data(self.data_file)
        if df_m15 is None:
            return None
        
        features = self.extract_features(df_m15)
        
        # Create sequences
        X, y = self.create_sequences(features, None, sequence_length)
        
        if len(X) == 0:
            print(f"❌ No sequences created for {model_name}")
            return None
        
        # Train/val/test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=TRAINING_CONFIG['TEST_SIZE'], 
            random_state=TRAINING_CONFIG['RANDOM_STATE'], stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=TRAINING_CONFIG['VALIDATION_SIZE'], 
            random_state=TRAINING_CONFIG['RANDOM_STATE'], stratify=y_temp
        )
        
        print(f"📊 Data splits:")
        print(f"   Train: {X_train.shape[0]:,} samples ({y_train.mean():.1%} positive)")
        print(f"   Val:   {X_val.shape[0]:,} samples ({y_val.mean():.1%} positive)")
        print(f"   Test:  {X_test.shape[0]:,} samples ({y_test.mean():.1%} positive)")
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        print(f"✅ Features scaled")
        
        # Create model
        model = self.create_model_architecture(sequence_length, X_train.shape[2], model_name)
        print(f"🏗️  Model created: {model.count_params():,} parameters")
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=TRAINING_CONFIG['PATIENCE'], restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train
        print(f"🏃 Training started...")
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=TRAINING_CONFIG['EPOCHS'],
            batch_size=TRAINING_CONFIG['BATCH_SIZE'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test_scaled, y_test, verbose=0)
        f1_score = 2 * (test_prec * test_rec) / (test_prec + test_rec) if (test_prec + test_rec) > 0 else 0
        
        print(f"\n📈 FINAL RESULTS:")
        print(f"   Test Accuracy:  {test_acc:.1%}")
        print(f"   Test Precision: {test_prec:.1%}")
        print(f"   Test Recall:    {test_rec:.1%}")
        print(f"   Test F1-Score:  {f1_score:.1%}")
        
        # Create model folder
        os.makedirs(model_name, exist_ok=True)
        
        # Save model (Keras format)
        model.save(f"{model_name}/model.keras")
        
        # Save scaler
        joblib.dump(scaler, f"{model_name}/scaler.pkl")
        
        # Save config (for compatibility)
        config_data = {
            'sequence_length': sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'target_threshold': self.target_threshold,
            'feature_count': X_train.shape[2],
            'model_architecture': 'LSTM',
            'training_samples': X_train.shape[0]
        }
        joblib.dump(config_data, f"{model_name}/config.pkl")
        
        # Save model info (identik dengan format prediction.py)
        model_info = {
            'sequence_length': sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'threshold': self.target_threshold,
            'features_count': X_train.shape[2],
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': config['description'],
            'test_accuracy': test_acc,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_f1': f1_score
        }
        joblib.dump(model_info, f"{model_name}/model_info.pkl")
        
        print(f"✅ Model saved to {model_name}/")
        
        return {
            'model_name': model_name,
            'sequence_length': sequence_length,
            'test_accuracy': test_acc,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_f1': f1_score,
            'history': history.history
        }

    def train_all_models(self):
        """Train all 10 models"""
        print(f"\n🚀 STARTING MULTI-MODEL TRAINING")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Target: {self.target_threshold*100:.1f}% profit in {self.prediction_horizon}h")
        print(f"📊 Models: {len(MODEL_CONFIGS)}")
        
        results = []
        
        for model_name in MODEL_CONFIGS.keys():
            try:
                result = self.train_single_model(model_name)
                if result:
                    results.append(result)
                print(f"✅ {model_name} completed successfully")
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                continue
        
        # Summary
        print(f"\n{'='*80}")
        print("📊 TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"{'Model':<10} | {'Seq':<4} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<6} | {'F1':<6}")
        print("-" * 60)
        
        for result in results:
            print(f"{result['model_name']:<10} | {result['sequence_length']:<4} | {result['test_accuracy']:<8.1%} | {result['test_precision']:<9.1%} | {result['test_recall']:<6.1%} | {result['test_f1']:<6.1%}")
        
        avg_acc = np.mean([r['test_accuracy'] for r in results])
        avg_prec = np.mean([r['test_precision'] for r in results])
        avg_rec = np.mean([r['test_recall'] for r in results])
        avg_f1 = np.mean([r['test_f1'] for r in results])
        
        print("-" * 60)
        print(f"{'AVERAGE':<10} | {'':4} | {avg_acc:<8.1%} | {avg_prec:<9.1%} | {avg_rec:<6.1%} | {avg_f1:<6.1%}")
        
        print(f"\n✅ Training completed: {len(results)}/{len(MODEL_CONFIGS)} models")
        print(f"📁 Models saved in respective folders")
        print(f"🔮 Ready for ensemble prediction!")

if __name__ == "__main__":
    # Initialize trainer
    trainer = BTCUSDTModelTrainer()
    
    # Train all models
    trainer.train_all_models()
