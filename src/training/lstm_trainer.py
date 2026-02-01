"""LSTM deep learning model trainer."""

import warnings
warnings.filterwarnings('ignore')

try:
    from training.evaluation import evaluate_model
except ImportError:
    from evaluation import evaluate_model


def train_lstm(X_train, y_train, X_test, y_test, scaler, save_path=None):
    """Train LSTM model and return (model, metrics, predictions) or (None, None, None) if TF unavailable."""
    print("\n" + "=" * 50)
    print("Training LSTM")
    print("=" * 50)
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        
        tf.get_logger().setLevel('ERROR')
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        print("Training LSTM model... (this may take a while)")
        model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        y_pred = model.predict(X_test, verbose=0).flatten()
        
        # Inverse transform predictions
        if scaler is not None:
            y_test_inv = y_test * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
            y_pred_inv = y_pred * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
        else:
            y_test_inv, y_pred_inv = y_test, y_pred
        
        metrics = evaluate_model(y_test_inv, y_pred_inv, "LSTM")
        
        if save_path:
            model.save(save_path)
            print(f"Model saved to {save_path}")
        
        return model, metrics, y_pred_inv
        
    except ImportError:
        print("TensorFlow not available. Skipping LSTM training.")
        return None, None, None
