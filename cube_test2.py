# pylint: disable=import-error
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import traceback
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

SEQUENCE_LENGTH = 50
SPLIT = 0.2
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.0001

def prepare_data(file_path, sequence_length=SEQUENCE_LENGTH):
    """
    Prepare time series data from TSV file for sequence prediction.
    Each sequence of inputs will predict the next motor values.
    """
    print(f"Loading data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip')
    except TypeError:
        df = pd.read_csv(file_path, sep='\t', error_bad_lines=False)
    
    print(f"Data loaded. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    required_columns = ['t', 
                      'm_1', 'm_2', 'm_3',  # outputs (motor values)
                      'o_x', 'o_y', 'o_z',  # inputs (orientation)
                      's_x', 's_y', 's_z',  # inputs (angular velocity of cube)
                      'w_1', 'w_2', 'w_3']  # inputs (angular velocity of wheels)
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in data: {missing_columns}")
    
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    mask = numeric_df.notna().all(axis=1)
    df = df[mask]
    
    print(f"After removing non-numeric rows: {df.shape}")
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if df.isna().any().any():
        print("Warning: NaN values found in data. Filling with forward fill then backward fill.")
        df = df.ffill().bfill()
    
    print("Handling outliers...")
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    input_features = ['o_x', 'o_y', 'o_z', 's_x', 's_y', 's_z', 'w_1', 'w_2', 'w_3']
    output_features = ['m_1', 'm_2', 'm_3']
    
    print("Adding engineered features...")
    for col in input_features:
        df[f"{col}_diff"] = df[col].diff().fillna(0)
    
    input_features = input_features + [f"{col}_diff" for col in input_features]
    
    window_sizes = [5, 10]
    for window in window_sizes:
        for col in input_features[:9]:
            df[f"{col}_roll_mean_{window}"] = df[col].rolling(window=window, min_periods=1).mean()
            df[f"{col}_roll_std_{window}"] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
            input_features.extend([f"{col}_roll_mean_{window}", f"{col}_roll_std_{window}"])

    inputs = df[input_features].values
    outputs = df[output_features].values
    
    input_scaler = MinMaxScaler(feature_range=(-1, 1))
    output_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    inputs_scaled = input_scaler.fit_transform(inputs)
    outputs_scaled = output_scaler.fit_transform(outputs)

    X, y = [], []
    
    step = 1
    
    for i in range(0, len(df) - sequence_length, step):
        X.append(inputs_scaled[i:i+sequence_length])
    
        y.append(outputs_scaled[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)

    print(f"Created {len(X)} sequences with shape {X.shape}")
    print(f"Input features: {len(input_features)}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=SPLIT, shuffle=False
    )
    
    return X_train, X_val, y_train, y_val, input_scaler, output_scaler, input_features, output_features

def create_dataset(X, y, batch_size=BATCH_SIZE, shuffle=True):
    """Create a TensorFlow dataset from sequences and targets."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(X), 10000))
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def residual_lstm_block(x, units, dropout_rate=0.3, recurrent_dropout=0.2):
    """Create a residual LSTM block with skip connection."""
    lstm_out = layers.LSTM(units, return_sequences=True, recurrent_dropout=recurrent_dropout)(x)
    lstm_out = layers.Dropout(dropout_rate)(lstm_out)
    
    if x.shape[-1] == units:

        return layers.Add()([x, lstm_out])
    else:

        skip = layers.TimeDistributed(layers.Dense(units))(x)
        return layers.Add()([skip, lstm_out])

def build_advanced_model(input_shape):
    """Build an improved sequence prediction model for cube balancing."""
    inputs = keras.Input(shape=input_shape)
    
    x = layers.LSTM(128, return_sequences=True, recurrent_dropout=0.2)(inputs)
    x = layers.Dropout(0.3)(x)
    
    x = residual_lstm_block(x, 128, dropout_rate=0.3)
    x = residual_lstm_block(x, 128, dropout_rate=0.3)
    
    x = layers.LSTM(128, recurrent_dropout=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(3, activation='tanh')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model

def visualize_predictions(model, X_val, y_val, output_scaler, num_samples=10):
    """Visualize model predictions against actual values."""
    start_idx = np.random.randint(0, len(X_val) - num_samples)
    X_slice = X_val[start_idx:start_idx + num_samples]
    y_slice = y_val[start_idx:start_idx + num_samples]
    
    predictions_scaled = model.predict(X_slice)
    
    predictions = output_scaler.inverse_transform(predictions_scaled)
    actual = output_scaler.inverse_transform(y_slice)
    
    motor_names = ['Motor 1', 'Motor 2', 'Motor 3']
    
    plt.figure(figsize=(15, 12))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(actual[:, i], 'b-', linewidth=2, label='Actual')
        plt.plot(predictions[:, i], 'r--', linewidth=2, label='Predicted')
        plt.title(f'{motor_names[i]} Values', fontsize=14)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Motor Value', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_cube_predictions.png')
    plt.show()
    
    correlations = []
    for i in range(3):
        corr = np.corrcoef(actual[:, i], predictions[:, i])[0, 1]
        correlations.append(corr)
        print(f"{motor_names[i]} prediction correlation with actual: {corr:.4f}")
    
    return correlations

def feature_importance_analysis(model, input_feature_names):
    """Analyze feature importance using a simple perturbation method."""
    weights = model.get_weights()
    
    first_layer_weights = np.abs(weights[0])
    importance_scores = np.mean(first_layer_weights, axis=(0, 2))
    
    importance_df = pd.DataFrame({
        'Feature': input_feature_names,
        'Importance': importance_scores
    })
    
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20])
    plt.xlabel('Relative Importance')
    plt.title('Top 20 Input Features by Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    return importance_df

def main(data_file='cube_data.tsv'):
    """Main function to run the improved cube balancing training pipeline."""
    X_train, X_val, y_train, y_val, input_scaler, output_scaler, input_features, output_features = prepare_data(data_file)
    
    print("Creating TensorFlow datasets...")
    train_dataset = create_dataset(X_train, y_train, batch_size=BATCH_SIZE)
    val_dataset = create_dataset(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Building improved model architecture...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_advanced_model(input_shape)
    model.summary()
    
    early_stopping = EarlyStopping(
        patience=15,
        monitor='val_loss',
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        patience=8,
        factor=0.5,
        min_lr=1e-6,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        'best_cube_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    print("Starting training with improved parameters...")
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[early_stopping, reduce_lr, checkpoint]
        )
        

        plt.figure(figsize=(15, 7))
        

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Mean Absolute Error', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('improved_cube_training_history.png')
        plt.show()
        

        best_model = keras.models.load_model('best_cube_model.keras')
        

        print("\nEvaluating model predictions...")
        correlations = visualize_predictions(best_model, X_val, y_val, output_scaler)
        

        print("\nAnalyzing feature importance...")
        importance_df = feature_importance_analysis(best_model, input_features)
        print("Top 10 most important features:")
        print(importance_df.head(10))
        

        best_model.save('improved_cube_balancing_model.keras')
        print("Final model saved successfully!")
        

        joblib.dump(input_scaler, 'improved_input_scaler.pkl')
        joblib.dump(output_scaler, 'improved_output_scaler.pkl')
        print("Scalers saved successfully!")
        

        print("\nTraining Results:")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        print(f"Final training MAE: {history.history['mae'][-1]:.4f}")
        print(f"Final validation MAE: {history.history['val_mae'][-1]:.4f}")
        
        print("\nModel Performance:")
        model_performance = pd.DataFrame({
            'Motor': output_features,
            'Correlation': correlations
        })
        print(model_performance)
        
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()

def create_prediction_function(model_path='improved_cube_balancing_model.keras', 
                               input_scaler_path='improved_input_scaler.pkl',
                               output_scaler_path='improved_output_scaler.pkl',
                               feature_list_path='feature_list.pkl'):
    """
    Create a function that can be used for real-time prediction.
    Returns a function that takes current state and predicts motor values.
    """
    
    model = keras.models.load_model(model_path)
    input_scaler = joblib.load(input_scaler_path)
    output_scaler = joblib.load(output_scaler_path)
    
    try:
        input_features = joblib.load(feature_list_path)
    except:

        input_features = ['o_x', 'o_y', 'o_z', 's_x', 's_y', 's_z', 'w_1', 'w_2', 'w_3']
        derived_features = [f"{col}_diff" for col in input_features]
        for window in [5, 10]:
            for col in input_features:
                derived_features.extend([f"{col}_roll_mean_{window}", f"{col}_roll_std_{window}"])
        input_features = input_features + derived_features
    
    print(f"Loaded prediction model with {len(input_features)} input features")
    
    historical_data = []
    max_history = 50
    sequence_length = model.input_shape[1]
    
    def predict_motor_values(orientation, cube_velocity, wheel_velocity):
        """
        Predict motor values based on current state.
        
        Args:
            orientation: [o_x, o_y, o_z] - orientation of the cube
            cube_velocity: [s_x, s_y, s_z] - angular velocity of the cube
            wheel_velocity: [w_1, w_2, w_3] - angular velocity of the wheels
            
        Returns:
            [m_1, m_2, m_3] - motor values
        """
        nonlocal historical_data
        

        current_state = orientation + cube_velocity + wheel_velocity
        

        historical_data.append(current_state)
        

        if len(historical_data) > max_history:
            historical_data = historical_data[-max_history:]
        

        if len(historical_data) < max_history:
            return [0, 0, 0]
        

        df = pd.DataFrame(historical_data, columns=['o_x', 'o_y', 'o_z', 's_x', 's_y', 's_z', 'w_1', 'w_2', 'w_3'])
        

        for col in df.columns:
            df[f"{col}_diff"] = df[col].diff().fillna(0)
        

        window_sizes = [5, 10]
        for window in window_sizes:
            for col in df.columns[:9]:
                df[f"{col}_roll_mean_{window}"] = df[col].rolling(window=window, min_periods=1).mean()
                df[f"{col}_roll_std_{window}"] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
        

        features = df[input_features].values
        features_scaled = input_scaler.transform(features)
        

        sequence = features_scaled[-sequence_length:].reshape(1, sequence_length, -1)
        

        prediction_scaled = model.predict(sequence, verbose=0)
        

        prediction = output_scaler.inverse_transform(prediction_scaled)
        
        return prediction[0]
    
    return predict_motor_values

if __name__ == "__main__":
    data_file = '../cube_data.tsv'
    main(data_file)