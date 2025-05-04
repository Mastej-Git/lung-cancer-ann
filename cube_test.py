# pylint: disable=import-error
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure GPU memory growth to prevent memory allocation issues
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Parameters
SEQUENCE_LENGTH = 20  # Number of time steps to include in each sample
SPLIT = 0.2  # Validation split ratio
EPOCHS = 30
BATCH_SIZE = 64

def prepare_data(file_path, sequence_length=SEQUENCE_LENGTH):
    """
    Prepare time series data from TSV file for sequence prediction.
    Each sequence of inputs will predict the next motor values.
    """
    print(f"Loading data from {file_path}...")
    
    # Load data from TSV file - add error_bad_lines=False to skip problematic rows
    try:
        df = pd.read_csv(file_path, sep='\t', error_bad_lines=False)  # For older pandas versions
    except TypeError:
        # For newer pandas versions where error_bad_lines is renamed
        df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip')
    
    print(f"Data loaded. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Ensure we have the expected columns
    required_columns = ['t', 
                      'm_1', 'm_2', 'm_3',  # outputs (motor values)
                      'o_x', 'o_y', 'o_z',  # inputs (orientation)
                      's_x', 's_y', 's_z',  # inputs (angular velocity of cube)
                      'w_1', 'w_2', 'w_3']  # inputs (angular velocity of wheels)
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in data: {missing_columns}")
    
    # Filter out non-numeric rows
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    mask = numeric_df.notna().all(axis=1)
    df = df[mask]
    
    print(f"After removing non-numeric rows: {df.shape}")
    
    # Convert all data to numeric format
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for and handle NaN values
    if df.isna().any().any():
        print("Warning: NaN values found in data. Filling with forward fill then backward fill.")
        df = df.ffill().bfill()  # Use newer pandas methods
    
    # Define input features and output targets
    input_features = ['o_x', 'o_y', 'o_z', 's_x', 's_y', 's_z', 'w_1', 'w_2', 'w_3']
    output_features = ['m_1', 'm_2', 'm_3']
    
    # Extract the data arrays
    inputs = df[input_features].values
    outputs = df[output_features].values
    
    # Normalize the data
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    
    inputs_scaled = input_scaler.fit_transform(inputs)
    outputs_scaled = output_scaler.fit_transform(outputs)
    
    # Create sequences for time series prediction
    X, y = [], []
    
    for i in range(len(df) - sequence_length):
        # Input sequence
        X.append(inputs_scaled[i:i+sequence_length])
        # Output is the next motor values after the sequence
        y.append(outputs_scaled[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)

    print(f"Created {len(X)} sequences with shape {X.shape}")
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=SPLIT, shuffle=False  # Keep time ordering for time series
    )
    
    return X_train, X_val, y_train, y_val, input_scaler, output_scaler

def create_dataset(X, y, batch_size=64, shuffle=True):
    """Create a TensorFlow dataset from sequences and targets."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    # Batch and prefetch for performance
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def build_model(input_shape):
    """Build a sequence prediction model for cube balancing."""
    model = keras.Sequential([
        # LSTM layers to process the sequence
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        
        # Dense layers for prediction
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(16, activation='relu'),
        layers.Dense(3)  # 3 outputs for motor values m_1, m_2, m_3
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # Mean squared error for regression
        metrics=['mae']  # Mean absolute error
    )
    
    return model

def visualize_predictions(model, X_val, y_val, output_scaler, num_samples=5):
    """Visualize model predictions against actual values."""
    # Make predictions
    predictions_scaled = model.predict(X_val[:num_samples])
    
    # Inverse transform to get original scale
    predictions = output_scaler.inverse_transform(predictions_scaled)
    actual = output_scaler.inverse_transform(y_val[:num_samples])
    
    # Plot predictions vs actual
    motor_names = ['Motor 1', 'Motor 2', 'Motor 3']
    
    plt.figure(figsize=(15, 10))
    for i in range(3):  # For each motor
        plt.subplot(3, 1, i+1)
        plt.plot(actual[:, i], 'b-', label='Actual')
        plt.plot(predictions[:, i], 'r--', label='Predicted')
        plt.title(f'{motor_names[i]} Values')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cube_predictions.png')
    plt.show()

def main(data_file='cube_data.tsv'):
    """Main function to run the cube balancing training pipeline."""
    # Prepare data
    X_train, X_val, y_train, y_val, input_scaler, output_scaler = prepare_data(data_file)
    
    # Create datasets
    print("Creating TensorFlow datasets...")
    train_dataset = create_dataset(X_train, y_train, batch_size=BATCH_SIZE)
    val_dataset = create_dataset(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    
    # Build model
    print("Building model...")
    input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, num_features)
    model = build_model(input_shape)
    model.summary()
    
    # Set up callbacks
    early_stopping = EarlyStopping(
        patience=5,
        monitor='val_loss',
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        patience=3,
        factor=0.5,
        verbose=1
    )
    checkpoint = ModelCheckpoint(
        'best_cube_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    # Train model
    print("Starting training...")
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[early_stopping, reduce_lr, checkpoint]
        )
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('cube_training_history.png')
        plt.show()
        
        # Visualize some predictions
        visualize_predictions(model, X_val, y_val, output_scaler)
        
        # Save model
        model.save('cube_balancing_model.keras')
        print("Model saved successfully!")
        
        # Save scalers for later use
        import joblib
        joblib.dump(input_scaler, 'input_scaler.pkl')
        joblib.dump(output_scaler, 'output_scaler.pkl')
        print("Scalers saved successfully!")
        
        # Print final metrics
        print("\nTraining Results:")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        print(f"Final training MAE: {history.history['mae'][-1]:.4f}")
        print(f"Final validation MAE: {history.history['val_mae'][-1]:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")

def create_prediction_function(model_path='cube_balancing_model.keras', 
                              input_scaler_path='input_scaler.pkl',
                              output_scaler_path='output_scaler.pkl'):
    """
    Create a function that can be used for real-time prediction.
    Returns a function that takes current state and predicts motor values.
    """
    import joblib
    
    # Load model and scalers
    model = keras.models.load_model(model_path)
    input_scaler = joblib.load(input_scaler_path)
    output_scaler = joblib.load(output_scaler_path)
    
    # The window of data we need to keep
    sequence_buffer = np.zeros((1, SEQUENCE_LENGTH, 9))  # (batch_size, sequence_length, features)
    
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
        nonlocal sequence_buffer
        
        # Create input vector from current state
        current_state = np.array([orientation + cube_velocity + wheel_velocity])
        
        # Scale the input
        current_state_scaled = input_scaler.transform(current_state)
        
        # Update the sequence buffer (shift left and add new state)
        sequence_buffer = np.roll(sequence_buffer, -1, axis=1)
        sequence_buffer[0, -1, :] = current_state_scaled[0]
        
        # Make prediction
        prediction_scaled = model.predict(sequence_buffer)
        
        # Convert back to original scale
        prediction = output_scaler.inverse_transform(prediction_scaled)
        
        return prediction[0]  # Return as a simple list [m_1, m_2, m_3]
    
    return predict_motor_values

if __name__ == "__main__":
    data_file = '../cube_data.tsv'  # Change this to your actual file path
    main(data_file)