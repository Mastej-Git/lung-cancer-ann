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
    
    # Map the new column names to the old structure
    # This is the mapping from new to old names
    column_mapping = {
        't [us]': 't',
        'ox [deg]': 'o_x',
        'oy [deg]': 'o_y',
        'oz [deg]': 'o_z',
        'sx [deg/s]': 's_x',
        'sy [deg/s]': 's_y',
        'sz [deg/s]': 's_z',
        'wx [rpm]': 'w_1',
        'wy [rpm]': 'w_2',
        'wz [rpm]': 'w_3',
        'cx [%]': 'cx',
        'cy [%]': 'cy',
        'cz [%]': 'cz'
    }
    
    # In case column names don't include units
    short_column_mapping = {
        't': 't',
        'ox': 'o_x',
        'oy': 'o_y',
        'oz': 'o_z',
        'sx': 's_x',
        'sy': 's_y',
        'sz': 's_z',
        'wx': 'w_1',
        'wy': 'w_2',
        'wz': 'w_3',
        'cx': 'cx',
        'cy': 'cy',
        'cz': 'cz'
    }
    
    # Rename columns based on available column names
    present_columns = df.columns.tolist()
    rename_dict = {}
    
    # Try with the full column names (with units)
    for new_col, old_col in column_mapping.items():
        if new_col in present_columns:
            rename_dict[new_col] = old_col
    
    # If we couldn't find the columns with units, try without units
    if not rename_dict:
        for new_col, old_col in short_column_mapping.items():
            if new_col in present_columns:
                rename_dict[new_col] = old_col
    
    # Rename columns if mapping is found
    if rename_dict:
        df = df.rename(columns=rename_dict)
        print(f"Renamed columns: {rename_dict}")
    
    # We need to determine what will be our motor outputs
    # In the original data, motor outputs were m_1, m_2, m_3
    # In the new data, we'll use the wheel velocities as motor outputs
    # Since there's no direct m_1, m_2, m_3 equivalent
    
    # First check if we have the original motor columns
    if 'm_1' in df.columns and 'm_2' in df.columns and 'm_3' in df.columns:
        output_features = ['m_1', 'm_2', 'm_3']
    else:
        # Use wheel velocities as motor outputs
        output_features = ['w_1', 'w_2', 'w_3']
    
    # Define the input features we'll use - exclude output features
    input_features = ['o_x', 'o_y', 'o_z', 's_x', 's_y', 's_z']
    
    # Add contact information if available
    if 'cx' in df.columns and 'cy' in df.columns and 'cz' in df.columns:
        input_features.extend(['cx', 'cy', 'cz'])
    
    # Check if we have all required columns
    required_columns = input_features + output_features + ['t']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in data: {missing_columns}")
    
    # Check for zeros or NaN in output features
    for col in output_features:
        zeros_count = (df[col] == 0).sum()
        if zeros_count / len(df) > 0.9:  # If more than 90% are zeros
            print(f"Warning: Column {col} contains {zeros_count} zeros out of {len(df)} rows.")
    
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
    
    # Calculate input features statistics for debugging
    print("\nInput features statistics:")
    for col in input_features:
        print(f"{col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}, std={df[col].std():.4f}, zeros={sum(df[col] == 0)}")
    
    print("\nOutput features statistics:")
    for col in output_features:
        print(f"{col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}, std={df[col].std():.4f}, zeros={sum(df[col] == 0)}")
    
    # Add engineered features
    print("Adding engineered features...")
    for col in input_features:
        df[f"{col}_diff"] = df[col].diff().fillna(0)
    
    input_features_with_diff = input_features + [f"{col}_diff" for col in input_features]
    
    window_sizes = [5, 10]
    rolling_features = []
    for window in window_sizes:
        for col in input_features:
            # Rolling mean
            roll_mean_col = f"{col}_roll_mean_{window}"
            df[roll_mean_col] = df[col].rolling(window=window, min_periods=1).mean()
            rolling_features.append(roll_mean_col)
            
            # Rolling std
            roll_std_col = f"{col}_roll_std_{window}"
            df[roll_std_col] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
            rolling_features.append(roll_std_col)
    
    # Final input features list
    all_input_features = input_features_with_diff + rolling_features
    
    # Verify no NaN in final features
    for col in all_input_features + output_features:
        if df[col].isna().any():
            print(f"Warning: NaN values found in column {col} after feature engineering. Filling with 0.")
            df[col] = df[col].fillna(0)
    
    inputs = df[all_input_features].values
    outputs = df[output_features].values
    
    # Check for low variance in output
    output_std = np.std(outputs, axis=0)
    for i, std in enumerate(output_std):
        if std < 1e-6:
            print(f"Warning: Very low variance in output {output_features[i]} (std={std}). This may cause training issues.")
    
    input_scaler = MinMaxScaler(feature_range=(-1, 1))
    output_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    inputs_scaled = input_scaler.fit_transform(inputs)
    
    # Handle case where output has zero variance
    for i in range(outputs.shape[1]):
        if np.std(outputs[:, i]) < 1e-6:
            print(f"Adding small noise to output {output_features[i]} to avoid zero variance")
            outputs[:, i] = outputs[:, i] + np.random.normal(0, 1e-5, size=outputs.shape[0])
    
    outputs_scaled = output_scaler.fit_transform(outputs)

    X, y = [], []
    
    step = 1
    
    for i in range(0, len(df) - sequence_length, step):
        X.append(inputs_scaled[i:i+sequence_length])
        y.append(outputs_scaled[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)

    print(f"Created {len(X)} sequences with shape {X.shape}")
    print(f"Input features: {len(all_input_features)}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=SPLIT, shuffle=False
    )
    
    # Save the list of input features for future reference
    joblib.dump(all_input_features, 'feature_list.pkl')
    
    return X_train, X_val, y_train, y_val, input_scaler, output_scaler, all_input_features, output_features

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

def visualize_predictions(model, X_val, y_val, output_scaler, output_features, num_samples=10):
    """Visualize model predictions against actual values."""
    start_idx = np.random.randint(0, len(X_val) - num_samples)
    X_slice = X_val[start_idx:start_idx + num_samples]
    y_slice = y_val[start_idx:start_idx + num_samples]
    
    predictions_scaled = model.predict(X_slice)
    
    predictions = output_scaler.inverse_transform(predictions_scaled)
    actual = output_scaler.inverse_transform(y_slice)
    
    # Create figure with subplots for each output feature
    plt.figure(figsize=(15, 12))
    
    correlations = []
    
    for i in range(len(output_features)):
        plt.subplot(len(output_features), 1, i+1)
        
        # If all actual values are the same, the correlation will be undefined (NaN)
        # Check for this case
        if np.std(actual[:, i]) < 1e-6:
            print(f"Warning: {output_features[i]} has constant actual values, correlation will be NaN")
            correlation = float('nan')
        else:
            # Calculate correlation if possible
            try:
                correlation = np.corrcoef(actual[:, i], predictions[:, i])[0, 1]
            except:
                correlation = float('nan')
        
        correlations.append(correlation)
        
        plt.plot(actual[:, i], 'b-', linewidth=2, label='Actual')
        plt.plot(predictions[:, i], 'r--', linewidth=2, label='Predicted')
        plt.title(f'{output_features[i]} Values', fontsize=14)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        print(f"{output_features[i]} prediction correlation with actual: {correlation:.4f}")
    
    plt.tight_layout()
    plt.savefig('improved_cube_predictions.png')
    plt.show()
    
    return correlations

def feature_importance_analysis(model, input_feature_names):
    """Analyze feature importance using gradient-based methods."""
    print("Analyzing feature importance...")
    
    # Extract weights from the first layer
    weights = model.get_weights()
    first_layer_weights = weights[0]  # This should be the LSTM input weights
    
    # The first layer weights for LSTM have 4 gates (input, forget, cell, output)
    # For feature importance, we'll take the average absolute value across all gates
    importance_scores = np.mean(np.abs(first_layer_weights), axis=1)
    
    # Average across all LSTM units
    feature_importance = np.mean(importance_scores, axis=0)
    
    # Create dataframe for visualization
    importance_df = pd.DataFrame({
        'Feature': input_feature_names,
        'Importance': feature_importance
    })
    
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20])
    plt.xlabel('Relative Importance')
    plt.title('Top 20 Input Features by Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    return importance_df

def main(data_file='cube_edge_data.tsv'):
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
        'best_cube_edge_model.keras',
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
        plt.savefig('improved_cube_edge_training_history.png')
        plt.show()
        
        best_model = keras.models.load_model('best_cube_edge_model.keras')
        
        print("\nEvaluating model predictions...")
        correlations = visualize_predictions(best_model, X_val, y_val, output_scaler, output_features)
        
        print("\nAnalyzing feature importance...")
        try:
            importance_df = feature_importance_analysis(best_model, input_features)
            print("Top 10 most important features:")
            print(importance_df.head(10))
        except Exception as e:
            print(f"Error in feature importance analysis: {e}")
            traceback.print_exc()
        
        best_model.save('improved_cube_edge_balancing_model.keras')
        print("Final model saved successfully!")
        
        joblib.dump(input_scaler, 'improved_input_scaler_edge.pkl')
        joblib.dump(output_scaler, 'improved_output_scaler_edge.pkl')
        print("Scalers saved successfully!")
        
        print("\nTraining Results:")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        print(f"Final training MAE: {history.history['mae'][-1]:.4f}")
        print(f"Final validation MAE: {history.history['val_mae'][-1]:.4f}")
        
        print("\nModel Performance:")
        model_performance = pd.DataFrame({
            'Output': output_features,
            'Correlation': correlations
        })
        print(model_performance)
        
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()

def create_prediction_function(model_path='improved_cube_edge_balancing_model.keras', 
                               input_scaler_path='improved_input_scaler_edge.pkl',
                               output_scaler_path='improved_output_scaler_edge.pkl',
                               feature_list_path='feature_list.pkl'):
    """
    Create a function that can be used for real-time prediction.
    Returns a function that takes current state and predicts outputs.
    """
    
    model = keras.models.load_model(model_path)
    input_scaler = joblib.load(input_scaler_path)
    output_scaler = joblib.load(output_scaler_path)
    
    try:
        input_features = joblib.load(feature_list_path)
    except:
        # Default features if file not found
        base_features = ['o_x', 'o_y', 'o_z', 's_x', 's_y', 's_z', 'cx', 'cy', 'cz']
        derived_features = [f"{col}_diff" for col in base_features]
        for window in [5, 10]:
            for col in base_features:
                derived_features.extend([f"{col}_roll_mean_{window}", f"{col}_roll_std_{window}"])
        input_features = base_features + derived_features
    
    print(f"Loaded prediction model with {len(input_features)} input features")
    
    historical_data = []
    max_history = 50
    sequence_length = model.input_shape[1]
    
    def predict_outputs(orientation, cube_velocity, contact_pct=None, wheel_velocity=None):
        """
        Predict outputs based on current state.
        
        Args:
            orientation: [o_x, o_y, o_z] - orientation of the cube in degrees
            cube_velocity: [s_x, s_y, s_z] - angular velocity of the cube in deg/s
            contact_pct: [cx, cy, cz] - contact percentage in each direction
            wheel_velocity: [w_1, w_2, w_3] - angular velocity of the wheels in rpm
            
        Returns:
            Predicted outputs (wheel velocities or motor values)
        """
        nonlocal historical_data
        
        # Prepare the current state
        current_state = orientation + cube_velocity
        
        # Add contact percentages if provided
        if contact_pct:
            current_state = current_state + contact_pct
        else:
            # Default zeros if not provided
            current_state = current_state + [0, 0, 0]
        
        # Store current state
        historical_data.append(current_state)
        
        # Maintain history window
        if len(historical_data) > max_history:
            historical_data = historical_data[-max_history:]
        
        # Wait until we have enough history
        if len(historical_data) < max_history:
            return [0, 0, 0]
        
        # Create a DataFrame from historical data
        columns = ['o_x', 'o_y', 'o_z', 's_x', 's_y', 's_z', 'cx', 'cy', 'cz']
        df = pd.DataFrame(historical_data, columns=columns)
        
        # Calculate derived features
        # First, the difference features
        for col in df.columns:
            df[f"{col}_diff"] = df[col].diff().fillna(0)
        
        # Then, rolling statistics
        window_sizes = [5, 10]
        for window in window_sizes:
            for col in df.columns[:9]:  # Only base features
                df[f"{col}_roll_mean_{window}"] = df[col].rolling(window=window, min_periods=1).mean()
                df[f"{col}_roll_std_{window}"] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
        
        # Get the subset of features that match the training input features
        available_features = []
        for feature in input_features:
            if feature in df.columns:
                available_features.append(feature)
            else:
                print(f"Warning: Feature {feature} not available in current data")
        
        # Extract feature values
        features = df[available_features].values
        
        # Check if we have missing features
        if len(available_features) < len(input_features):
            missing_count = len(input_features) - len(available_features)
            print(f"Warning: Missing {missing_count} features from input_features")
            # Create a dummy array with zeros for missing features
            dummy_features = np.zeros((features.shape[0], missing_count))
            features = np.hstack((features, dummy_features))
        
        # Scale features
        features_scaled = input_scaler.transform(features)
        
        # Create sequence for prediction
        sequence = features_scaled[-sequence_length:].reshape(1, sequence_length, -1)
        
        # Make prediction
        prediction_scaled = model.predict(sequence, verbose=0)
        
        # Inverse transform to get actual values
        prediction = output_scaler.inverse_transform(prediction_scaled)
        
        return prediction[0]
    
    return predict_outputs

if __name__ == "__main__":
    data_file = 'cube_edge_data.tsv'
    main(data_file)