# pylint: disable=import-error
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring

import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn import metrics

from PIL import Image
import cv2

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

# Use tf.data API for more efficient data loading
def create_dataset(images, labels, batch_size=64, shuffle=True, augment=False):
    """Create a TensorFlow dataset from images and labels."""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    
    if augment:
        # Add data augmentation if needed
        def augment_fn(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.1)
            return image, label
        
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch for performance
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Path and parameters
PATH = "../lung_colon_image_set/lung_image_sets"
IMG_SIZE = 256
SPLIT = 0.2
EPOCHS = 10
BATCH_SIZE = 32  # Reduced batch size to prevent memory issues

def show_example_pics():
    """Show example images from each class."""
    catalogs = os.listdir(PATH)

    for catalog in catalogs:
        image_dir = f"{PATH}/{catalog}"
        images = os.listdir(image_dir)

        fig, ax = plt.subplots(1, 3)
        fig.suptitle(f"Images for {catalog} category: ")

        for i in range(3):
            k = np.random.randint(0, len(images))
            img = np.array(Image.open(f"{PATH}/{catalog}/{images[k]}"))
            ax[i].imshow(img)
            ax[i].axis("off")
        plt.show()

def prepare_data():
    """Prepare the data for training in smaller batches."""
    x, y = [], []

    catalogs = sorted(os.listdir(PATH))
    # Print categories for debugging
    print(f"Categories found: {catalogs}")
    
    for i, catalog in enumerate(catalogs):
        images = glob(f"{PATH}/{catalog}/*.jpeg")
        print(f"Found {len(images)} images in category {catalog}")
        
        # Load a sample of images first to test
        sample_size = min(len(images), 500)  # Load at most 500 images per class initially
        for image in images[:sample_size]:
            try:
                img = cv2.imread(image)
                if img is None:
                    print(f"Warning: Could not read image {image}")
                    continue
                    
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
                x.append(img_resized)
                y.append(i)
            except Exception as e:
                print(f"Error processing image {image}: {e}")
                continue
    
    x = np.asarray(x)
    one_hot_encoded_y = pd.get_dummies(y).values
    
    print(f"Data shape: {x.shape}, Labels shape: {one_hot_encoded_y.shape}")
    
    # Split data
    x_train, x_val, y_train, y_val = train_test_split(
        x, one_hot_encoded_y, test_size=SPLIT, random_state=2022
    )
    
    return x_train, x_val, y_train, y_val

def build_model():
    """Build the CNN model."""
    model = keras.models.Sequential([
        layers.Conv2D(filters=32,
                    kernel_size=(5, 5),
                    activation='relu',
                    input_shape=(IMG_SIZE, IMG_SIZE, 3),
                    padding='same'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(filters=64,
                    kernel_size=(3, 3),
                    activation='relu',
                    padding='same'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(filters=128,
                    kernel_size=(3, 3),
                    activation='relu',
                    padding='same'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(3, activation='softmax')  # Ensure this matches the number of classes
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

class AccuracyCallback(keras.callbacks.Callback):
    """Custom callback that stops training when validation accuracy reaches a threshold."""
    def __init__(self, threshold=0.9):
        super(AccuracyCallback, self).__init__()
        self.threshold = threshold
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_accuracy = logs.get('val_accuracy')
        if val_accuracy is not None and val_accuracy > self.threshold:
            print(f'\nValidation accuracy reached {val_accuracy:.4f}, stopping training')
            self.model.stop_training = True

def main():
    """Main function to run the training pipeline."""
    # Prepare data
    print("Loading and preparing data...")
    x_train, x_val, y_train, y_val = prepare_data()
    
    # Create datasets
    print("Creating TensorFlow datasets...")
    train_dataset = create_dataset(x_train, y_train, batch_size=BATCH_SIZE, augment=True)
    val_dataset = create_dataset(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    
    # Build model
    print("Building model...")
    model = build_model()
    model.summary()
    
    # Set up callbacks
    accuracy_callback = AccuracyCallback(threshold=0.9)
    early_stopping = EarlyStopping(
        patience=3,
        monitor='val_accuracy',
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        patience=2,
        factor=0.5,
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
            callbacks=[early_stopping, reduce_lr, accuracy_callback]
        )
        
        # Plot results
        history_df = pd.DataFrame(history.history)
        history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('model_accuracy.png')
        plt.show()
        
        # Save model
        model.save('lung_image_classifier.keras')
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    # Set memory limit for TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Limit GPU memory
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]  # Limit to 4GB
            )
        except RuntimeError as e:
            print(e)
    
    main()