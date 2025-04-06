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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import regularizers

from sklearn.model_selection import train_test_split, KFold
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
            # Apply more aggressive augmentation
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.7, 1.3)
            image = tf.image.random_saturation(image, 0.7, 1.3)
            
            # Random crop and resize to maintain shape consistency
            image = tf.image.random_crop(image, [int(IMG_SIZE*0.85), int(IMG_SIZE*0.85), 3])
            image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
            
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
                img_normalize = (img_resized - np.mean(img_resized)) / np.std(img_resized)
                x.append(img_normalize)
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
    # model = keras.models.Sequential([
    #     layers.Conv2D(filters=32,
    #                 kernel_size=(5, 5),
    #                 activation='relu',
    #                 input_shape=(IMG_SIZE, IMG_SIZE, 3),
    #                 padding='same',
    #                 kernel_regularizer=regularizers.l2(0.0001)),
    #     layers.BatchNormalization(),
    #     layers.MaxPooling2D(2, 2),

    #     layers.Conv2D(filters=64,
    #                 kernel_size=(3, 3),
    #                 activation='relu',
    #                 padding='same',
    #                 kernel_regularizer=regularizers.l2(0.0001)),
    #     layers.BatchNormalization(),
    #     layers.MaxPooling2D(2, 2),

    #     layers.Conv2D(filters=128,
    #                 kernel_size=(3, 3),
    #                 activation='relu',
    #                 padding='same',
    #                 kernel_regularizer=regularizers.l2(0.0001)),
    #     layers.BatchNormalization(),
    #     layers.MaxPooling2D(2, 2),

    #     layers.Flatten(),
    #     layers.Dense(256, activation='relu'),
    #     layers.BatchNormalization(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dropout(0.3),
    #     layers.BatchNormalization(),
    #     layers.Dense(3, activation='softmax')  # Ensure this matches the number of classes
    # ])
    
    model = keras.models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', 
                     input_shape=(IMG_SIZE, IMG_SIZE, 3), kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.7),  # Increased dropout
        layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0001,
            clipnorm=1.0,
        ),
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

def lr_schedule(epoch, lr):
    if epoch > 5:
        return lr * 0.5
    return lr

def main():
    """Main function to run the training pipeline with cross-validation."""
    # Prepare data
    print("Loading and preparing data...")
    x, y = [], []
    
    catalogs = sorted(os.listdir(PATH))
    print(f"Categories found: {catalogs}")
    
    for i, catalog in enumerate(catalogs):
        images = glob(f"{PATH}/{catalog}/*.jpeg")
        print(f"Found {len(images)} images in category {catalog}")
        
        sample_size = min(len(images), 500)
        for image in images[:sample_size]:
            try:
                img = cv2.imread(image)
                if img is None:
                    continue
                    
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
                # Optional: Add normalization
                # img_normalized = (img_resized - np.mean(img_resized)) / np.std(img_resized)
                # x.append(img_normalized)
                x.append(img_resized)
                y.append(i)
            except Exception as e:
                print(f"Error processing image {image}: {e}")
                continue
    
    x = np.asarray(x)
    y = np.array(y)

    one_hot_encoded_y = pd.get_dummies(y).values
    
    print(f"Data shape: {x.shape}, Labels shape: {one_hot_encoded_y.shape}")
    
    # Calculate class weights
    unique_classes, class_counts = np.unique(y, return_counts=True)
    total_samples = len(y)

    class_weights = {cls: total_samples / (len(unique_classes) * count) 
                    for cls, count in zip(unique_classes, class_counts)}
    print(f"Class weights: {class_weights}")
    
    # Implement k-fold cross-validation
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
        print(f"\n=== Training fold {fold+1}/5 ===")
        
        # Split data for this fold
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = one_hot_encoded_y[train_idx], one_hot_encoded_y[val_idx]
        
        # Create datasets
        train_dataset = create_dataset(x_train, y_train, batch_size=BATCH_SIZE, augment=True)
        val_dataset = create_dataset(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
        
        # Build model
        model = build_model()
        if fold == 0:  # Only show summary for first fold
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
        checkpoint = ModelCheckpoint(
            f'best_model_fold_{fold+1}.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        lr_scheduler = LearningRateScheduler(lr_schedule)
        
        # Train model
        try:
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=EPOCHS,
                verbose=1,
                class_weight=class_weights,  # Add class weights
                callbacks=[early_stopping, reduce_lr, accuracy_callback, checkpoint, lr_scheduler]
            )
            
            # Plot results for this fold
            history_df = pd.DataFrame(history.history)
            plt.figure(figsize=(10, 6))
            history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
            plt.title(f'Model Accuracy - Fold {fold+1}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.savefig(f'model_accuracy_fold_{fold+1}.png')
            plt.close()
            
            # Store metrics for this fold
            fold_metrics.append({
                'train_acc': history.history['accuracy'][-1],
                'val_acc': history.history['val_accuracy'][-1],
                'train_loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1]
            })
            
            print(f"Fold {fold+1} results:")
            print(f"  Training accuracy: {history.history['accuracy'][-1]:.4f}")
            print(f"  Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
            
        except Exception as e:
            print(f"Error during training fold {fold+1}: {e}")
    
    # Summarize cross-validation results
    print("\n=== Cross-validation results ===")
    for i, metrics in enumerate(fold_metrics):
        print(f"Fold {i+1}: train_acc={metrics['train_acc']:.4f}, val_acc={metrics['val_acc']:.4f}")
    
    avg_train_acc = np.mean([m['train_acc'] for m in fold_metrics])
    avg_val_acc = np.mean([m['val_acc'] for m in fold_metrics])
    print(f"\nAverage train accuracy: {avg_train_acc:.4f}")
    print(f"Average validation accuracy: {avg_val_acc:.4f}")
    
    # Train final model on all data if desired
    print("\n=== Training final model on all data ===")
    final_dataset = create_dataset(x, one_hot_encoded_y, batch_size=BATCH_SIZE, augment=True)
    final_model = build_model()
    
    final_checkpoint = ModelCheckpoint(
        'final_model.keras',
        monitor='accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    final_model.fit(
        final_dataset,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[reduce_lr, lr_scheduler, final_checkpoint]
    )
    
    print("Final model saved successfully!")

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
