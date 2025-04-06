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

from sklearn.model_selection import KFold
import gc  # Garbage collector
import cv2

# Configure GPU memory growth to prevent memory allocation issues
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Path and parameters
PATH = "../lung_colon_image_set/lung_image_sets"
IMG_SIZE = 128  # Reduced image size
SPLIT = 0.2
EPOCHS = 10
BATCH_SIZE = 16  # Smaller batch size

# Generator function to load images in batches
def image_generator(image_paths, labels, batch_size=16):
    num_samples = len(image_paths)
    while True:
        # Shuffle at the beginning of each epoch
        indices = np.random.permutation(num_samples)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_images = []
            batch_labels = []
            
            for idx in batch_indices:
                try:
                    img = cv2.imread(image_paths[idx])
                    if img is None:
                        continue
                    
                    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
                    batch_images.append(img_resized)
                    batch_labels.append(labels[idx])
                except Exception as e:
                    print(f"Error processing image {image_paths[idx]}: {e}")
                    continue
            
            if batch_images:  # Only yield if batch is not empty
                yield np.array(batch_images), np.array(batch_labels)

# Data augmentation function
def augment_image(image):
    # Simple augmentation to avoid memory overhead
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)  # Horizontal flip
    
    if np.random.random() > 0.5:
        image = cv2.flip(image, 0)  # Vertical flip
    
    return image

def build_model():
    """Build a more memory-efficient CNN model."""
    model = keras.models.Sequential([
        layers.Conv2D(8, (3, 3), activation='relu', padding='same', 
                     input_shape=(IMG_SIZE, IMG_SIZE, 3), kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D(2, 2),
        
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Main function with memory-efficient implementation."""
    print("Processing data...")
    
    # Get image paths and labels
    image_paths = []
    labels = []
    
    catalogs = sorted(os.listdir(PATH))
    print(f"Categories found: {catalogs}")
    
    for i, catalog in enumerate(catalogs):
        catalog_images = glob(f"{PATH}/{catalog}/*.jpeg")
        print(f"Found {len(catalog_images)} images in category {catalog}")
        
        # Add to lists
        image_paths.extend(catalog_images)
        labels.extend([i] * len(catalog_images))
    
    # Create label indices
    unique_labels = np.unique(labels)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    indices = [label_to_index[label] for label in labels]
    one_hot_labels = np.eye(len(unique_labels))[indices]
    
    # Calculate class weights
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    class_weights = {cls: total_samples / (len(unique_classes) * count) 
                    for cls, count in zip(unique_classes, class_counts)}
    print(f"Class weights: {class_weights}")
    
    # Implement k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
        print(f"\n=== Training fold {fold+1}/5 ===")
        
        # Split data paths for this fold
        train_paths = [image_paths[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        train_labels = one_hot_labels[train_idx]
        val_labels = one_hot_labels[val_idx]
        
        # Build model
        model = build_model()
        if fold == 0:  # Only show summary for first fold
            model.summary()
        
        # Set up callbacks
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
        
        # Create generators
        train_gen = image_generator(train_paths, train_labels, BATCH_SIZE)
        val_gen = image_generator(val_paths, val_labels, BATCH_SIZE)
        
        # Calculate steps per epoch
        train_steps = len(train_paths) // BATCH_SIZE + (1 if len(train_paths) % BATCH_SIZE > 0 else 0)
        val_steps = len(val_paths) // BATCH_SIZE + (1 if len(val_paths) % BATCH_SIZE > 0 else 0)
        
        # Train model
        try:
            history = model.fit(
                train_gen,
                steps_per_epoch=train_steps,
                validation_data=val_gen,
                validation_steps=val_steps,
                epochs=EPOCHS,
                verbose=1,
                class_weight=class_weights,
                callbacks=[early_stopping, reduce_lr, checkpoint]
            )
            
            # Compute final metrics
            # Evaluate model on training data
            train_results = []
            for i in range(0, len(train_paths), BATCH_SIZE):
                if i + BATCH_SIZE <= len(train_paths):
                    batch_paths = train_paths[i:i + BATCH_SIZE]
                    batch_labels = train_labels[i:i + BATCH_SIZE]
                    
                    # Load images
                    batch_images = []
                    for path in batch_paths:
                        try:
                            img = cv2.imread(path)
                            if img is not None:
                                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
                                batch_images.append(img_resized)
                        except Exception:
                            pass
                    
                    if batch_images:
                        batch_images = np.array(batch_images)
                        batch_preds = model.predict(batch_images, verbose=0)
                        batch_correct = np.sum(np.argmax(batch_preds, axis=1) == np.argmax(batch_labels[:len(batch_images)], axis=1))
                        train_results.append((batch_correct, len(batch_images)))
            
            train_acc = sum(correct for correct, _ in train_results) / sum(total for _, total in train_results)
            
            # Evaluate model on validation data
            val_results = []
            for i in range(0, len(val_paths), BATCH_SIZE):
                if i + BATCH_SIZE <= len(val_paths):
                    batch_paths = val_paths[i:i + BATCH_SIZE]
                    batch_labels = val_labels[i:i + BATCH_SIZE]
                    
                    # Load images
                    batch_images = []
                    for path in batch_paths:
                        try:
                            img = cv2.imread(path)
                            if img is not None:
                                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
                                batch_images.append(img_resized)
                        except Exception:
                            pass
                    
                    if batch_images:
                        batch_images = np.array(batch_images)
                        batch_preds = model.predict(batch_images, verbose=0)
                        batch_correct = np.sum(np.argmax(batch_preds, axis=1) == np.argmax(batch_labels[:len(batch_images)], axis=1))
                        val_results.append((batch_correct, len(batch_images)))
            
            val_acc = sum(correct for correct, _ in val_results) / sum(total for _, total in val_results)
            
            fold_metrics.append({
                'train_acc': train_acc,
                'val_acc': val_acc
            })
            
            print(f"Fold {fold+1} results:")
            print(f"  Training accuracy: {train_acc:.4f}")
            print(f"  Validation accuracy: {val_acc:.4f}")
            
        except Exception as e:
            print(f"Error during training fold {fold+1}: {e}")
        
        # Clean up memory
        del model
        gc.collect()
        tf.keras.backend.clear_session()
    
    # Summarize cross-validation results
    print("\n=== Cross-validation results ===")
    for i, metrics in enumerate(fold_metrics):
        print(f"Fold {i+1}: train_acc={metrics['train_acc']:.4f}, val_acc={metrics['val_acc']:.4f}")
    
    avg_train_acc = np.mean([m['train_acc'] for m in fold_metrics])
    avg_val_acc = np.mean([m['val_acc'] for m in fold_metrics])
    print(f"\nAverage train accuracy: {avg_train_acc:.4f}")
    print(f"Average validation accuracy: {avg_val_acc:.4f}")
    
    # Train final lightweight model on full dataset
    print("\nTraining final model with subset of data...")
    
    # Use a subset of data for final model
    np.random.seed(42)
    subset_indices = np.random.choice(len(image_paths), min(3000, len(image_paths)), replace=False)
    subset_paths = [image_paths[i] for i in subset_indices]
    subset_labels = one_hot_labels[subset_indices]
    
    # Create generator
    final_gen = image_generator(subset_paths, subset_labels, BATCH_SIZE)
    final_steps = len(subset_paths) // BATCH_SIZE + (1 if len(subset_paths) % BATCH_SIZE > 0 else 0)
    
    # Build and train final model
    final_model = build_model()
    final_checkpoint = ModelCheckpoint(
        'final_model.keras',
        monitor='loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    final_model.fit(
        final_gen,
        steps_per_epoch=final_steps,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[reduce_lr, final_checkpoint]
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
                [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # Limit to 4GB
            )
        except RuntimeError as e:
            print(e)
    
    # Set memory limits in TensorFlow
    tf.keras.backend.set_floatx('float32')  # Use float32 instead of float64
    
    main()
