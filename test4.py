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

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

PATH = "../lung_colon_image_set/lung_image_sets"
IMG_SIZE = 128 
SPLIT = 0.2
EPOCHS = 10
BATCH_SIZE = 16

class BalancedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=16, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(image_paths))
        
        self.class_counts = np.sum(labels, axis=0)
        self.class_weights = np.sum(self.class_counts) / (len(self.class_counts) * self.class_counts)
        self.sample_weights = np.zeros(len(labels))
        
        for i in range(len(labels)):
            class_idx = np.argmax(labels[i])
            self.sample_weights[i] = self.class_weights[class_idx]
            
        self.sample_weights = self.sample_weights / np.sum(self.sample_weights) * len(self.sample_weights)
        
        np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        if idx == 0:
            p = self.sample_weights / np.sum(self.sample_weights)
            self.indices = np.random.choice(
                len(self.image_paths), 
                size=len(self.image_paths), 
                replace=False, 
                p=p
            )
            
        batch_indices = self.indices[idx * self.batch_size:min((idx + 1) * self.batch_size, len(self.image_paths))]
        
        batch_images = []
        batch_labels = []
        
        for i in batch_indices:
            try:
                img = cv2.imread(self.image_paths[i])
                if img is None:
                    continue
                
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
                
                if self.augment:
                    if np.random.random() > 0.5:
                        img_resized = cv2.flip(img_resized, 1)
                    
                    if np.random.random() > 0.5:
                        img_resized = cv2.flip(img_resized, 0)
                        
                    if np.random.random() > 0.5:
                        brightness = np.random.uniform(0.8, 1.2)
                        img_resized = np.clip(img_resized * brightness, 0, 1)
                
                batch_images.append(img_resized)
                batch_labels.append(self.labels[i])
            except Exception as e:
                print(f"Error processing image {self.image_paths[i]}: {e}")
                continue
        
        if not batch_images:
            dummy_img = np.zeros((1, IMG_SIZE, IMG_SIZE, 3))
            dummy_label = np.zeros((1, self.labels.shape[1]))
            return dummy_img, dummy_label
            
        return np.array(batch_images), np.array(batch_labels)
    
    def on_epoch_end(self):
        pass

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

def lr_schedule(epoch, lr):
    if epoch > 5:
        return lr * 0.5
    return lr

def main():
    """Main function with memory-efficient implementation."""
    print("Processing data...")
    
    image_paths = []
    labels = []
    
    catalogs = sorted(os.listdir(PATH))
    print(f"Categories found: {catalogs}")
    
    for i, catalog in enumerate(catalogs):
        catalog_images = glob(f"{PATH}/{catalog}/*.jpeg")
        print(f"Found {len(catalog_images)} images in category {catalog}")
        
        image_paths.extend(catalog_images)
        labels.extend([i] * len(catalog_images))
    
    unique_labels = np.unique(labels)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    indices = [label_to_index[label] for label in labels]
    one_hot_labels = np.eye(len(unique_labels))[indices]
    
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    class_weights = {cls: total_samples / (len(unique_classes) * count) 
                    for cls, count in zip(unique_classes, class_counts)}
    print(f"Class weights: {class_weights}")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    all_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
        print(f"\n=== Training fold {fold+1}/5 ===")
        
        train_paths = [image_paths[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        train_labels = one_hot_labels[train_idx]
        val_labels = one_hot_labels[val_idx]
        
        model = build_model()
        if fold == 0:
            model.summary()
        
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
        
        train_gen = BalancedDataGenerator(train_paths, train_labels, BATCH_SIZE, augment=True)
        val_gen = BalancedDataGenerator(val_paths, val_labels, BATCH_SIZE, augment=False)
        
        try:
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=EPOCHS,
                verbose=1,
                callbacks=[early_stopping, reduce_lr, checkpoint, lr_scheduler]
            )
            
            all_histories.append(history.history)
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title(f'Model Accuracy - Fold {fold+1}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(['Train', 'Validation'], loc='lower right')
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title(f'Model Loss - Fold {fold+1}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Train', 'Validation'], loc='upper right')
            
            plt.tight_layout()
            plt.savefig(f'model_performance_fold_{fold+1}.png')
            plt.close()
            
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
        
        del model, train_gen, val_gen
        gc.collect()
        tf.keras.backend.clear_session()
    
    print("\n=== Cross-validation results ===")
    for i, metrics in enumerate(fold_metrics):
        print(f"Fold {i+1}: train_acc={metrics['train_acc']:.4f}, val_acc={metrics['val_acc']:.4f}")
    
    if fold_metrics:
        avg_train_acc = np.mean([m['train_acc'] for m in fold_metrics])
        avg_val_acc = np.mean([m['val_acc'] for m in fold_metrics])
        print(f"\nAverage train accuracy: {avg_train_acc:.4f}")
        print(f"Average validation accuracy: {avg_val_acc:.4f}")
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        for i, history in enumerate(all_histories):
            plt.plot(history['accuracy'], alpha=0.3, color='blue')
            plt.plot(history['val_accuracy'], alpha=0.3, color='orange')
        
        avg_train_acc = np.mean([history['accuracy'] for history in all_histories], axis=0)
        avg_val_acc = np.mean([history['val_accuracy'] for history in all_histories], axis=0)
        plt.plot(avg_train_acc, linewidth=2, color='blue')
        plt.plot(avg_val_acc, linewidth=2, color='orange')
        
        plt.title('Average Model Accuracy Across Folds')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        for i, history in enumerate(all_histories):
            plt.plot(history['loss'], alpha=0.3, color='blue')
            plt.plot(history['val_loss'], alpha=0.3, color='orange')
        
        avg_train_loss = np.mean([history['loss'] for history in all_histories], axis=0)
        avg_val_loss = np.mean([history['val_loss'] for history in all_histories], axis=0)
        plt.plot(avg_train_loss, linewidth=2, color='blue')
        plt.plot(avg_val_loss, linewidth=2, color='orange')
        
        plt.title('Average Model Loss Across Folds')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig('average_model_performance.png')
        plt.close()
    
    print("\nTraining final model with subset of data...")
    
    np.random.seed(42)
    max_images = 3000
    subset_size = min(max_images, len(image_paths))
    subset_indices = np.random.choice(len(image_paths), subset_size, replace=False)
    subset_paths = [image_paths[i] for i in subset_indices]
    subset_labels = one_hot_labels[subset_indices]
    
    final_gen = BalancedDataGenerator(subset_paths, subset_labels, BATCH_SIZE, augment=True)
    
    final_model = build_model()
    final_checkpoint = ModelCheckpoint(
        'final_model.keras',
        monitor='loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    try:
        final_history = final_model.fit(
            final_gen,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[reduce_lr, lr_scheduler, final_checkpoint]
        )
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(final_history.history['accuracy'])
        plt.title('Final Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.subplot(1, 2, 2)
        plt.plot(final_history.history['loss'])
        plt.title('Final Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('final_model_performance.png')
        plt.close()
        
        print("Final model saved successfully!")
        
    except Exception as e:
        print(f"Error training final model: {e}")

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
            )
        except RuntimeError as e:
            print(e)
    
    tf.keras.backend.set_floatx('float32')
    
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    main()
