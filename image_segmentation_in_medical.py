
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

from google.colab import files

import zipfile

os.makedirs('/root/.kaggle', exist_ok=True)
!mv kaggle.json /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json

# Download the dataset (this is the resized 224x224 version)
!kaggle datasets download -d nih-chest-xrays/data

# Unzip the dataset
!unzip -q data.zip -d /content/nih_dataset

# Paths for Colab
base_dir = '/content/nih_dataset'
images_dir = os.path.join(base_dir, 'images-224/images-224')
bbox_csv_path = os.path.join(base_dir, 'BBox_List_2017_Official_NIH.csv')
metadata_csv_path = os.path.join(base_dir, 'Data_Entry_2017.csv')

# Load CSVs
bboxes_df = pd.read_csv(bbox_csv_path)
metadata_df = pd.read_csv(metadata_csv_path)

# Basic data exploration
print("Bounding Box DataFrame shape:", bboxes_df.shape)
print("Metadata DataFrame shape:", metadata_df.shape)

bboxes_df.head()

bboxes_df['Finding Label'].unique()

metadata_df.head()

# Focus on one disease for now
target_disease = 'Pneumothorax'
# Filter metadata for images with this disease
targets_df = bboxes_df[bboxes_df['Finding Label'].str.contains(target_disease)]

def create_mask(image_name, bboxes_df, img_size=(224, 224)):
    mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
    image_bboxes = bboxes_df[bboxes_df['Image Index'] == image_name]
    scale_factor = 224/1024
    for _, row in image_bboxes.iterrows():
        # Extract bbox coordinates
        x = int(int(row['Bbox [x'])*scale_factor)
        y = int(int(row['y'])*scale_factor)
        w = int(int(row['w'])*scale_factor)
        h = int(int(row['h]'])*scale_factor)
        # Draw rectangle on mask
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)

    return mask

# Loading images and their corresponding masks
def load_images_and_masks():
    images = []
    masks = []
    for idx, row in targets_df.iterrows():
        filename = row['Image Index']
        img_path = os.path.join(images_dir, filename)
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (224, 224))
        # Normalize
        img_resized = img_resized / 255.0
        images.append(img_resized)

        # Create mask from bounding boxes
        mask = create_mask(filename, bboxes_df)
        mask = mask/ 255.0
        masks.append(mask)
    return np.array(images), np.array(masks)

X, y = load_images_and_masks()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

import random

def show_samples_with_overlay(X_train, y_train, model=None, num_samples=3):

    indices = random.sample(range(len(X_train)), num_samples)

    for idx in indices:
        img = X_train[idx].squeeze()  # shape (H, W)
        mask = y_train[idx].squeeze()  # shape (H, W)

        plt.figure(figsize=(15, 5))

        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Input Image')
        plt.axis('off')

        # Plot ground truth mask
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

        # Plot overlay of mask on image
        if model:
            # Get prediction
            pred_mask = model.predict(X_train[idx][np.newaxis, ...])[0]
            overlay_mask = pred_mask.squeeze()
        else:
            overlay_mask = None

        # If model provided, overlay the predicted mask
        plt.subplot(1, 3, 3)
        plt.imshow(img, cmap='gray')
        if overlay_mask is not None:
            plt.imshow(overlay_mask, cmap='jet', alpha=0.5)
        else:
            plt.imshow(mask, cmap='jet', alpha=0.5)
        plt.title('Overlay' + (' with Prediction' if model else ''))
        plt.axis('off')

        plt.show()

show_samples_with_overlay(X_train, y_train, model=None, num_samples=5)

def preprocess_and_augment(image, mask):
    # Normalize
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0

    # Expand dims for compatibility
    image = tf.expand_dims(image, -1)
    mask = tf.expand_dims(mask, -1)

    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.1)

    return tf.squeeze(image, -1), tf.squeeze(mask, -1)

def load_dataset(X, y, augment=False, batch_size=16):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if augment:
        dataset = dataset.map(preprocess_and_augment, num_parallel_calls=AUTOTUNE)
    else:
        dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
    return dataset.shuffle(100).batch(batch_size).prefetch(AUTOTUNE)


train_ds = load_dataset(X_train, y_train, augment=True)
val_ds = load_dataset(X_val, y_val, augment=False)

def unet_model(input_size=(224, 224, 1)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.nn.sigmoid(y_pred)

    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

"""# Training"""

model = unet_model()
model.summary()

# Plotting the training & validation accuracy and loss
def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

"""## U-net with binary_crossentropy Loss"""

from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', dice_coef, tf.keras.metrics.MeanIoU(num_classes=2)]
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=16,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

# Plot training history
plot_training_history(history)

"""## U-net with Combined Loss"""

model = unet_model()

from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=combined_loss,
    metrics=['accuracy', dice_coef, tf.keras.metrics.MeanIoU(num_classes=2)]
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=16,
    callbacks=callbacks,
    epochs=50,
    verbose=1
)

plot_training_history(history)

"""# Evaluation"""

best_model = model

results = model.evaluate(X_val, y_val)
print(f'Loss: {results[0]:.4f}')
print(f'Accuracy: {results[1]:.4f}')
print(f'Mean IoU: {results[2]:.4f}')

