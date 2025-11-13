# imports

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose,
    concatenate, BatchNormalization, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pydicom

from configs import *

### target file system

# data/
# ———> train/
# ———> validate/
# ———> test/
# models/
# outputs/

### config

SOURCE_DIR = "data/play"
TARGET_DIR = ""
COPY_DATA_TO_TARGET = False # make copies of data?

MODEL_DIR = "models"
MODEL = "test.keras" # always .keras

TRAIN_IMGS = "data/train" 
TRAIN_MSKS = "data/train"
VAL_IMGS = "data/validate"
VAL_MSKS = "data/validate"
TEST_IMGS = "data/test"
TEST_MSKS = "data/test"
# OUTPUT_DIR = "outputs"

R_TRAIN = 0.8
R_VAL = 0.2
# R_TEST = 0.0

IMG_WIDTH = 256 # this may need to be higher depending on dicom images
IMG_HEIGHT = 256
IMG_CHANNELS = 1
BATCH_SIZE = 16 # default=16, could be higher if running on cloud
BUFFER_SIZE = 100 # depends on training set size

STEP = 1e-2 # learning rate, default=1e-4
EPOCHS = 2 # does fewer data mean fewer epochs? default=5? 25?
LAYERS = 4 # default=4

### create directories

if not TARGET_DIR:
    TARGET_DIR = os.getcwd()
os.makedirs(os.path.join(TARGET_DIR, TRAIN_IMGS), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, TRAIN_MSKS), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, VAL_IMGS), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, VAL_MSKS), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, TEST_IMGS), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, TEST_MSKS), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, MODEL_DIR), exist_ok=True)

### train test split

class UnequalImageMaskError(Exception): pass
try:
    if SOURCE_DIR:

        img_fs = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.dcm')]
        msk_fs = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.npy')]

        if len(img_fs) != len(msk_fs):
            raise UnequalImageMaskError("image and mask list lengths are not equal")
except UnequalImageMaskError as uime:
    print(uime)
    print("checking for multiple masks")

    adds = list()
    for f in img_fs:
        fname = f.removesuffix('.dcm') # apply more transforms if needed
        corr_masknames = [fm for fm in msk_fs if fname.strip() in fm]
        if len(corr_masknames) < 1:
            raise UnequalImageMaskError(f"image {f} has no corresponding mask")
        for cm in corr_masknames[1:]:
            adds.append(str(f))
            print(f"duplicating {f} in training set")
            # msk_fs.remove(cm)
            # print(f"removing {cm} from training set")
    img_fs.extend(adds)

if SOURCE_DIR:
    seed = 42
    labels = None # we do have classes
    train_img_fs, val_img_fs = train_test_split(
        sorted(img_fs), 
        train_size=R_TRAIN, test_size=R_VAL, 
        random_state=seed,
        stratify=labels
    )
    train_msk_fs, val_msk_fs = train_test_split(
        sorted(msk_fs), 
        train_size=R_TRAIN, test_size=R_VAL, 
        random_state=seed,
        stratify=labels
    )

    if (len(train_img_fs) != len(train_msk_fs)) or (len(val_img_fs) != len(val_msk_fs)):
        raise UnequalImageMaskError("image and mask list lengths are **still** not equal")
    

if SOURCE_DIR and COPY_DATA_TO_TARGET:
    for f in train_img_fs: shutil.copy2(
        os.path.join(SOURCE_DIR, f),
        os.path.join(TARGET_DIR, TRAIN_IMGS, f)
    )
    for f in val_img_fs: shutil.copy2(
        os.path.join(SOURCE_DIR, f),
        os.path.join(TARGET_DIR, VAL_IMGS, f)
    )
    for f in train_msk_fs: shutil.copy2(
        os.path.join(SOURCE_DIR, f),
        os.path.join(TARGET_DIR, TRAIN_MSKS, f)
    )
    for f in val_msk_fs: shutil.copy2(
        os.path.join(SOURCE_DIR, f),
        os.path.join(TARGET_DIR, VAL_MSKS, f)
    )
        
### FUNC load an image

def _load_dicom_and_numpy(image_path, mask_path):
    image_path = image_path.numpy().decode('utf-8')
    mask_path = mask_path.numpy().decode('utf-8')

    # image
    ds = pydicom.dcmread(image_path)
    img = ds.pixel_array.astype(np.float32)

    # standardize channels
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    n_channels = img.shape[2]
    if n_channels == IMG_CHANNELS: # if correct, do nothing
        pass
    elif n_channels == 1: # multiply
        img = np.tile(img, (1, 1, IMG_CHANNELS))
    elif n_channels == 3: # reduce to 1 then multiply
        img = np.mean(img, axis=-1, keepdims=True)
        img = np.tile(img, (1, 1, IMG_CHANNELS))
    elif n_channels < IMG_CHANNELS: # add padding
        padding = np.zeros(img.shape[:2] + (IMG_CHANNELS-n_channels,), dtype=img.dtype) 
        img = np.concatenate([img, padding], axis=-1)
    elif n_channels > IMG_CHANNELS: # slice down
        img = img[..., :IMG_CHANNELS] 

    # normalize image
    if "RescaleSlope" in ds and "RescaleIntercept" in ds:
        img = img * ds.RescaleSlope + ds.RescaleIntercept
    min_val, max_val = np.min(img), np.max(img)
    if max_val - min_val > 1e-6: 
        img = (img - min_val) / (max_val - min_val)
    else:
        img = img - min_val

    # mask
    mask = np.load(mask_path).astype(np.float32)
    mask = np.clip(mask / 255.0, 0.0, 1.0) # ensure mask is binary

    # add channel dim
    mask = np.expand_dims(mask, axis=-1)

    return img, mask

def load_and_preprocess(image_path, mask_path):
    img, mask = tf.py_function( # wrapper for _load_dicom_and_numpy
        _load_dicom_and_numpy,
        [image_path, mask_path],
        [tf.float32, tf.float32]
    )
    
    img.set_shape([None, None, IMG_CHANNELS])
    img = tf.image.resize_with_pad(img, IMG_HEIGHT, IMG_WIDTH)
    img.set_shape([IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])

    mask.set_shape([None, None, 1])
    mask = tf.image.resize_with_pad(mask, IMG_HEIGHT, IMG_WIDTH, method='nearest')
    mask.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])
    
    return img, mask

### FUNC augment image

def augment_data(img, mask):
    if tf.random.uniform(()) > 0.5: # randomly flip image
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32) # randomly rotate 90*
    img = tf.image.rot90(img, k=k)
    mask = tf.image.rot90(mask, k=k)
    return img, mask

### training pipeline

if COPY_DATA_TO_TARGET:
    train_image_paths = sorted([os.path.join(TARGET_DIR, TRAIN_IMGS, f) for f in os.listdir(os.path.join(TARGET_DIR, TRAIN_IMGS)) if f.endswith(".dcm")])
    train_mask_paths = sorted([os.path.join(TARGET_DIR, TRAIN_MSKS, f) for f in os.listdir(os.path.join(TARGET_DIR, TRAIN_MSKS)) if f.endswith(".npy")])
else:
    train_image_paths = sorted([os.path.join(SOURCE_DIR, f) for f in train_img_fs if f.endswith('.dcm')]) # sorted([os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if f.endswith(".dcm") and f in train_img_fs])
    train_mask_paths = sorted([os.path.join(SOURCE_DIR, f) for f in train_msk_fs if f.endswith('.npy')]) # sorted([os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if f.endswith(".npy") and f in train_msk_fs])

train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mask_paths))
train_dataset = train_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.cache() # optional
train_dataset = train_dataset.shuffle(BUFFER_SIZE) # unsure how much data you have, hence buffer and batch
train_dataset = train_dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

### validation pipeline

if COPY_DATA_TO_TARGET:
    val_image_paths = sorted([os.path.join(TARGET_DIR, VAL_IMGS, f) for f in os.listdir(os.path.join(TARGET_DIR, VAL_IMGS)) if f.endswith(".dcm")])
    val_mask_paths = sorted([os.path.join(TARGET_DIR, VAL_MSKS, f) for f in os.listdir(os.path.join(TARGET_DIR, VAL_MSKS)) if f.endswith(".npy")])
else:
    val_image_paths = sorted([os.path.join(SOURCE_DIR, f) for f in val_img_fs if f.endswith('.dcm')]) # sorted([os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if f.endswith(".dcm") and f in val_img_fs])
    val_mask_paths = sorted([os.path.join(SOURCE_DIR, f) for f in val_msk_fs if f.endswith('.npy')]) # sorted([os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if f.endswith(".npy") and f in val_msk_fs])

val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_mask_paths))
val_dataset = val_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

### display sample image

try:
    for images, masks in train_dataset.take(1):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("sample image")
        plt.imshow(images[0, :, :, 0], cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title("sample mask")
        plt.imshow(masks[0, :, :, 0], cmap='gray')

        # plt.savefig(f"view_{1}")
        plt.show()
except Exception as e:
    print(e)

### define model

def encoder(inputs, num_filters): # encoder block
    c = Conv2D(num_filters, 3, activation='relu', padding='same')(inputs) 
    c = BatchNormalization()(c)
    c = Conv2D(num_filters, 3, activation='relu', padding='same')(c)
    p = MaxPooling2D(pool_size=(2, 2))(c)
    return c, p

def bottleneck(inputs, num_filters): # final convolution before decoder
    b = Conv2D(num_filters, 3, activation='relu', padding='same')(inputs)
    b = BatchNormalization()(b)
    b = Conv2D(num_filters, 3, activation='relu', padding='same', name="final_convolution_before_decoder")(b)
    return b

def decoder(inputs, skip_connection, num_filters): # decoder block
    d = Conv2DTranspose(num_filters, 2, strides=2, padding='same')(inputs)
    d = concatenate([d, skip_connection])
    d = Conv2D(num_filters, 3, activation='relu', padding='same')(d)
    d = BatchNormalization()(d)
    d = Conv2D(num_filters, 3, activation='relu', padding='same')(d)
    return d

c = {}
p = {0: Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))}

for layer in range(1, 1 + LAYERS): c[layer], p[layer] = encoder(p[layer-1], 2**(5+layer))

d = {0: bottleneck(p[LAYERS], 2**(6+LAYERS))}

for layer in range(LAYERS): d[layer+1] = decoder(d[layer], c[LAYERS-layer], 2**(5+LAYERS-layer))

model = Model(
    inputs=p[0], 
    outputs=Conv2D(1, 1, padding='same', activation ='sigmoid', name='output_mask')(d[LAYERS]), 
    name="UNET"
)
model.summary()

### FUNC loss

# consider focal loss
# dice loss commonly used for segmentation to reduce overlap / account for small foreground object
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice

### compile model

# Adam is default, consider stochastic gradient descent for more control over learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=STEP),
    loss=dice_loss,
    metrics=['accuracy']
)

### train model

history = model.fit(
    train_dataset, # .take(2)
    validation_data=val_dataset,
    epochs=EPOCHS,
)
model.save(os.path.join(TARGET_DIR, MODEL_DIR, MODEL))

### see training

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label="training loss")
if 'val_loss' in history.history: 
    plt.plot(history.history['val_loss'], label="validation loss")
plt.title("dice loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label="training accuracy")
if 'val_accuracy' in history.history: 
    plt.plot(history.history['val_accuracy'], label="validation accuracy")
plt.title("pixel accuracy")
plt.legend()

# plt.savefig("history")
plt.show()

