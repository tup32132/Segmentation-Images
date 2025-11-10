import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from pydicom import dcmread, dcmwrite
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, PYDICOM_IMPLEMENTATION_UID

import cv2
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

# --- CONFIGS

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1

DEFAULT_MODEL_PATH = "models/segmentation_model.keras"

DEFAULT_SAVE_PATH = "overlay.dcm"

# --- custom dice loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice

# --- create and populate dicom
def create_dicom(save_path, pixels):
    file_meta = FileMetaDataset()
    ds = FileDataset(save_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # uses dummy and default values
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian # common default
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7' # 'secondary capture', general purpose class
    file_meta.MediaStorageSOPInstanceUID = generate_uid() # unique id
    file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID # match meta header
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    ds.PatientName = "Test^Patient" # patient info
    ds.PatientID = "123456"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "SC" # secondary capture
    ds.SeriesNumber = "1"
    ds.InstanceNumber = "1"

    now = datetime.datetime.now() # current date and time
    ds.StudyDate = now.strftime('%Y%m%d')
    ds.StudyTime = now.strftime('%H%M%S.%f')
    ds.ContentDate = ds.StudyDate
    ds.ContentTime = ds.StudyTime

    ds.Rows = pixels.shape[0]
    ds.Columns = pixels.shape[1]
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB" # color
    ds.PlanarConfiguration = 0
    ds.PixelRepresentation = 0 # 0 for unsigned integer, 1 for signed
    ds.BitsAllocated = 8 # using uint16
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.RescaleSlope = "1"
    ds.RescaleIntercept = "0"

    ds.PixelData = pixels.tobytes() # the actual image

    try:
        dcmwrite(save_path, ds, write_like_original=False)
        print(f"wrote dicom file to {save_path}")

        # --- 4. (Optional) Verification Step ---
        print(f"verifying...", end="")
        # Read the file back and check if the pixel data matches
        ds_read = dcmread(save_path)
        
        # The .pixel_array attribute automatically converts the raw bytes back to a NumPy array
        read_pixels = ds_read.pixel_array
        
        if np.array_equal(pixels, read_pixels):
            print("verified")
        else:
            print("verification failed")
        
        # plt.figure(figsize=(20, 10)) # see dicoms anyways

        # plt.subplot(1, 2, 1) # og
        # plt.title("og") 
        # plt.imshow(pixels)
        # plt.axis('off')

        # plt.subplot(1, 2, 2) # read pixels
        # plt.title("verified") 
        # plt.imshow(read_pixels)
        # plt.axis('off')

        # plt.show()

    except Exception as e:
        print(f"error, file not saved\n{e}")

    return ds

# --- main
def main(image_path, model_path, save_path):

    # load model
    print(f"loading model from {model_path}")
    try:
        trained_model = keras.models.load_model( 
            "models/test_generated.keras",
            custom_objects={"dice_loss": dice_loss}
        )
    except IOError:
        print(f"model not found at {model_path}")
        return
    except KeyError as e:
        print(f"error\n{e}")
        return
    
    # preprocess image
    print(f"loading image from {image_path}")
    ds = dcmread(image_path)
    img = ds.pixel_array.astype(np.float32)
    img = np.expand_dims(img, axis=-1)

    img_tensor = tf.convert_to_tensor(img)
    img_resized = tf.image.resize_with_pad(img_tensor, IMG_HEIGHT, IMG_WIDTH)
    
    if "RescaleSlope" in ds and "RescaleIntercept" in ds:
        img_resized = img_resized * ds.RescaleSlope + ds.RescaleIntercept
    min_val, max_val = np.min(img_resized), np.max(img_resized)
    if max_val - min_val > 1e-6: 
        img_normalized = (img_resized - min_val) / (max_val - min_val)
    else:
        img_normalized = img_resized - min_val
    img_batch = tf.expand_dims(img_normalized, axis=0) # batch of 1

    # create model
    grad_model = Model(
        trained_model.inputs,
        [trained_model.output]
    )

    # calculate gradients
    model_output = grad_model(img_batch)

    # get mask prediction
    print("generating segmentation mask")
    unprocessed_pred_mask = model_output[0].numpy()
    unprocessed_pred_mask_color = np.uint8(255 * unprocessed_pred_mask)
    unprocessed_pred_mask_color = cv2.applyColorMap(unprocessed_pred_mask_color, cv2.COLORMAP_JET)
    pred_mask = (unprocessed_pred_mask > 0.5).astype(np.uint8)

    # create overlay
    img_overlay = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    overlay = cv2.cvtColor(img_overlay.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    overlay_resized = cv2.resize(overlay, (IMG_WIDTH, IMG_HEIGHT))
    heatmap_resized = cv2.resize(unprocessed_pred_mask_color, (IMG_WIDTH, IMG_HEIGHT))
    heatmap_color = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)

    heatmap_overlay = cv2.addWeighted(overlay_resized, 0.7, heatmap_color, 0.3, 0)

    # saving dicom file
    print(f"saving dicom files at {save_path}")
    ds = create_dicom(save_path, heatmap_overlay)

    # display
    print(f"displaying results for {os.path.basename(image_path)}")
    print()

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 4, 1) # og
    plt.title("xray") 
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2) # prediction mask
    plt.title("predicted mask")
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3) # heatmap
    plt.title("probability heatmap")
    plt.imshow(unprocessed_pred_mask_color)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("heatmap overlay") # heatmap overlayed with image
    plt.imshow(heatmap_overlay) 
    plt.axis('off')

    plt.show()

# --- command line 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="run and view segmentation on xray images"
    )
    
    # image path
    parser.add_argument(
        "image", 
        type=str, 
        help="path to single input image"
    )
    
    # model path
    parser.add_argument(
        "-model", 
        type=str, 
        default=DEFAULT_MODEL_PATH, 
        help=f"path to trained .keras model (default {DEFAULT_MODEL_PATH})"
    )

    # save path
    parser.add_argument(
        "-save", 
        type=str, 
        default=DEFAULT_SAVE_PATH, 
        help=f"path to save directory (default {DEFAULT_SAVE_PATH})"
    )
    
    args = parser.parse_args()

    # validate
    if not os.path.exists(args.image):
        print(f"image dne at {args.image_path}")
        exit()
        
    if not os.path.exists(args.model):
        print(f"model dne at {args.model_path}")
        exit()

    if not os.path.exists(args.save):
        print(f"saving at {DEFAULT_SAVE_PATH}")

    # run
    main(args.image, args.model, args.save)