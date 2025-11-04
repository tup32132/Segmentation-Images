# --- export_dataset.py ---
# This script "makes a bunch" of synthetic images and saves them
# to a structured folder, ready to be sent to your coworker.

import os
import numpy as np
from PIL import Image
import cv2 # For saving the mask
from tqdm import tqdm # A nice progress bar

# Import our custom modules
try:
    import config
    # We only need the single-sample generator
    from data_generator import create_synthetic_sample
except ImportError:
    print("Error: Could not import config.py or data_generator.py.")
    print("Make sure they are all in the same directory.")
    exit()

# --- 1. SETTINGS: How many images to make? ---
NUM_TRAIN_SAMPLES = 5000
NUM_VAL_SAMPLES = 500

BASE_OUTPUT_DIR = "generated_dataset"
# -----------------------------------------------

def setup_directories():
    """Creates the train/val folder structure."""
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, "train", "masks"), exist_ok=True)
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, "validation", "images"), exist_ok=True)
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, "validation", "masks"), exist_ok=True)
    print(f"Created directory structure at '{BASE_OUTPUT_DIR}'")

def generate_and_save_dataset(num_samples, split_name):
    """
    Loops `num_samples` times, generates a new sample,
    and saves it to the correct folder.
    
    Args:
        num_samples (int): Number of images to generate.
        split_name (str): "train" or "validation"
    """
    print(f"Generating {num_samples} samples for '{split_name}'...")
    
    img_dir = os.path.join(BASE_OUTPUT_DIR, split_name, "images")
    mask_dir = os.path.join(BASE_OUTPUT_DIR, split_name, "masks")
    
    for i in tqdm(range(num_samples)):
        # 1. Generate one sample
        # (Image is 0-1 float, mask is 0, 1, 2 int)
        image_np, mask_np = create_synthetic_sample((config.IMG_HEIGHT, config.IMG_WIDTH))
        
        # 2. Define file paths
        filename = f"sample_{i:05d}.png"
        img_path = os.path.join(img_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        
        # 3. Save the image
        # Convert from 0-1 float to 0-255 uint8
        image_to_save = (image_np.squeeze() * 255).astype(np.uint8)
        img_pil = Image.fromarray(image_to_save, 'L')
        img_pil.save(img_path)
        
        # 4. Save the mask
        # Masks (0, 1, 2) must be saved as-is, so we use cv2.imwrite
        # PIL can sometimes mess up single-channel integer values
        cv2.imwrite(mask_path, mask_np.astype(np.uint8))

if __name__ == "__main__":
    # 1. Create all the folders
    setup_directories()
    
    # 2. Generate the training set
    generate_and_save_dataset(NUM_TRAIN_SAMPLES, "train")
    
    # 3. Generate the validation set
    generate_and_save_dataset(NUM_VAL_SAMPLES, "validation")
    
    print("\n--- DATASET EXPORT COMPLETE ---")
    print(f"Your dataset is ready in the '{BASE_OUTPUT_DIR}' folder.")
    print("You can now zip this folder and send it to your coworker.")
