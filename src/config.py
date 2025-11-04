# --- config.py ---
# This file holds all your project settings.

# --- 1. GLOBAL SETTINGS ---
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 1  # Grayscale
NUM_CLASSES = 3   # 0: Background, 1: RSO (bad), 2: IMD (good)

# --- 2. TRAINING SETTINGS ---
BATCH_SIZE = 4    # Start small (e.g., 2 or 4) based on your GPU memory
EPOCHS = 50       # How long to train for
STEPS_PER_EPOCH = 200 # How many batches to generate per epoch

CLUSTER_ASSETS = [
    "peanut_sponges_asset.png",
    "button_sponge_asset.png" # Add any other small assets
]

# --- 3. ASSET LIBRARY PATHS (UPDATE THESE) ---
BG_DIR = "C:/Users/KyleKinney/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3/images_001/images"
RSO_DIR = "C:/Users/KyleKinney/Pictures/RetainedInstruments/Assets"
IMD_DIR = "C:/Users/KyleKinney/Pictures/RetainedInstruments/Assets"

# --- 4. SAVED MODEL ---
MODEL_SAVE_PATH = "rso_detector_model_v1.h5"
