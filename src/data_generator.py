# --- data_generator.py ---
# This is your "factory" script for creating synthetic data.
# VERSION 6.1: Added robust error handling to prevent crashes.

import albumentations as A
import numpy as np
import cv2  # OpenCV, a dependency for Albumentations
import os
import random
import io
from PIL import Image, ImageChops

# Import settings from your config file
try:
    import config
except ImportError:
    print("Error: Could not import config.py. Make sure it's in the same directory.")
    exit()

# --- 1. SETUP: Load File Paths ---
try:
    background_files = [os.path.join(config.BG_DIR, f) for f in os.listdir(config.BG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    rso_files = [os.path.join(config.RSO_DIR, f) for f in os.listdir(config.RSO_DIR) if f.endswith('_asset.png')]
    imd_files = [os.path.join(config.IMD_DIR, f) for f in os.listdir(config.IMD_DIR) if f.endswith('_asset.png')]

    if not background_files:
        print(f"FATAL: No background files found in {config.BG_DIR}. Stopping.")
        exit()
    if not rso_files:
        print(f"Warning: No RSO asset files found in {config.RSO_DIR}.")
    if not imd_files:
        print(f"Warning: No IMD asset files found in {config.IMD_DIR}.")

except FileNotFoundError as e:
    print(f"Error: A directory was not found. Did you update config.py?")
    print(e)
    exit()
except Exception as e:
    print(f"An unexpected error occurred loading file paths: {e}")
    exit()


# --- 2. AUGMENTATION PIPELINE (Albumentations) ---
transform = A.Compose([
    A.RandomScale(scale_limit=(-0.8, -0.3), p=1.0, interpolation=cv2.INTER_NEAREST), 
    A.Rotate(limit=45, p=0.7, border_mode=cv2.BORDER_CONSTANT),
    A.Perspective(scale=(0.1, 0.3), p=0.7, border_mode=cv2.BORDER_CONSTANT),
    A.RandomBrightnessContrast(brightness_limit=(0.1, 0.4), contrast_limit=(0.1, 0.4), p=0.8),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5)
])

# --- 3. HELPER FUNCTION: JPEG "Dirtying" ---
def apply_jpeg_artifacts(pil_image, quality=90):
    try:
        buffer = io.BytesIO()
        pil_image.convert('L').save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)
    except Exception as e:
        print(f"Error in apply_jpeg_artifacts: {e}. Returning original image.")
        return pil_image.convert('L')

# --- 4. SINGLE SAMPLE GENERATOR ---
def create_synthetic_sample(target_size=(config.IMG_HEIGHT, config.IMG_WIDTH)):
    """Generates one (dirty_image, final_mask) training pair."""
    
    try:
        # 1. Load and resize background
        bg_path = random.choice(background_files)
        background = Image.open(bg_path).convert('L').resize(target_size)
        final_mask = Image.new('L', target_size, 0) # 0 = background class
        bg_np = np.array(background) # Convert to numpy for processing
    except Exception as e:
        print(f"CRITICAL: Failed to load background image: {bg_path}. Error: {e}")
        # Return empty images to avoid crashing the trainer
        return (
            np.zeros((target_size[0], target_size[1], 1)), 
            np.zeros((target_size[0], target_size[1]))
        )

    # --- (Fix 3) Find the "Body" using Contour Finding ---
    valid_coords = []
    try:
        _, binary_mask = cv2.threshold(bg_np, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Only use contours that are a reasonable size (e.g., > 1% of image)
            if cv2.contourArea(largest_contour) > (target_size[0] * target_size[1] * 0.01):
                body_mask = np.zeros_like(binary_mask)
                cv2.drawContours(body_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                valid_coords = np.argwhere(body_mask > 0)
    except Exception as e:
        print(f"Warning: Contour finding failed for {bg_path}. Error: {e}. Falling back to threshold.")
        valid_coords = [] # Ensure it's empty to trigger failsafe

    if len(valid_coords) == 0:
        # Failsafe 1: simple threshold
        valid_coords = np.argwhere(bg_np > 20)
    if len(valid_coords) == 0:
        # Failsafe 2: all pixels
        print(f"Warning: No valid body pixels found for {bg_path}. Pasting anywhere.")
        valid_coords = np.argwhere(np.ones_like(bg_np))
    
    # 2. Decide what to add
    decision = random.choice(["normal", "rso", "imd", "both"])
    objects_to_add = []
    
    if (decision == "rso" or decision == "both") and rso_files:
        obj_path = random.choice(rso_files)
        mask_path = obj_path.replace("_asset.png", "_mask.png")
        is_clusterable = os.path.basename(obj_path) in config.CLUSTER_ASSETS
        objects_to_add.append({"obj_path": obj_path, "mask_path": mask_path, "class_id": 1, "is_clusterable": is_clusterable})

    if (decision == "imd" or decision == "both") and imd_files:
        obj_path = random.choice(imd_files)
        mask_path = obj_path.replace("_asset.png", "_mask.png")
        is_clusterable = os.path.basename(obj_path) in config.CLUSTER_ASSETS
        objects_to_add.append({"obj_path": obj_path, "mask_path": mask_path, "class_id": 2, "is_clusterable": is_clusterable})

    paste_canvas = Image.new('L', background.size, 0)

    # 3. Paste objects
    for obj in objects_to_add:
        try:
            asset_img_cv = cv2.imread(obj["obj_path"], cv2.IMREAD_UNCHANGED) 
            asset_mask_cv = cv2.imread(obj["mask_path"], cv2.IMREAD_GRAYSCALE)
            
            if asset_img_cv is None or asset_mask_cv is None: 
                print(f"Warning: Could not read asset/mask for {obj['obj_path']}. Skipping.")
                continue 

            if obj["is_clusterable"]:
                num_rows = random.randint(2, 3)
                num_cols = random.randint(2, 3)
                num_pastes = num_rows * num_cols
                spacing = random.randint(15, 30)
                grid_center_y, grid_center_x = random.choice(valid_coords)
                grid_angle = random.uniform(-15, 15)
                M_grid = cv2.getRotationMatrix2D((0,0), grid_angle, 1)
            else:
                num_pastes = 1 
            
            for i in range(num_pastes):
                transformed = transform(image=asset_img_cv, mask=asset_mask_cv)
                aug_asset_cv = transformed['image']
                aug_mask_cv = transformed['mask']

                aug_asset_pil = Image.fromarray(aug_asset_cv, 'RGBA')
                aug_mask_pil = Image.fromarray(aug_mask_cv, 'L')

                asset_texture = aug_asset_pil.convert('L')
                new_intensity = random.uniform(0.65, 0.95)
                asset_texture = asset_texture.point(lambda p: p * new_intensity)
                
                if obj["is_clusterable"]:
                    row = i // num_cols
                    col = i % num_cols
                    x_offset = (col - (num_cols - 1) / 2.0) * spacing
                    y_offset = (row - (num_rows - 1) / 2.0) * spacing
                    x_offset += random.randint(-5, 5)
                    y_offset += random.randint(-5, 5)
                    rotated_offset = M_grid.dot(np.array([x_offset, y_offset, 1]))
                    paste_x = int(grid_center_x + rotated_offset[0] - (aug_asset_pil.width // 2))
                    paste_y = int(grid_center_y + rotated_offset[1] - (aug_asset_pil.height // 2))
                else:
                    center_y, center_x = random.choice(valid_coords)
                    paste_x = center_x - (aug_asset_pil.width // 2)
                    paste_y = center_y - (aug_asset_pil.height // 2)

                # --- Added safety check for paste operation ---
                if paste_x < -aug_asset_pil.width or paste_x > target_size[0] or \
                   paste_y < -aug_asset_pil.height or paste_y > target_size[1]:
                   print(f"Warning: Calculated paste coordinates ({paste_x}, {paste_y}) are way off-screen. Skipping paste.")
                   continue

                paste_canvas.paste(asset_texture, (paste_x, paste_y), aug_mask_pil)
                mask_to_paste = Image.new('L', aug_mask_pil.size, obj["class_id"])
                final_mask.paste(mask_to_paste, (paste_x, paste_y), aug_mask_pil)

        except Exception as e:
            print(f"---!!! UNEXPECTED ERROR during paste loop for {obj['obj_path']} !!!---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 7. Blend the background with the canvas
    background = ImageChops.screen(background, paste_canvas)

    # 9. Apply JPEG "Dirtying"
    final_image = apply_jpeg_artifacts(background)
    
    # 10. Return final numpy arrays
    final_image_np = np.array(final_image).reshape(target_size[0], target_size[1], 1)
    final_mask_np = np.array(final_mask)
    
    return final_image_np / 255.0, final_mask_np

# --- 5. THE KERAS-COMPATIBLE GENERATOR ---
def get_data_generator(batch_size):
    """Yields batches of synthetic data forever."""
    while True:
        try:
            batch_images = np.zeros((batch_size, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS))
            batch_masks = np.zeros((batch_size, config.IMG_HEIGHT, config.IMG_WIDTH))
            
            for i in range(batch_size):
                img, mask = create_synthetic_sample((config.IMG_HEIGHT, config.IMG_WIDTH))
                batch_images[i] = img
                batch_masks[i] = mask
                
            yield batch_images, batch_masks
        except Exception as e:
            print(f"---!!! FATAL ERROR IN data_generator loop !!!---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            # Yield empty data to prevent trainer from crashing
            yield (
                np.zeros((batch_size, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)), 
                np.zeros((batch_size, config.IMG_HEIGHT, config.IMG_WIDTH))
            )


# --- 6. TEST BLOCK ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Testing the data generator... (V6.1: Robust Error Handling)")
    
    my_generator = get_data_generator(batch_size=4)
    images, masks = next(my_generator)
    
    print(f"Batch image shape: {images.shape}")
    print(f"Batch mask shape: {masks.shape}")
    print(f"Classes in mask 0: {np.unique(masks[0])}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(images[0].squeeze(), cmap='gray')
    ax[0].set_title("Synthetic Image")
    ax[1].imshow(masks[0], cmap='jet', vmin=0, vmax=config.NUM_CLASSES-1) 
    ax[1].set_title("Ground Truth Mask")
    plt.show()
    
    print("Generator test successful!")

