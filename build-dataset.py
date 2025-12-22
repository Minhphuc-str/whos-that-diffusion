import os
import requests
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import time

# --- Configuration ---
# We want Gen 1 (001 to 151)
POKEMON_IDS = range(1, 152) 
DATASET_DIR = "dataset"
IMG_SIZE = 256  # Pix2Pix standard

# User-Agent to avoid being blocked by the CDN
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def create_directory_structure():
    os.makedirs(os.path.join(DATASET_DIR, "train", "A"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "train", "B"), exist_ok=True)

def download_official_art(pokedex_id, save_path):
    """Downloads directly from Pokemon.com assets."""
    # Format ID to 3 digits (e.g., 1 -> '001')
    url = f"https://assets.pokemon.com/assets/cms2/img/pokedex/full/{pokedex_id:03d}.png"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"  [!] Failed {url} - Status: {response.status_code}")
    except Exception as e:
        print(f"  [!] Error downloading ID {pokedex_id}: {e}")
    return False

def elastic_transform(image, alpha, sigma, random_state=None):
    """
    Distorts the image to simulate wobbly hand-drawn lines.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def process_image(img_path, output_name):
    # Read image including alpha channel
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False

    # Resize maintaining aspect ratio
    h, w = img.shape[:2]
    scale = IMG_SIZE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Canvas
    canvas = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
    
    # Handle Transparency (Official assets are always PNG with Alpha)
    if img.shape[2] == 4:
        b, g, r, alpha = cv2.split(img)
        img_rgb = cv2.merge([b, g, r])
        mask = alpha
    else:
        # Fallback for JPG (unlikely with this script)
        img_rgb = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Centering math
    y_offset = (IMG_SIZE - new_h) // 2
    x_offset = (IMG_SIZE - new_w) // 2
    
    # --- Create Ground Truth (Target B) ---
    target_img = canvas.copy()
    roi = target_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
    
    # Alpha Blending
    alpha_factor = mask[:, :, np.newaxis] / 255.0
    roi[:] = (1.0 - alpha_factor) * roi + alpha_factor * img_rgb
    target_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi

    # --- Create Noisy Silhouette (Input A) ---
    # 1. Place mask on canvas
    bin_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    bin_mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = mask

    # 2. Elastic Deformation (Wobbly effect)
    noisy_mask = elastic_transform(bin_mask, alpha=IMG_SIZE*0.05, sigma=IMG_SIZE*0.02)

    # 3. Morphological noise (varying thickness)
    kernel = np.ones((3,3), np.uint8)
    if np.random.rand() > 0.5:
        noisy_mask = cv2.dilate(noisy_mask, kernel, iterations=np.random.randint(1, 3))
    else:
        noisy_mask = cv2.erode(noisy_mask, kernel, iterations=np.random.randint(1, 2))

    # 4. Threshold and Invert
    _, noisy_mask = cv2.threshold(noisy_mask, 127, 255, cv2.THRESH_BINARY)
    input_img = cv2.bitwise_not(noisy_mask)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)

    # Save
    cv2.imwrite(os.path.join(DATASET_DIR, "train", "A", output_name), input_img)
    cv2.imwrite(os.path.join(DATASET_DIR, "train", "B", output_name), target_img)
    return True

def main():
    create_directory_structure()
    print(f"Starting direct download for {len(POKEMON_IDS)} Pokemon...")
    
    for pid in POKEMON_IDS:
        temp_name = f"temp_{pid}.png"
        final_name = f"{pid}.png"
        
        print(f"Processing ID: {pid:03d}...", end="\r")
        
        if download_official_art(pid, temp_name):
            process_image(temp_name, final_name)
            if os.path.exists(temp_name):
                os.remove(temp_name)
        
        # Be nice to the server
        time.sleep(0.5)
            
    print("\nDone! Check the 'dataset/train' folder.")

if __name__ == "__main__":
    main()