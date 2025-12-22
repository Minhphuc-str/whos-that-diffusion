import torch
import cv2
import numpy as np
import argparse
import os
from networks import UNetGenerator
import torchvision.utils as vutils

# --- Configuration ---
MODEL_PATH = "checkpoints/netG_final.pth" # Or use a specific epoch like "checkpoints/netG_195.pth"
OUTPUT_DIR = "inference_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def preprocess_image(image_path):
    """
    Loads user image, resizes to 256x256, and normalizes to [-1, 1].
    """
    # Read image as grayscale to get the shape
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Resize to 256x256 (Model expects this exact size)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    
    # Threshold to ensure binary black/white (removes anti-aliasing artifacts)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Convert back to RGB (3 channels) because the model expects 3 channels
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Convert to standard Model formatting
    # HWC to CHW, scale 0-255 to -1 to 1
    img = img.astype(np.float32) / 127.5 - 1.0
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0) # Add batch dim
    
    return img

def run_inference(image_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    netG = UNetGenerator(input_nc=3, output_nc=3).to(DEVICE)
    
    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    netG.load_state_dict(checkpoint)
    netG.eval() # Set to evaluation mode (turns off Dropout/BatchNorm updates)
    
    # 2. Process Input
    print(f"Processing {image_path}...")
    input_tensor = preprocess_image(image_path).to(DEVICE)
    
    # 3. Generate
    with torch.no_grad():
        output_tensor = netG(input_tensor)
        
    # 4. Save Result
    # Denormalize: [-1, 1] -> [0, 1]
    input_display = (input_tensor + 1) / 2.0
    output_display = (output_tensor + 1) / 2.0
    
    # Stack them side-by-side
    result = torch.cat([input_display, output_display], dim=3)
    
    filename = os.path.basename(image_path)
    save_path = os.path.join(OUTPUT_DIR, f"gen_{filename}")
    vutils.save_image(result, save_path)
    
    print(f"Done! Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to your silhouette drawing")
    args = parser.parse_args()
    
    run_inference(args.input)