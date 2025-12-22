import os
import json
import shutil

# Config
SOURCE_DIR = "dataset/train"
OUTPUT_DIR = "dataset_controlnet"

def build_dataset():
    # Create structure expected by HuggingFace datasets
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "conditioning_images"), exist_ok=True)
    
    # Check if source exists
    if not os.path.exists(os.path.join(SOURCE_DIR, "A")):
        print(f"Error: Could not find {SOURCE_DIR}/A. Are you in the right folder?")
        return

    files = sorted(os.listdir(os.path.join(SOURCE_DIR, "A")))
    files = [f for f in files if f.endswith(('.png', '.jpg'))]
    
    metadata = []
    
    print(f"Formatting {len(files)} pairs for ControlNet...")
    
    for filename in files:
        # Source Paths
        src_silhouette = os.path.join(SOURCE_DIR, "A", filename)
        src_color = os.path.join(SOURCE_DIR, "B", filename)
        
        # Check if pair exists
        if not os.path.exists(src_color):
            continue

        # Destination Paths
        dst_silhouette = os.path.join(OUTPUT_DIR, "conditioning_images", filename)
        dst_color = os.path.join(OUTPUT_DIR, "images", filename)
        
        # Copy files
        shutil.copy(src_silhouette, dst_silhouette)
        shutil.copy(src_color, dst_color)
        
        # Create Metadata Entry with ABSOLUTE PATHS
        # This fixes the FileNotFoundError by telling the training script exactly where to look
        entry = {
            "text": "official art of a pokemon, ken sugimori style, white background",
            "image": os.path.abspath(dst_color),
            "conditioning_image": os.path.abspath(dst_silhouette)
        }
        metadata.append(entry)
        
    # Write JSONL file
    with open(os.path.join(OUTPUT_DIR, "train.jsonl"), 'w') as f:
        for entry in metadata:
            json.dump(entry, f)
            f.write('\n')
            
    print(f"Done! Dataset ready at '{OUTPUT_DIR}' with absolute paths.")

if __name__ == "__main__":
    build_dataset()