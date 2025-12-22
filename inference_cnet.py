import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import argparse
import os

# --- Config ---
# We load the checkpoint that will be saved at the end of your 500 steps
CHECKPOINT_DIR = "./controlnet_output"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
DEVICE = "mps" # Mac Silicon

def process_image(image_path):
    """
    ControlNet needs the input to be:
    1. A PIL Image
    2. Dimension divisible by 8 (e.g., 512x512)
    """
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    # Convert to grayscale to get the raw shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to ensure binary black/white
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Invert if necessary (ControlNet usually expects white lines on black, 
    # but our training data was Black Shape on White. Let's keep it consistent 
    # with how you trained it: Black Shape on White Background.)
    
    # Convert back to RGB 3-channel
    image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    
    # Resize to 512x512 (Standard SD v1.5 size)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    
    return Image.fromarray(image)

def generate(input_path, output_path):
    print(f"Loading ControlNet from {CHECKPOINT_DIR}...")
    
    # 1. Load your trained adapter
    controlnet = ControlNetModel.from_pretrained(
        CHECKPOINT_DIR, 
        torch_dtype=torch.float32 # Use float32 for Mac MPS
    )

    # 2. Load the base Stable Diffusion model with your adapter attached
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL, 
        controlnet=controlnet, 
        torch_dtype=torch.float32,
        safety_checker=None # Disable safety checker to save memory/speed
    )

    # 3. Optimize for Mac
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(DEVICE)
    # Enable memory slicing to avoid "Out of Memory" crashes on Mac
    pipe.enable_attention_slicing() 

    # 4. Prepare Input
    control_image = process_image(input_path)
    
    # 5. The Magic Prompt
    # We use the EXACT same prompt format as training
    prompt = "official art of a pokemon, ken sugimori style, white background"
    negative_prompt = "low quality, bad anatomy, worst quality, text, watermark, glitch, deformed, ugly"

    print("Generating...")
    result = pipe(
        prompt, 
        image=control_image, 
        negative_prompt=negative_prompt,
        num_inference_steps=20, # 20 steps is standard for UniPC scheduler
        guidance_scale=7.5      # How strictly to follow the text prompt
    ).images[0]

    # 6. Save
    result.save(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to silhouette")
    parser.add_argument("--output", type=str, default="cnet_result.png")
    args = parser.parse_args()
    
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Wait! Training isn't done yet. Could not find: {CHECKPOINT_DIR}")
    else:
        generate(args.input, args.output)