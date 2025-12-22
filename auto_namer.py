import argparse
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Config ---
# 1. The "Eyes": Captioning Model (Small & Fast)
VISION_MODEL_ID = "Salesforce/blip-image-captioning-base"

# 2. The "Brain": Naming Model (Instruction Tuned)
TEXT_MODEL_ID = "google/flan-t5-base"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def load_models():
    print("Loading Vision Model (BLIP)...")
    processor = BlipProcessor.from_pretrained(VISION_MODEL_ID)
    vision_model = BlipForConditionalGeneration.from_pretrained(
        VISION_MODEL_ID, 
        use_safetensors=True
    ).to(DEVICE)

    print("Loading Language Model (Flan-T5)...")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    text_model = AutoModelForSeq2SeqLM.from_pretrained(
        TEXT_MODEL_ID, 
        use_safetensors=True
    ).to(DEVICE)
    
    return processor, vision_model, tokenizer, text_model

def generate_name(image_path):
    processor, vision_model, tokenizer, text_model = load_models()
    
    # 1. Load Image
    try:
        raw_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 2. Vision Step: Unconditional (No Hints!)
    print("Analyzing image...")
    
    # We remove the text prompt so it describes the VISUALS, not the category.
    inputs = processor(raw_image, return_tensors="pt").to(DEVICE)

    out = vision_model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"👀 AI sees: '{caption}'")

    # 3. Language Step: Invent Name
    print("Inventing name...")
    
    # We ask for a "Fantasy RPG" name to avoid getting "Frog" or "Dog"
    input_text = f"Invent a unique fantasy RPG monster name for a creature that looks like this: {caption}. Name:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(DEVICE)

    outputs = text_model.generate(
        input_ids, 
        max_length=15, 
        do_sample=True, 
        temperature=1.1, # High creativity
        top_k=50
    )
    pokemon_name = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("-" * 30)
    print(f"✨ Name: {pokemon_name}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the generated image")
    args = parser.parse_args()
    
    # Enable fallback for MPS math operations just in case
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    generate_name(args.image)