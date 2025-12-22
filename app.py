import gradio as gr
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import os

# --- Config ---
# Path to your trained model (The 500 step one)
CHECKPOINT_DIR = "./controlnet_output" 
DEVICE = "mps" 

# Enable Mac Fallback for the math operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

print("🚀 Loading Models... (This might take a minute)")

# 1. Load ControlNet Pipeline
controlnet = ControlNetModel.from_pretrained(CHECKPOINT_DIR, torch_dtype=torch.float32)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=torch.float32,
    safety_checker=None
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(DEVICE)
pipe.enable_attention_slicing()

# 2. Load Namer Pipeline (BLIP + Flan-T5)
vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", use_safetensors=True).to(DEVICE)
text_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
text_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", use_safetensors=True).to(DEVICE)

print("✅ Models Ready!")

def process_sketch(sketch):
    """
    Takes the raw sketch from the UI and turns it into a solid silhouette.
    """
    if sketch is None:
        return None

    # Gradio returns a dictionary with 'composite' (the actual image)
    # We convert it to a numpy array (OpenCV format)
    image = sketch["composite"]
    
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    
    # 2. Threshold to get a strict Black/White binary image
    # (Any pixel that isn't white becomes black)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # 3. "The Magic Fill" - Find contours and fill them
    # This turns a line drawing of a circle into a filled black circle
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_silhouette = np.zeros_like(binary)
    cv2.drawContours(filled_silhouette, contours, -1, (255), thickness=cv2.FILLED)
    
    # If the user just drew a solid blob, keep it. If lines, use the filled version.
    # We combine them just to be safe.
    final_mask = cv2.bitwise_or(binary, filled_silhouette)
    
    # Convert to RGB for ControlNet (White Background, Black Shape)
    # The mask is currently White Shape on Black Background.
    # We invert it to match your training data (Black shape on White).
    final_image = cv2.bitwise_not(final_mask)
    final_image = cv2.cvtColor(final_image, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(final_image)

def generate_pokemon(sketch_dict, prompt_text):
    # 1. Prepare Silhouette
    control_image = process_sketch(sketch_dict)
    control_image = control_image.resize((512, 512))

    # 2. Run Diffusion
    # We use a generic prompt to let the model be creative, but respect the user's text if added
    full_prompt = f"official art of a pokemon, ken sugimori style, white background, {prompt_text}"
    
    generated_image = pipe(
        full_prompt, 
        image=control_image, 
        num_inference_steps=20,
        controlnet_conditioning_scale=1.0, # Adjust to 1.5 if you want it strict
        negative_prompt="low quality, bad anatomy, text, watermark"
    ).images[0]

    # 3. Name It
    # Vision
    inputs = vision_processor(generated_image, return_tensors="pt").to(DEVICE)
    out = vision_model.generate(**inputs, max_new_tokens=50)
    caption = vision_processor.decode(out[0], skip_special_tokens=True)
    
    # Text
    input_text = f"Invent a fantasy RPG name for a creature that looks like this: {caption}. Name:"
    input_ids = text_tokenizer(input_text, return_tensors="pt").input_ids.to(DEVICE)
    outputs = text_model.generate(input_ids, max_length=15, do_sample=True, temperature=1.1)
    name = text_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return control_image, generated_image, f"✨ {name} ✨"

# --- The UI Layout ---
with gr.Blocks(title="Who's That Diffusion?") as demo:
    gr.Markdown("# 🎨 Draw-a-Pokémon")
    gr.Markdown("Draw a black outline in the box below. The AI will fill it in, generate a creature, and name it!")
    
    with gr.Row():
        with gr.Column():
            # The Sketchpad
            sketchpad = gr.Sketchpad(
                label="Draw Here (Close your loops!)", 
                type="numpy", 
                brush=gr.Brush(colors=["#000000"], default_size=4) # Black brush
            )
            prompt = gr.Textbox(label="Optional details (e.g., 'fire type')", placeholder="fire type, angry eyes")
            run_btn = gr.Button("Generate!", variant="primary")
        
        with gr.Column():
            # Outputs
            result_img = gr.Image(label="Your Pokémon")
            pokemon_name = gr.Label(label="Name")
            debug_view = gr.Image(label="Debug: How the AI saw your drawing", height=200)

    run_btn.click(
        fn=generate_pokemon, 
        inputs=[sketchpad, prompt], 
        outputs=[debug_view, result_img, pokemon_name]
    )

if __name__ == "__main__":
    demo.launch(share=True) # share=True creates a public link you can send to friends!