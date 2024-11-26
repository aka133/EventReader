from fastapi import FastAPI, HTTPException
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
import uvicorn
from PIL import Image
import io
import base64
import numpy as np
import traceback

app = FastAPI()

# Add near the top, after app = FastAPI()
@app.get("/")
async def root():
    return {"status": "running", "model": "llava-1.5-7b-hf", "device": str(device)}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "memory_allocated": f"{torch.cuda.memory_allocated()/1e9:.2f}GB"}

# Initialize model and processor
print("Loading model and processor...")
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Configure processor correctly for LLaVA
processor.image_processor.size = {"height": 336, "width": 336}
processor.image_processor.is_vqa = True
processor.tokenizer.padding_side = "right"
processor.tokenizer.truncation_side = "left"
model.config.use_cache = False

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

def process_base64_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

@app.post("/analyze")
async def analyze_frame(data: dict):
    try:
        print(f"[DEBUG] Received request with prompt length: {len(data.get('prompt', ''))}")
        print(f"[DEBUG] Number of images: {len(data.get('images', []))}")
        
        if 'images' not in data or not data['images']:
            raise HTTPException(status_code=400, detail="No images provided")
        if 'prompt' not in data:
            raise HTTPException(status_code=400, detail="No prompt provided")
            
        try:
            # Process single image
            image = process_base64_image(data['images'][0])
            if image.mode != 'RGB':
                image = image.convert('RGB')
            print(f"[DEBUG] Processed image size: {image.size}, mode: {image.mode}")
            
            # Process with special token handling for LLaVA
            prompt = data['prompt']
            if not prompt.startswith('<image>'):
                prompt = '<image>\n' + prompt
            
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                add_special_tokens=True
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            print("[DEBUG] Input keys:", inputs.keys())
            print("[DEBUG] Input shapes:", {k: v.shape for k, v in inputs.items()})
            print("[DEBUG] First few tokens:", 
                  processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][:10]))
            
        except Exception as e:
            print(f"[ERROR] Processing failed: {str(e)}")
            print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=20000,
                    num_beams=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True  # Enable for generation
                )
            print("[DEBUG] Successfully generated output")
            
        except Exception as e:
            print(f"[ERROR] Generation failed: {str(e)}")
            print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
        
        response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        ssl_keyfile=None,  # Remove SSL for now
        ssl_certfile=None
    )