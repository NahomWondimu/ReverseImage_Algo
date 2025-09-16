from typing import Union
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from PIL import Image
import torch
import io
import requests

from pymongo import MongoClient

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
)


# --- fastapi configs ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connecting to MongoDB
"""
client = MongoClient('mongodb://localhost:27012/')
db = client['HWH_db']
collection = db['clothing_catalog']
"""

# --- Loading models ---
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Helpers

def gen_caption(image: Image.Image) -> str:
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image")

    inputs = blip_processor(image, return_tensors="pt")
    output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption

def gen_embedding(image: Image.Image, desc: str) -> torch.Tensor:
    
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image")

    if not isinstance(desc, str):
        raise TypeError("Expected a string caption")

    inputs = clip_processor(text=desc, images=image, return_tensors="pt", padding=True, do_convert_rgb=False)
    with torch.no_grad():
        outputs = clip_model(**inputs)

    image_embed = outputs.image_embeds
    text_embed = outputs.text_embeds
    combined = (image_embed + text_embed) / 2

    return combined.squeeze().tolist()

# --
@app.get("/")
def read_root(image_url: str = Query(..., description="URL of the image to process")):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to load image: {str(e)}"})

    try:
        caption = gen_caption(image)
        image_embedding = gen_embedding(image, caption)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Processing error: {str(e)}"})

    return JSONResponse({
        "description": caption,
        "embedding_vector": image_embedding
    })


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/embed-image/")
async def embed_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    caption = gen_caption(image)
    image_embedding = gen_embedding(image, caption)

    return JSONResponse({
        "description": caption,
        "embedding_vector": image_embedding
    })