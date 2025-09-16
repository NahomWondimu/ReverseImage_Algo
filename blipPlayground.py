from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import gradio as gr

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def gen_caption(image) -> str:
    img_input = Image.fromarray(image)
    inputs = processor(img_input, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

demo = gr.Interface(fn=gen_caption, inputs=[gr.Image(label="Image")], outputs=[gr.Text(label="Caption"),],)

demo.launch()