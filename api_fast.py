from fastapi import FastAPI
from transformers import pipeline
import torch
from diffusers import StableDiffusionPipeline
import base64
from io import BytesIO

app = FastAPI(title="AI API")

# Hugging Face pipelines
sentiment_analyzer = pipeline("sentiment-analysis")
translator = pipeline("translation_en_to_fr")

# Image generation pipeline (Stable Diffusion)
dreamlike_pipe = StableDiffusionPipeline.from_pretrained(
    "dreamlike-art/dreamlike-anime-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")  # Make sure to use GPU if available


@app.get("/")
def read_root():
    return {"message": "Welcome to the AI API!"}


@app.get("/greet")
def greet(name: str = "user"):
    return {"message": f"Hello, {name}!"}


@app.get("/double")
def double(number: int):
    return {"result": number * 2}


@app.get("/sentiment_analysis")
def sentiment_analysis(text: str):
    result = sentiment_analyzer(text)
    return {"sentiment": result}


@app.get("/translate")
def translate(text: str):
    result = translator(text)
    return {"translation": result[0]["translation_text"]}


@app.get("/generate_image")
def generate_image(prompt: str = "an anime girl in a magical forest"):
    image = dreamlike_pipe(prompt).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"prompt": prompt, "image_base64": img_base64}
