from fastapi import FastAPI
from transformers import pipeline, AutoTokenizer, AutoModel
from diffusers import StableDiffusionPipeline
import torch
import base64
from io import BytesIO
import mlflow
from sklearn.datasets import fetch_20newsgroups

app = FastAPI(title="API")

# === MODELOS DE TRANSFORMERS ===
sentiment_analyzer = pipeline("sentiment-analysis")
translator = pipeline("translation_en_to_fr")

# === MODELO DE EMBEDDINGS ===
tokenizer_embed = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model_embed = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

# === MODELO DE IMAGEN (Stable Diffusion) ===
dreamlike_pipe = StableDiffusionPipeline.from_pretrained(
    "dreamlike-art/dreamlike-anime-1.0", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")


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


@app.get("/embed_text")
def embed_text(text: str):
    inputs = tokenizer_embed(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model_embed(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
    return {"text": text, "embedding": embeddings[0].tolist()}


# === FUNCIONES MLflow ===


def sanitize_name(name):
    return name.replace(".", "_").replace("/", "_").replace(" ", "_")


@app.get("/predict_text_category")
def predict_text_category(text: str = "NASA launched a new satellite this month"):
    # Define categorías
    categoria1 = "sci.space"
    categoria2 = "comp.graphics"
    categorias = [categoria1, categoria2]

    # Obtener los nombres de clases
    target_names = fetch_20newsgroups(
        subset="train", categories=categorias
    ).target_names

    # Modelo en MLflow
    model_name = f"TextoClasificador_{categoria1}_vs_{categoria2}"
    model_name = sanitize_name(model_name)
    model_uri = f"models:/{model_name}/latest"

    try:
        model = mlflow.pyfunc.load_model(model_uri)
        prediccion = model.predict([text])
        clase_predicha = target_names[prediccion[0]]

        return {
            "text": text,
            "predicted_label": int(prediccion[0]),
            "predicted_category": clase_predicha,
        }

    except Exception as e:
        return {"error": str(e)}
