import mlflow
import sys
from sklearn.datasets import fetch_20newsgroups

def sanitize_name(name):
    return name.replace(".", "_").replace("/", "_").replace(" ", "_")

if len(sys.argv) < 2:
    print("Uso: python mlflow_predict.py \"Texto a clasificar\"")
    sys.exit(1)

texto_usuario = " ".join(sys.argv[1:])

# Define las categorías
categoria1 = "sci.space"
categoria2 = "comp.graphics"
categorias = [categoria1, categoria2]

# Obtener el orden correcto de las clases
target_names = fetch_20newsgroups(subset="train", categories=categorias).target_names

# Nombre del modelo
model_name = f"TextoClasificador_{categoria1}_vs_{categoria2}"
model_name = sanitize_name(model_name)
model_uri = f"models:/{model_name}/latest"

# Cargar modelo
try:
    model = mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    print(f"Error cargando el modelo '{model_name}': {e}")
    sys.exit(1)

# Predecir
try:
    prediccion = model.predict([texto_usuario])
    clase_predicha = target_names[prediccion[0]]

    print(f"Texto: {texto_usuario}")
    print(f"Predicción (label numérico): {prediccion[0]}")
    print(f"Categoría predicha: {clase_predicha}")
except Exception as e:
    print(f"Error haciendo la predicción: {e}")
