import argparse
import mlflow
import mlflow.sklearn

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }


def create_pipeline(
    max_iter=100,
    C=1.0,
    max_features=None,
    ngram_max=1,
):
    pipeline = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, ngram_max),
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter,
                    C=C,
                ),
            ),
        ]
    )
    return pipeline


def sanitize_name(name):
    return name.replace(".", "_").replace("/", "_").replace(" ", "_")


def main(
    categoria1,
    categoria2,
    max_iter,
    C,
    max_features,
    ngram_max,
):
    data = fetch_20newsgroups(
        subset="all",
        categories=[categoria1, categoria2],
        remove=("headers", "footers", "quotes"),
    )
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    pipeline = create_pipeline(
        max_iter=max_iter,
        C=C,
        max_features=max_features,
        ngram_max=ngram_max,
    )

    mlflow.set_experiment("text_classification")
    with mlflow.start_run(run_name=f"{categoria1}_vs_{categoria2}"):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metricas = calculate_metrics(y_test, y_pred)
        for clave, valor in metricas.items():
            mlflow.log_metric(clave, valor)

        # Loguear hiperparámetros
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("C", C)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("ngram_max", ngram_max)

        mlflow.log_param("modelo", f"LogisticRegression_{categoria1}_vs_{categoria2}")
        mlflow.log_param("vectorizer", f"TfidfVectorizer_{categoria1}_vs_{categoria2}")

        modelo_name = f"modelo_texto_{categoria1}_vs_{categoria2}"
        modelo_name = sanitize_name(modelo_name)

        registered_name = f"TextoClasificador_{categoria1}_vs_{categoria2}"
        registered_name = sanitize_name(registered_name)

        mlflow.sklearn.log_model(
            pipeline, modelo_name, registered_model_name=registered_name
        )

        print(f"\nResultado entre '{categoria1}' y '{categoria2}':")
        for k, v in metricas.items():
            print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categoria1",
        required=True,
        help="Nombre de la primera categoría (ej: 'sci.space')",
    )
    parser.add_argument(
        "--categoria2",
        required=True,
        help="Nombre de la segunda categoría (ej: 'comp.graphics')",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Número máximo de iteraciones para LogisticRegression",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Parámetro de regularización para LogisticRegression",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=None,
        help="Máximo número de características para TfidfVectorizer",
    )
    parser.add_argument(
        "--ngram_max", type=int, default=1, help="Máximo ngram para TfidfVectorizer"
    )

    args = parser.parse_args()

    main(
        args.categoria1,
        args.categoria2,
        args.max_iter,
        args.C,
        args.max_features,
        args.ngram_max,
    )
