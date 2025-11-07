"""
recommend.py
-------------
Genera recomendaciones personalizadas usando el RBM entrenado.

Funciones principales:
- recommend_for_user(): genera top-N recomendaciones para un usuario concreto.
- Carga los mappings y la matriz original para saber qué películas ya vio el usuario.
"""

import json
import numpy as np
import tensorflow as tf
from rbm_model_tf import RBM


def load_mappings(path: str):
    """Carga los diccionarios de índices/IDs desde mappings.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recommend_for_user(
    rbm: RBM,
    X: np.ndarray,
    mappings: dict,
    user_id: int,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """
    Genera recomendaciones para un usuario específico.

    Parámetros:
    -----------
    rbm : modelo RBM entrenado
    X : matriz de usuarios × películas
    mappings : diccionario con índices/IDs (de preprocess)
    user_id : ID original del usuario (no índice interno)
    top_k : número de películas a recomendar

    Devuelve:
    ---------
    Lista de (movie_id, score) ordenada de mayor a menor.
    """
    user_idx = mappings["user_id_to_index"].get(str(user_id)) or mappings["user_id_to_index"].get(user_id)
    if user_idx is None:
        raise ValueError(f"Usuario {user_id} no encontrado en mappings.")

    # Obtenemos las valoraciones del usuario
    user_vector = X[user_idx].reshape(1, -1)

    # Reconstruimos con el RBM (probabilidades predichas)
    user_vector_tf = tf.constant(user_vector, dtype=tf.float32)
    reconstructed = rbm.reconstruct(user_vector_tf).numpy().flatten()

    # Películas ya vistas → las ponemos a -inf para no recomendarlas
    seen_mask = user_vector.flatten() > 0
    reconstructed[seen_mask] = -np.inf

    # Top-N índices
    top_indices = np.argsort(reconstructed)[::-1][:top_k]

    # Convertir a movie_ids reales
    index_to_movie_id = mappings["index_to_movie_id"]
    recommendations = [(int(index_to_movie_id[str(i)]), float(reconstructed[i])) for i in top_indices]

    return recommendations


if __name__ == "__main__":
    # Ejemplo rápido: simular flujo de recomendación después de entrenar
    print("[recommend] Cargando modelo y datos...")
    rbm = RBM(n_visible=1682, n_hidden=128)
    rbm.build((None, 1682))
    rbm.load_weights("models/rbm_weights.weights.h5")  # si guardaste los pesos al entrenar

    X = np.load("data/X.npy")
    mappings = load_mappings("data/mappings.json")

    user_id = 25
    recs = recommend_for_user(rbm, X, mappings, user_id=user_id, top_k=5)

    print(f"\nTop 5 recomendaciones para el usuario {user_id}:")
    for movie_id, score in recs:
        print(f" - Película ID {movie_id}: score {score:.4f}")
