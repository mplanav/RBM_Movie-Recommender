"""
preprocess.py
--------------
Utilidades de preprocesamiento para MovieLens 100K orientadas a un RBM en TensorFlow.

Qué hace y por qué (resumen):
- Carga las calificaciones de MovieLens 100K (u.data).
- Construye la matriz usuario×película.
- Opción de binarizar ratings (Bernoulli visible units): rating >= threshold -> 1, rating < threshold -> 0.
  * Esto es lo más estándar para un RBM Bernoulli-Bernoulli.
- Mantiene una máscara de observaciones para **ignorar celdas no vistas** en la función de pérdida.
- Expone un tf.data.Dataset listo para batching y entrenamiento.

Salidas típicas:
- X: matriz (users, items) en {0,1} con 0 para "no le gusta" y 0 también para faltantes; se usan con `mask` para distinguir.
- mask: matriz (users, items) en {0,1} donde 1 = celda observada, 0 = faltante.
- Mappings: diccionarios para ir de índices internos a IDs de MovieLens.

Requisitos: pandas, numpy, tensorflow
"""
from __future__ import annotations

import os
import json
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import tensorflow as tf


def load_movielens_100k(path: str) -> pd.DataFrame:
    """
    Carga el fichero 'u.data' de MovieLens 100K.

    Parámetros
    ----------
    path : str
        Ruta al fichero u.data (por ejemplo: 'data/ml-100k/u.data').

    Returns
    -------
    pd.DataFrame con columnas: user_id, movie_id, rating, timestamp
    """
    columns = ["user_id", "movie_id", "rating", "timestamp"]
    df = pd.read_csv(path, sep="\t", names=columns, engine="python")
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    df["rating"] = df["rating"].astype(int)
    return df


def build_user_item_matrix(
    ratings_df: pd.DataFrame,
    binarize: bool = True,
    threshold: int = 3,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, dict]]:
    """
    Construye la matriz usuario×película y la máscara de observaciones.

    - Si `binarize=True`, convierte rating >= threshold -> 1, rating < threshold -> 0.
      Esto permite usar unidades visibles Bernoulli en el RBM.
    - Las celdas no observadas NO pueden distinguirse del 0 en la matriz X; por eso devolvemos `mask`.

    Returns
    -------
    X : np.ndarray (num_users, num_items) float32 en {0,1}
    mask : np.ndarray (num_users, num_items) float32 en {0,1}
    mappings : dict con:
        - user_id_to_index / index_to_user_id
        - movie_id_to_index / index_to_movie_id
    """
    user_ids = np.sort(ratings_df["user_id"].unique())
    movie_ids = np.sort(ratings_df["movie_id"].unique())

    user_id_to_index = {uid: i for i, uid in enumerate(user_ids)}
    movie_id_to_index = {mid: j for j, mid in enumerate(movie_ids)}
    index_to_user_id = {i: int(uid) for i, uid in enumerate(user_ids)}
    index_to_movie_id = {j: int(mid) for j, mid in enumerate(movie_ids)}

    num_users = len(user_ids)
    num_items = len(movie_ids)

    X = np.zeros((num_users, num_items), dtype=np.float32)
    mask = np.zeros((num_users, num_items), dtype=np.float32)

    if binarize:
        values = (ratings_df["rating"].values >= threshold).astype(np.float32)
    else:
        # Escalado simple a [0,1] desde [1,5]
        values = ((ratings_df["rating"].values - 1.0) / 4.0).astype(np.float32)

    ui = ratings_df["user_id"].map(user_id_to_index).values
    mj = ratings_df["movie_id"].map(movie_id_to_index).values

    X[ui, mj] = values
    mask[ui, mj] = 1.0

    mappings = {
        "user_id_to_index": user_id_to_index,
        "movie_id_to_index": movie_id_to_index,
        "index_to_user_id": index_to_user_id,
        "index_to_movie_id": index_to_movie_id,
    }
    return X, mask, mappings


def train_val_split_by_user(
    X: np.ndarray,
    mask: np.ndarray,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Split por usuarios completos para evitar fuga de información.
    Devuelve (X_train, mask_train), (X_val, mask_val)
    """
    rng = np.random.default_rng(seed)
    num_users = X.shape[0]
    idx = np.arange(num_users)
    rng.shuffle(idx)

    cut = int((1.0 - val_ratio) * num_users)
    train_idx, val_idx = idx[:cut], idx[cut:]

    return (X[train_idx], mask[train_idx]), (X[val_idx], mask[val_idx])


def to_tf_dataset(
    X: np.ndarray,
    mask: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Convierte (X, mask) a un tf.data.Dataset de pares (input, mask).

    El RBM consumirá `input` y usará `mask` para computar la pérdida solo sobre celdas observadas.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, mask))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def save_artifacts(
    out_dir: str,
    X: np.ndarray,
    mask: np.ndarray,
    mappings: Dict[str, dict],
) -> None:
    """
    Guarda matrices y diccionarios para reutilizarlos en entrenamiento/evaluación.
    """
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "mask.npy"), mask)
    with open(os.path.join(out_dir, "mappings.json"), "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False)


def quick_prepare_pipeline(
    data_root: str = "data/ml-100k",
    binarize: bool = True,
    threshold: int = 3,
    val_ratio: float = 0.1,
    batch_size: int = 64,
    seed: int = 42,
):
    """
    Pipeline "rápido" para:
      1) Cargar u.data
      2) Construir matriz + máscara
      3) Split train/val y devolver datasets listos para el entrenamiento del RBM.

    Returns
    -------
    ds_train, ds_val, meta
      - meta: dict con num_users, num_items, threshold, binarize, etc.
    """
    ratings_path = os.path.join(data_root, "u.data")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(
            f"No se encontró {ratings_path}. Descarga MovieLens 100K y colócalo en {data_root}."
        )

    df = load_movielens_100k(ratings_path)
    X, mask, mappings = build_user_item_matrix(df, binarize=binarize, threshold=threshold)
    (Xtr, Mtr), (Xva, Mva) = train_val_split_by_user(X, mask, val_ratio=val_ratio, seed=seed)

    ds_train = to_tf_dataset(Xtr, Mtr, batch_size=batch_size, shuffle=True, seed=seed)
    ds_val = to_tf_dataset(Xva, Mva, batch_size=batch_size, shuffle=False, seed=seed)

    meta = {
        "num_users": int(X.shape[0]),
        "num_items": int(X.shape[1]),
        "binarize": binarize,
        "threshold": threshold,
        "val_ratio": val_ratio,
        "batch_size": batch_size,
    }
    return ds_train, ds_val, meta


if __name__ == "__main__":
    # Ejemplo rápido de uso desde línea de comandos:
    # python src/preprocess.py
    data_root = "data/ml-100k"
    print("[preprocess] Cargando y preparando dataset...")
    ds_train, ds_val, meta = quick_prepare_pipeline(data_root=data_root)
    print("[preprocess] Hecho. Meta:", meta)
