"""
train.py
--------
Script de entrenamiento para el RBM en TensorFlow usando MovieLens 100K.

Flujo:
1. Cargar y preprocesar datos (quick_prepare_pipeline).
2. Crear el modelo RBM.
3. Entrenar usando Contrastive Divergence (CD-k).
4. Evaluar periódicamente en un conjunto de validación.

Este archivo se centra en:
- Cómo se usa el RBM.
- Cómo encajan X, mask y el entrenamiento no supervisado.
"""

"""
train.py
--------
Script de entrenamiento para el RBM en TensorFlow usando MovieLens 100K.

Ahora incluye:
✅ Guardado automático de los pesos del modelo.
✅ Carga opcional desde pesos existentes (para continuar entrenamiento o recomendar).
"""

import os
import tensorflow as tf

from preprocess import quick_prepare_pipeline
from rbm_model_tf import RBM



def compute_reconstruction_loss(rbm: RBM, dataset: tf.data.Dataset) -> tf.Tensor:
    """
    Calcula la pérdida de reconstrucción (MSE con máscara) en un dataset
    SIN actualizar los pesos (modo evaluación).
    """
    all_losses = []

    for batch_v, batch_mask in dataset:
        batch_v = tf.cast(batch_v, tf.float32)
        batch_mask = tf.cast(batch_mask, tf.float32)

        # Paso hacia arriba y hacia abajo (sin muestreo duro)
        p_h = rbm.propup(batch_v)
        p_v_recon = rbm.propdown(p_h)

        # Solo evaluamos donde mask == 1 (ratings conocidos)
        v_true = batch_v * batch_mask
        v_pred = p_v_recon * batch_mask

        loss = tf.reduce_mean(tf.square(v_true - v_pred))
        all_losses.append(loss)

    if not all_losses:
        return tf.constant(0.0, dtype=tf.float32)

    return tf.reduce_mean(all_losses)


def train_rbm(
    data_root: str = "data/ml-100k",
    n_hidden: int = 128,
    batch_size: int = 64,
    lr: float = 0.01,
    k: int = 1,
    binarize: bool = True,
    threshold: int = 3,
    val_ratio: float = 0.1,
    epochs: int = 30,
    save_dir: str = "models",
    resume: bool = False,
):
    """
    Entrena un RBM con los parámetros dados.

    Parámetros clave:
    - n_hidden: nº de unidades ocultas.
    - lr: learning rate de Contrastive Divergence.
    - k: pasos de Gibbs (CD-k). Normalmente k=1 (CD-1).
    - binarize: si True, ratings >= threshold se convierten en 1 (me gusta).
    - save_dir: carpeta donde guardar los pesos.
    - resume: si True, intenta cargar pesos previos de save_dir.
    """

    print("[train] Preparando datos...")
    ds_train, ds_val, meta = quick_prepare_pipeline(
        data_root=data_root,
        binarize=binarize,
        threshold=threshold,
        val_ratio=val_ratio,
        batch_size=batch_size,
    )

    import numpy as np
    import json
    np.save("data/X.npy", meta["X"])
    np.save("data/mask.npy", meta["mask"])
    mappings_str = {}
    for key, value in meta["mappings"].items():
        if isinstance(value, dict):
            mappings_str[key] = {str(k): v for k, v in value.items()}
        else:
            mappings_str[key] = value

    with open("data/mappings.json", "w", encoding="utf-8") as f:
        json.dump(mappings_str, f)
    print("[train] Datos preprocesados guardados en data/")

    num_items = meta["num_items"]
    print(f"[train] Nº usuarios: {meta['num_users']} | Nº películas: {num_items}")

    print("[train] Inicializando RBM...")
    rbm = RBM(n_visible=num_items, n_hidden=n_hidden)

    # Construir el modelo (necesario para crear variables)
    rbm.build((None, num_items))

    dummy_input = tf.zeros((1, num_items), dtype=tf.float32)
    dummy_mask = tf.ones((1, num_items), dtype=tf.float32)
    _ = rbm.contrastive_divergence(dummy_input, dummy_mask, lr=0.0, k=1)

    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, "rbm_weights.weights.h5")

    # Si se desea reanudar entrenamiento
    if resume and os.path.exists(weights_path):
        rbm.load_weights(weights_path)
        print(f"[train] Pesos cargados desde {weights_path}")

    # Bucle de entrenamiento
    for epoch in range(1, epochs + 1):
        epoch_losses = []

        for step, (batch_v, batch_mask) in enumerate(ds_train):
            # Un paso de Contrastive Divergence (actualiza pesos)
            loss = rbm.contrastive_divergence(
                v0=batch_v,
                mask=batch_mask,
                lr=lr,
                k=k,
            )
            epoch_losses.append(loss.numpy())

        # Pérdida media de entrenamiento en la epoch
        train_loss = float(tf.reduce_mean(epoch_losses))
        val_loss = float(compute_reconstruction_loss(rbm, ds_val))

        print(
            f"[train] Epoch {epoch:02d}/{epochs} "
            f"- train_loss: {train_loss:.6f} "
            f"- val_loss: {val_loss:.6f}"
        )

        # Guardar pesos cada 5 épocas
        if epoch % 5 == 0 or epoch == epochs:
            rbm.save_weights(weights_path)
            print(f"[train] Pesos guardados en {weights_path}")

    print("[train] Entrenamiento finalizado.")
    print(f"[train] Pesos finales guardados en {weights_path}")
    return rbm


if __name__ == "__main__":
    # Ejecutar desde terminal:
    # python src/train.py
    rbm_model = train_rbm()

