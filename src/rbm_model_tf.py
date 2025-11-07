"""
rbm_model_tf.py
----------------
Implementación de un Restricted Boltzmann Machine (RBM) en TensorFlow
para recomendación de películas con MovieLens 100K.

Diseñado para trabajar con:
- Entradas binarias (binarize=True en el preprocesado)
- Matriz usuario × película
- Máscara (mask) para saber qué ratings son observados

Conceptos clave que implementa:
- p(h|v): prob. de activación de unidades ocultas dado v
- p(v|h): prob. de activación de unidades visibles dado h
- Muestreo Bernoulli a partir de probabilidades
- Contrastive Divergence (CD-1) para actualizar W, b_v, b_h
"""

from __future__ import annotations

import tensorflow as tf


class RBM(tf.keras.Model):
    def __init__(self, n_visible: int, n_hidden: int, seed: int = 42):
        """
        n_visible: número de unidades visibles (películas)
        n_hidden: número de unidades ocultas (factores latentes)
        seed: semilla para inicializar pesos de forma reproducible
        """
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.seed = seed

        # Inicializador Glorot/Xavier con semilla fija
        initializer = tf.keras.initializers.GlorotNormal(seed=self.seed)

        # Matriz de pesos W: [n_visible, n_hidden]
        self.W = tf.Variable(
            initializer(shape=(self.n_visible, self.n_hidden)),
            name="W"
        )

        # Sesgos visibles (una por película)
        self.b_v = tf.Variable(
            tf.zeros([self.n_visible], dtype=tf.float32),
            name="b_v"
        )

        # Sesgos ocultos (una por unidad oculta)
        self.b_h = tf.Variable(
            tf.zeros([self.n_hidden], dtype=tf.float32),
            name="b_h"
        )

        def build(self, input_shape):
            """Construye el modelo registrando las formas de entrada."""
            self.built = True

    # --------- Utilidades internas ---------

    @staticmethod
    def sample_prob(probs: tf.Tensor) -> tf.Tensor:
        """
        Recibe probabilidades en [0,1] y devuelve muestras Bernoulli (0 o 1).
        """
        random_uniform = tf.random.uniform(tf.shape(probs), 0.0, 1.0)
        return tf.cast(probs > random_uniform, tf.float32)

    def propup(self, v: tf.Tensor) -> tf.Tensor:
        """
        p(h=1 | v) = sigmoid(vW + b_h)
        v: [batch_size, n_visible]
        """
        return tf.nn.sigmoid(tf.matmul(v, self.W) + self.b_h)

    def propdown(self, h: tf.Tensor) -> tf.Tensor:
        """
        p(v=1 | h) = sigmoid(hW^T + b_v)
        h: [batch_size, n_hidden]
        """
        return tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.b_v)

    def sample_h_given_v(self, v: tf.Tensor):
        """
        Dado v:
        - calcula p(h|v)
        - devuelve (p(h|v), muestra Bernoulli de h)
        """
        p_h = self.propup(v)
        h_sample = self.sample_prob(p_h)
        return p_h, h_sample

    def sample_v_given_h(self, h: tf.Tensor):
        """
        Dado h:
        - calcula p(v|h)
        - devuelve (p(v|h), muestra Bernoulli de v)
        """
        p_v = self.propdown(h)
        v_sample = self.sample_prob(p_v)
        return p_v, v_sample

    # --------- Entrenamiento: Contrastive Divergence ---------

    def contrastive_divergence(
        self,
        v0: tf.Tensor,
        mask: tf.Tensor,
        lr: float = 0.01,
        k: int = 1,
    ) -> tf.Tensor:
        """
        Un paso de Contrastive Divergence (CD-k, por defecto k=1).

        v0: batch de entradas originales [batch_size, n_visible]
        mask: misma shape, 1 donde hay rating observado, 0 donde no
        lr: learning rate
        k: número de pasos Gibbs (CD-k)

        Devuelve:
        - pérdida de reconstrucción (MSE sobre posiciones observadas)
        """
        v0 = tf.cast(v0, tf.float32)
        mask = tf.cast(mask, tf.float32)

        # Fase positiva
        p_h0, h0 = self.sample_h_given_v(v0)

        # Gibbs sampling (k pasos)
        vk = v0
        hk = h0
        for _ in range(k):
            p_vk, vk = self.sample_v_given_h(hk)
            p_hk, hk = self.sample_h_given_v(vk)

        # Aplicar máscara: solo donde había rating
        v0_masked = v0 * mask
        vk_masked = vk * mask

        batch_size = tf.cast(tf.shape(v0)[0], tf.float32)

        # Asociaciones positivas y negativas
        positive_grad = tf.matmul(tf.transpose(v0_masked), p_h0)
        negative_grad = tf.matmul(tf.transpose(vk_masked), p_hk)

        dW = (positive_grad - negative_grad) / batch_size
        db_v = tf.reduce_mean(v0_masked - vk_masked, axis=0)
        db_h = tf.reduce_mean(p_h0 - p_hk, axis=0)

        # Actualizar parámetros
        self.W.assign_add(lr * dW)
        self.b_v.assign_add(lr * db_v)
        self.b_h.assign_add(lr * db_h)

        # Pérdida de reconstrucción (MSE en posiciones observadas)
        loss = tf.reduce_mean(tf.square(v0_masked - vk_masked))
        return loss

    # --------- Reconstrucción (para recomendar) ---------

    def reconstruct(self, v: tf.Tensor) -> tf.Tensor:
        """
        Dado un vector v de un usuario:
        - subimos a h
        - bajamos a v de nuevo
        Devolvemos las probabilidades reconstruidas (scores por película).
        """
        v = tf.cast(v, tf.float32)
        p_h = self.propup(v)
        p_v_recon = self.propdown(p_h)
        return p_v_recon
