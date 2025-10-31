# Sistema de Recomendación de Películas con RBM

> **Proyecto de Deep Learning con TensorFlow/PyTorch**  
> Basado en **Restricted Boltzmann Machines (RBM)** y **aprendizaje no supervisado**  
> Dataset: [MovieLens 100K](https://files.grouplens.org/datasets/movielens/ml-100k.zip)

---

## Objetivo

Construir un **sistema de recomendación de películas** que prediga qué películas le gustarán a un usuario,  
a partir de las valoraciones de otros usuarios, utilizando un **Restricted Boltzmann Machine (RBM)**.

El modelo aprenderá **patrones de preferencias** de manera no supervisada,  
descubriendo relaciones ocultas entre usuarios y películas.

---

## Conceptos Aplicados

- **Restricted Boltzmann Machine (RBM)**  
  Modelo probabilístico con dos capas (visible y oculta) que aprende representaciones internas de los datos.

- **Entrenamiento no supervisado**  
  El modelo aprende sin etiquetas, reconstruyendo las entradas originales.

- **Contrastive Divergence (CD)**  
  Algoritmo para ajustar los pesos del RBM.

- **Filtrado Colaborativo**  
  Recomendación basada en el comportamiento de usuarios, no en metadatos de las películas.

- **Reducción de Dimensionalidad Implícita**  
  Cada usuario y película se representan en un espacio latente comprimido.

---

## Estructura Conceptual del RBM

| Capa | Representa | Tamaño aproximado | Descripción |
|------|-------------|-------------------|--------------|
| Visible (v) | Películas | 1682 | Una neurona por película |
| Oculta (h) | Factores de gusto | 100–200 | Captura características latentes (géneros, estilos, etc.) |

El RBM aprende los pesos entre las capas de modo que pueda **reconstruir** las preferencias del usuario.

---

## Estructura del Proyecto

```
movie_recommender_rbm/
│
├── data/
│   └── ml-100k/                     # Dataset MovieLens 100K
│
├── src/
│   ├── preprocess.py                # Limpieza y preparación de datos
│   ├── rbm_model.py                 # Implementación del RBM
│   ├── train.py                     # Entrenamiento del modelo
│   ├── evaluate.py                  # Cálculo de métricas (RMSE, precisión)
│   └── recommend.py                 # Generación de recomendaciones
│
├── notebooks/
│   └── rbm_experiments.ipynb        # Exploraciones y pruebas
│
├── results/
│   ├── training_logs.csv
│   └── visualizations/
│
├── requirements.txt
└── README.md
```

---

## Instalación y Preparación

```bash
# Clonar el repositorio
git clone https://github.com/tuusuario/movie_recommender_rbm.git
cd movie_recommender_rbm

# Crear entorno virtual (opcional)
python -m venv venv
source venv/bin/activate  # en Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Descargar y descomprimir el dataset
!wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
!unzip -o ml-100k.zip -d data/
```

**requirements.txt**
```
torch
pandas
numpy
matplotlib
scikit-learn
```

---

## Entrenamiento del Modelo

Ejemplo (PyTorch):

```python
import torch
from rbm_model import RBM
from train import train_rbm

num_visible = 1682
num_hidden = 128
rbm = RBM(num_visible, num_hidden)

train_rbm(rbm, data, epochs=30, batch_size=64, lr=0.01)
```

### Algoritmo de Entrenamiento (Contrastive Divergence)

**Fase positiva:**
```python
p_h_given_v = torch.sigmoid(torch.matmul(v, W) + h_bias)
h_sample = torch.bernoulli(p_h_given_v)
```

**Fase negativa (reconstrucción):**
```python
p_v_given_h = torch.sigmoid(torch.matmul(h_sample, W.t()) + v_bias)
v_recon = torch.bernoulli(p_v_given_h)
```

**Actualización de pesos:**
```python
W += lr * (torch.matmul(v.t(), p_h_given_v) - torch.matmul(v_recon.t(), p_h_given_v))
```

---

## Evaluación

### Métrica principal: RMSE
```python
rmse = torch.sqrt(torch.mean((pred - real)**2))
```

### Métrica adicional: Precisión en top-N recomendaciones
1. Generar lista de películas no vistas.  
2. Ordenarlas por puntuación predicha.  
3. Ver cuántas coinciden con las mejores valoradas por el usuario.

---

## Recomendación Personalizada

```python
user_input = torch.tensor(user_ratings)
predicted = rbm.reconstruct(user_input)
recommendations = get_top_movies(predicted, seen_movies)
```

Salida esperada:
```
Top 5 recomendaciones para el usuario 25:
1. The Matrix (1999)
2. Fight Club (1999)
3. Pulp Fiction (1994)
4. The Usual Suspects (1995)
5. The Shawshank Redemption (1994)
```

---

## Visualización e Interpretación

- **Pesos aprendidos** → muestran qué películas activan neuronas similares.  
- **Espacio latente (t-SNE)** → usuarios similares se agrupan naturalmente.  
- **Comparativa RMSE** con modelos base (media global o SVD).

---

## Aprendizaje Demostrado

- Implementación desde cero de un **RBM funcional**.  
- Aplicación del **aprendizaje no supervisado** mediante Contrastive Divergence.  
- Comprensión del **espacio latente** como representación de preferencias.  
- Evaluación de modelos de recomendación con **métricas cuantitativas (RMSE, precisión)**.  
- Capacidad de **traducir teoría de redes neuronales a un caso práctico**.

---

## Conclusiones

Este proyecto demuestra cómo un modelo clásico como el RBM puede capturar patrones complejos de preferencia entre usuarios y películas.  
A pesar de su antigüedad, sigue siendo una base teórica clave para entender modelos modernos de recomendación y redes profundas.

---

## Autor

**Marc Plana Villalbi**  
📧 contacto: [marc.planavillalbi@gmail.com]  
📂 GitHub: [github.com/mplanav](https://github.com/mplanav)

---
