# Carga y clasificación de frutas (apple, banana, orange) con CNN
# - Cargar solo imágenes de las frutas seleccionadas (apple, banana, orange)
# - Redimensionar a 128x128
# - Normalizar RGB a [0,1]
# - Dividir en train/test 80/20
# - CNN con >= 3 capas Conv2D (ReLU) + MaxPooling2D
# - Flatten + Dense(256) + Dropout(0.3)

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.data import AUTOTUNE
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling
)
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report, confusion_matrix

# Rutas y parámetros
CLASSES = ["apple", "banana", "orange"]
IMG_SIZE = (128, 128)
BATCH_SIZE = 64
SEED = 42
EPOCHS = 50
SKIP_PLOTS = os.getenv("SKIP_PLOTS", "0") == "1"

# Ruta base robusta relativa a este archivo: CNN/data/frutas/
BASE_DIR = os.path.join(os.path.dirname(__file__), "data", "frutas")
TRAIN_DIR = os.path.join(BASE_DIR, "train")

# Cargar datasets desde carpetas: 80% train, 20% test (usando validation_split)
train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASSES,              # solo estas clases
    validation_split=0.2,         # 80/20
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
)

test_ds = image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASSES,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
)

# Normalización a [0,1] y optimización del pipeline
norm = Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

num_classes = len(CLASSES)
print("Clases:", CLASSES)

# Mostrar la primera imagen de train y de test

def _first_example(ds):
    for x, y in ds.take(1):
        x0 = x[0].numpy()
        y0 = int(tf.argmax(y[0]).numpy())
        return x0, y0
    return None, None

x_train0, y_train0 = _first_example(train_ds)
x_test0, y_test0 = _first_example(test_ds)

if x_train0 is not None and x_test0 is not None:
    plt.figure(figsize=(5, 5))
    plt.subplot(121)
    plt.imshow(x_train0)
    plt.title(f"Train: {CLASSES[y_train0]}")
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(x_test0)
    plt.title(f"Test: {CLASSES[y_test0]}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Grilla 3x3 de ejemplos de train y test con etiquetas

def _plot_grid(ds, title, n=9):
    imgs, labels = [], []
    for x, y in ds.unbatch().take(n):
        imgs.append(x.numpy())
        labels.append(int(tf.argmax(y).numpy()))
    if not imgs:
        return
    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, (img, lab) in enumerate(zip(imgs, labels)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(CLASSES[lab])
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

_plot_grid(train_ds, "Train samples (3x3)")
_plot_grid(test_ds, "Test samples (3x3)")

# Definición del modelo CNN (3x Conv2D+ReLU+MaxPool) + Flatten + Dense(256) + Dropout(0.3)
model = Sequential([
    Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPoolqing2D((2, 2)),

    Conv2D(64, (3, 3), padding="same", activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), padding="same", activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax"),
])

model.compile(
    loss="binary_crossentropy",
    optimizer='adam',
    metrics=["accuracy"],
)

model.summary()

# Entrenamiento
EPOCHS = int(os.getenv("EPOCHS", EPOCHS))
STEPS_PER_EPOCH = os.getenv("STEPS_PER_EPOCH")
VALIDATION_STEPS = os.getenv("VALIDATION_STEPS")
STEPS_PER_EPOCH = int(STEPS_PER_EPOCH) if STEPS_PER_EPOCH is not None else None
VALIDATION_STEPS = int(VALIDATION_STEPS) if VALIDATION_STEPS is not None else None

fit_kwargs = {}
if STEPS_PER_EPOCH is not None:
    fit_kwargs["steps_per_epoch"] = STEPS_PER_EPOCH
if VALIDATION_STEPS is not None:
    fit_kwargs["validation_steps"] = VALIDATION_STEPS

history = model.fit(
    train_ds,
    validation_data=test_ds,  # usamos el 20% como conjunto de prueba/validación
    epochs=EPOCHS,
    verbose=1,
    **fit_kwargs,
)

# Evaluación en test (20%)
print("\nEvaluación en test (20%):")
loss, acc = model.evaluate(test_ds, verbose=0)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {acc:.4f}")

# Reporte de clasificación y matriz de confusión
# Recolectar etiquetas verdaderas y predichas del test_ds
y_true = []
for _, y in test_ds.unbatch():
    y_true.append(tf.argmax(y, axis=-1).numpy())
y_true = np.array(y_true)

# Recolectar imágenes y etiquetas en orden determinista para graficar
_test_images = []
_test_labels = []
for x, y in test_ds.unbatch():
    _test_images.append(x.numpy())
    _test_labels.append(int(tf.argmax(y).numpy()))
if _test_images:
    _test_images = np.stack(_test_images)
    y_true = np.array(_test_labels)
    # Predecir 1:1 sobre el array (alineado con las imágenes)
    y_pred_prob = model.predict(_test_images, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
else:
    # Fallback a la predicción anterior sobre el dataset
    y_pred_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

# Mostrar imágenes correctas e incorrectas por clase

def _plot_samples(indices, title, max_n=6):
    if not indices:
        return
    n = min(len(indices), max_n)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, idx in enumerate(indices[:n]):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(_test_images[idx])
        plt.title(f"True: {CLASSES[y_true[idx]]}\nPred: {CLASSES[y_pred[idx]]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

for c, name in enumerate(CLASSES):
    correct_idx = [i for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt == c and yp == c]
    wrong_idx = [i for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt == c and yp != c]

    # Correctos por clase
    _plot_samples(
        correct_idx,
        title=f"Correctos: {name} (n={len(correct_idx)})",
        max_n=6,
    )
    # Incorrectos por clase (muestra el verdadero y el predicho en el título por imagen)
    if wrong_idx:
        n = min(len(wrong_idx), 6)
        cols = 3
        rows = (n + cols - 1) // cols
        plt.figure(figsize=(cols * 3, rows * 3))
        for i, idx in enumerate(wrong_idx[:n]):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(_test_images[idx])
            plt.title(f"True: {CLASSES[y_true[idx]]}\nPred: {CLASSES[y_pred[idx]]}")
            plt.axis('off')
        plt.suptitle(f"Incorrectos: {name} (n={len(wrong_idx)})")
        plt.tight_layout()
        plt.show()

print("\nReporte de clasificación:")
print(classification_report(y_true, y_pred, labels=list(range(num_classes)), target_names=CLASSES, zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
print("Matriz de confusión:\n", cm)

# Curvas de entrenamiento
acc_hist = history.history.get('accuracy', [])
val_acc_hist = history.history.get('val_accuracy', [])
loss_hist = history.history.get('loss', [])
val_loss_hist = history.history.get('val_loss', [])

if acc_hist and val_acc_hist:
    epochs_range = range(len(acc_hist))
    plt.figure()
    plt.plot(epochs_range, acc_hist, 'bo-', label='Training accuracy')
    plt.plot(epochs_range, val_acc_hist, 'r*-', label='Validation/Test accuracy')
    plt.title('Accuracy')
    plt.legend()

if loss_hist and val_loss_hist:
    epochs_range = range(len(loss_hist))
    plt.figure()
    plt.plot(epochs_range, loss_hist, 'bo-', label='Training loss')
    plt.plot(epochs_range, val_loss_hist, 'r*-', label='Validation/Test loss')
    plt.title('Loss')
    plt.legend()

plt.show()
