"""
scripts/evaluate_rf_on_unet_testset.py
=======================================
Evalúa el Random Forest de ML_masc.ipynb (models/water_segmenter_rf.pkl)
sobre las MISMAS 763 imágenes de test que usó la U-Net
(data/splits/unet/test.csv), para comparar el IoU de ambos modelos de
forma justa: mismas imágenes, mismas máscaras ground truth, misma
fórmula de métricas (unet.metrics.compute_metrics).

POR QUÉ ESTO ES NECESARIO
--------------------------
El F1=0.8603 reportado en models/training_report.txt se calculó sobre un
train_test_split de PÍXELES (80/20, sklearn train_test_split), muestreados
de 1000 imágenes. Píxeles de la MISMA imagen pueden caer tanto en train
como en test ahí -> no es una evaluación limpia por imagen held-out, y
tampoco son las mismas imágenes que el test set de la U-Net.

Este script corrige ambas cosas: reconstruye la máscara COMPLETA de cada
imagen de test (igual que segmentar_agua_supervisado() en ML_masc.ipynb)
y calcula Dice/IoU/Precision/Recall sobre la imagen completa, exactamente
como lo hizo el validator.py de la U-Net.

USO
---
    python notebooks/evaluate_rf_on_unet_testset.py
"""

import sys
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BASE_PATH = Path(r"D:\proyecto_eutrofizacion")
sys.path.insert(0, str(BASE_PATH / "src"))

from unet.metrics import compute_metrics, MetricAccumulator

MODEL_PATH = BASE_PATH / "models" / "water_segmenter_rf.pkl"
TEST_SPLIT_PATH = BASE_PATH / "data" / "splits" / "unet" / "test.csv"
DOWNSCALE = 4  # mismo downscale que se usó al entrenar el RF (ML_masc.ipynb)

# Métricas de la U-Net sobre este mismo test set (del log de entrenamiento),
# solo para imprimir la comparación al final.
UNET_GLOBAL = dict(dice=0.9194, iou=0.8508, prec=0.9169, rec=0.9219)
UNET_MEDIA = dict(dice=0.8878, iou=0.8201, prec=0.8863, rec=0.9166)


def extraer_features_pixel(imagen_bgr):
    """Copia exacta de la función de ML_masc.ipynb: 6 features por píxel
    (H, S, V, a*, b*, textura local en ventana 5x5)."""
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    media_local = cv2.blur(gray, (5, 5))
    media_sq_local = cv2.blur(gray ** 2, (5, 5))
    textura = np.sqrt(np.abs(media_sq_local - media_local ** 2))

    features = np.stack([
        hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2],
        lab[:, :, 1], lab[:, :, 2],
        textura,
    ], axis=-1)

    return features.reshape(-1, 6)


def cargar_mascara_desde_labelimg(ruta_json, alto_img, ancho_img):
    """Copia de la función equivalente en ML_masc.ipynb / water_masc1.py."""
    mascara = np.zeros((alto_img, ancho_img), dtype=np.uint8)
    if not Path(ruta_json).is_file():
        return mascara
    try:
        with open(ruta_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return mascara
    for shape in data.get("shapes", []):
        if shape.get("label", "").lower() == "water":
            points = np.array(shape["points"], dtype=np.int32)
            if len(points) > 0:
                cv2.fillPoly(mascara, [points], 255)
    return mascara


def predecir_mascara_rf(modelo, ruta_imagen, downscale=DOWNSCALE):
    """Reproduce segmentar_agua_supervisado() de ML_masc.ipynb: carga,
    downscalea, extrae features y predice la máscara completa."""
    imagen = cv2.imread(str(ruta_imagen))
    if imagen is None:
        raise FileNotFoundError(f"No se pudo cargar: {ruta_imagen}")

    alto_orig, ancho_orig = imagen.shape[:2]
    ancho_small = ancho_orig // downscale
    alto_small = alto_orig // downscale

    imagen_small = cv2.resize(imagen, (ancho_small, alto_small), interpolation=cv2.INTER_AREA)

    features = extraer_features_pixel(imagen_small)
    prediccion = modelo.predict(features)
    mascara_small = prediccion.reshape(alto_small, ancho_small).astype(np.float32)

    return mascara_small, (alto_orig, ancho_orig)


def main():
    print(f"Cargando modelo: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        modelo = pickle.load(f)

    # El modelo fue entrenado con verbose=1 (queda guardado en el pickle),
    # lo que hace que joblib imprima el progreso de sus 100 árboles en
    # paralelo en CADA llamada a predict() (una vez por imagen). Lo apagamos
    # para que solo se vea nuestro propio progreso cada 50 imágenes.
    modelo.verbose = 0

    df_test = pd.read_csv(TEST_SPLIT_PATH)
    print(f"Evaluando sobre {len(df_test)} imágenes (mismo test set que la U-Net)\n")

    acumulador = MetricAccumulator()
    t0 = time.time()

    for i, row in df_test.iterrows():
        img_path = BASE_PATH / row["filepath"]
        mask_path = BASE_PATH / row["segmentation_mask_path"]
        nombre = Path(row["filepath"]).name

        try:
            mascara_pred, (alto_orig, ancho_orig) = predecir_mascara_rf(modelo, img_path)

            mascara_gt = cargar_mascara_desde_labelimg(str(mask_path), alto_orig, ancho_orig)
            alto_small, ancho_small = mascara_pred.shape
            mascara_gt_small = cv2.resize(
                mascara_gt, (ancho_small, alto_small), interpolation=cv2.INTER_NEAREST
            )
            gt_float = (mascara_gt_small > 127).astype(np.float32)

            metrics = compute_metrics(mascara_pred, gt_float, threshold=0.5)
            acumulador.update(metrics)

            if i % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(df_test)}] {nombre}: {metrics}  | {elapsed:.0f}s")

        except Exception as e:
            print(f"  ⚠ Error en {nombre}: {e}")
            continue

    global_metrics = acumulador.compute_global()
    mean_metrics = acumulador.compute_mean()

    print(f"\n{'='*70}")
    print(f"RANDOM FOREST (ML_masc.ipynb) sobre el test set de la U-Net ({len(df_test)} imgs)")
    print(f"{'='*70}")
    print(f"  Global (micro): {global_metrics}")
    print(f"  Media (macro):  {mean_metrics}")
    print(f"{'='*70}")

    print("\nCOMPARACIÓN DIRECTA (mismas 763 imágenes, misma fórmula de métricas)")
    print(f"{'Modelo':<10}{'Promedio':<10}{'Dice':>8}{'IoU':>8}{'Prec':>8}{'Rec':>8}")
    print(f"{'U-Net':<10}{'Global':<10}{UNET_GLOBAL['dice']:>8.4f}{UNET_GLOBAL['iou']:>8.4f}"
          f"{UNET_GLOBAL['prec']:>8.4f}{UNET_GLOBAL['rec']:>8.4f}")
    print(f"{'U-Net':<10}{'Media':<10}{UNET_MEDIA['dice']:>8.4f}{UNET_MEDIA['iou']:>8.4f}"
          f"{UNET_MEDIA['prec']:>8.4f}{UNET_MEDIA['rec']:>8.4f}")
    print(f"{'RF':<10}{'Global':<10}{global_metrics.dice:>8.4f}{global_metrics.iou:>8.4f}"
          f"{global_metrics.precision:>8.4f}{global_metrics.recall:>8.4f}")
    print(f"{'RF':<10}{'Media':<10}{mean_metrics.dice:>8.4f}{mean_metrics.iou:>8.4f}"
          f"{mean_metrics.precision:>8.4f}{mean_metrics.recall:>8.4f}")


if __name__ == "__main__":
    main()
