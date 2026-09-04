"""
scripts/train_rf_full_dataset.py
=================================
Reentrena el Random Forest de segmentación de agua (basado en las
funciones de ML_masc.ipynb), con dos cambios respecto a la versión
original para que sea comparable con la U-Net:

1. USA TODAS LAS IMÁGENES DE TRAIN, no una muestra aleatoria de 1000.
   Sigue muestreando píxeles balanceados (agua/no-agua) por imagen
   -- entrenar con 6000+ imágenes completas, píxel por píxel, sería
   inviable en RAM -- pero ahora ve variedad de TODAS las imágenes de
   train, no solo una sexta parte de ellas.

2. USA LOS MISMOS SPLITS train/val/test QUE LA U-NET
   (data/splits/unet/, a nivel de IMAGEN completa, seed=42), en vez de
   un train_test_split de píxeles sueltos. Esto es clave: el modelo
   nunca ve las imágenes de val/test durante el entrenamiento, y se
   evalúa reconstruyendo la máscara de la imagen COMPLETA (igual que
   hace la U-Net), no sobre una muestra de píxeles ya balanceada
   artificialmente. Así el IoU final es comparable de verdad contra
   el de la U-Net.

USO
---
    python notebooks/train_rf_full_dataset.py

Tarda bastante (extrae features de ~6100 imágenes de train + evalúa
~1500 imágenes completas de val/test). Déjalo correr en background si
hace falta.
"""

import sys
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE_PATH = Path(r"D:\proyecto_eutrofizacion")
sys.path.insert(0, str(BASE_PATH / "src"))

from unet.config import UNetConfig
from unet.splits import load_or_create_splits
from unet.metrics import compute_metrics, MetricAccumulator

MODEL_OUTPUT = BASE_PATH / "models" / "water_segmenter_rf.pkl"
REPORT_OUTPUT = BASE_PATH / "models" / "training_report.txt"

DOWNSCALE = 4          # igual que ML_masc.ipynb
MUESTREO_PIXELES = 2000  # píxeles balanceados (agua/no-agua) por imagen de train
N_ESTIMATORS = 100
MAX_DEPTH = 15
SEED = 42


# ---------------------------------------------------------------------
# Funciones tomadas de ML_masc.ipynb (sin cambios en la lógica interna)
# ---------------------------------------------------------------------

def extraer_features_pixel(imagen_bgr):
    """6 features por píxel: H, S, V, a*, b*, textura local (5x5)."""
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
    """Máscara binaria (255=agua) desde el JSON de anotación (labelme/labelimg)."""
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
    """Predicción por imagen completa: carga, downscalea, features, predict."""
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


# ---------------------------------------------------------------------
# Nuevo: dataset de entrenamiento sobre TODAS las imágenes de un split
# ---------------------------------------------------------------------

def construir_dataset_entrenamiento(df, base_path, muestreo_pixeles=MUESTREO_PIXELES,
                                     downscale=DOWNSCALE, seed=SEED):
    """
    Extrae píxeles balanceados (agua/no-agua) de CADA imagen del DataFrame.
    A diferencia de ML_masc.ipynb, no se submuestrean imágenes: se procesan
    todas las de df (normalmente df_train completo).
    """
    rng = np.random.default_rng(seed)
    X_all, y_all = [], []
    procesadas, saltadas = 0, 0
    t0 = time.time()
    n = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        img_path = base_path / row["filepath"]
        mask_path = base_path / row["segmentation_mask_path"]

        img = cv2.imread(str(img_path))
        if img is None:
            saltadas += 1
            continue

        alto_orig, ancho_orig = img.shape[:2]
        mask_gt = cargar_mascara_desde_labelimg(str(mask_path), alto_orig, ancho_orig)

        if (mask_gt > 0).sum() == 0:
            saltadas += 1
            continue

        img_small = cv2.resize(
            img, (ancho_orig // downscale, alto_orig // downscale), interpolation=cv2.INTER_AREA
        )
        mask_small = cv2.resize(
            mask_gt, (ancho_orig // downscale, alto_orig // downscale), interpolation=cv2.INTER_NEAREST
        )

        features = extraer_features_pixel(img_small)
        labels = (mask_small.flatten() > 0).astype(int)

        idx_agua = np.where(labels == 1)[0]
        idx_no_agua = np.where(labels == 0)[0]
        n_muestra = min(muestreo_pixeles // 2, len(idx_agua), len(idx_no_agua))

        if n_muestra == 0:
            saltadas += 1
            continue

        idx_sel = np.concatenate([
            rng.choice(idx_agua, n_muestra, replace=False),
            rng.choice(idx_no_agua, n_muestra, replace=False),
        ])

        X_all.append(features[idx_sel])
        y_all.append(labels[idx_sel])
        procesadas += 1

        if procesadas % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n}] {procesadas} imágenes procesadas... ({elapsed:.0f}s)")

    print(f"\nProcesadas: {procesadas} | Saltadas (sin agua/error): {saltadas}")

    if not X_all:
        raise RuntimeError("No se pudo extraer ningún píxel de entrenamiento.")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    print(f"Píxeles totales: {len(X):,} | Agua: {(y == 1).sum():,} | No-agua: {(y == 0).sum():,}")
    return X, y


def evaluar_sobre_imagenes_completas(modelo, df, base_path, downscale=DOWNSCALE, log_cada=100):
    """Evalúa el modelo reconstruyendo la máscara COMPLETA de cada imagen
    (no píxeles sueltos), igual que hace validator.py de la U-Net."""
    acumulador = MetricAccumulator()
    n = len(df)
    t0 = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        img_path = base_path / row["filepath"]
        mask_path = base_path / row["segmentation_mask_path"]
        nombre = Path(row["filepath"]).name

        try:
            mascara_pred, (alto_orig, ancho_orig) = predecir_mascara_rf(modelo, img_path, downscale)
            mascara_gt = cargar_mascara_desde_labelimg(str(mask_path), alto_orig, ancho_orig)
            alto_small, ancho_small = mascara_pred.shape
            mascara_gt_small = cv2.resize(
                mascara_gt, (ancho_small, alto_small), interpolation=cv2.INTER_NEAREST
            )
            gt_float = (mascara_gt_small > 127).astype(np.float32)

            m = compute_metrics(mascara_pred, gt_float, threshold=0.5)
            acumulador.update(m)

            if i % log_cada == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{n}] {nombre}: {m}  | {elapsed:.0f}s")

        except Exception as e:
            print(f"  ⚠ Error en {nombre}: {e}")
            continue

    return acumulador.compute_global(), acumulador.compute_mean()


def main():
    config = UNetConfig(base_path=BASE_PATH)

    print("=== Cargando splits (los mismos que usa la U-Net) ===")
    df_train, df_val, df_test = load_or_create_splits(
        csv_path=config.get_path(config.csv_path),
        splits_dir=config.get_path(config.splits_dir),
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}\n")

    print("=== PASO 1: Extrayendo píxeles de TODAS las imágenes de train ===")
    t0 = time.time()
    X, y = construir_dataset_entrenamiento(df_train, BASE_PATH)
    print(f"Extracción completada en {time.time() - t0:.0f}s\n")

    print("=== PASO 2: Entrenando Random Forest ===")
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=SEED,
        n_jobs=-1,
        verbose=1,
    )
    clf.fit(X, y)
    clf.verbose = 0  # apagar antes de guardar/predecir (si no, spamea joblib en cada predict())
    print(f"Entrenamiento completado en {time.time() - t0:.0f}s\n")

    print("=== PASO 3: Evaluando sobre imágenes COMPLETAS de val ===")
    val_global, val_mean = evaluar_sobre_imagenes_completas(clf, df_val, BASE_PATH)
    print(f"\n  Val Global (micro): {val_global}")
    print(f"  Val Media  (macro): {val_mean}\n")

    print("=== PASO 4: Evaluando sobre imágenes COMPLETAS de test ===")
    test_global, test_mean = evaluar_sobre_imagenes_completas(clf, df_test, BASE_PATH)
    print(f"\n  Test Global (micro): {test_global}")
    print(f"  Test Media  (macro): {test_mean}\n")

    print("=== PASO 5: Guardando modelo y reporte ===")
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(clf, f)
    print(f"Modelo guardado: {MODEL_OUTPUT}")

    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("REPORTE DE ENTRENAMIENTO - SEGMENTADOR DE AGUA (Random Forest)\n")
        f.write("Entrenado con TODAS las imagenes de train; evaluado por\n")
        f.write("imagen COMPLETA sobre los mismos splits que la U-Net.\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Imagenes -> train: {len(df_train)} | val: {len(df_val)} | test: {len(df_test)}\n")
        f.write(f"Pixeles de entrenamiento: {len(X)}\n\n")
        f.write("VAL (imagen completa)\n")
        f.write(f"  Global (micro): {val_global}\n")
        f.write(f"  Media  (macro): {val_mean}\n\n")
        f.write("TEST (imagen completa)\n")
        f.write(f"  Global (micro): {test_global}\n")
        f.write(f"  Media  (macro): {test_mean}\n")
    print(f"Reporte guardado: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
