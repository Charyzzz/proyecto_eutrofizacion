"""
scripts/detectar_anomalias_pipeline.py
=======================================
Pipeline completo de detección de anomalías sobre RivAIrSet:

    imagen -> máscara binaria (pipeline de 03_pipeline_combinado.ipynb:
              U-Net + cerrar máscaras + canales a*/L combinados con AND)
           -> superpíxeles (extraer_caracteristicas_superpixeles, igual
              que en 02_superpixels_training.ipynb)
           -> Isolation Forest

DISEÑO: fit en años base, predict en años de prueba
-----------------------------------------------------
A diferencia de 02_superpixels_training.ipynb (que hace fit_predict()
sobre el mismo lote de imágenes), aquí SEPARAMOS entrenamiento y
evaluación por AÑO:

    ANIOS_ENTRENAMIENTO -> se usa para AJUSTAR (fit) el Isolation Forest
                           y el StandardScaler. Define qué es "normal"
                           para este río.
    ANIOS_PRUEBA        -> se EVALÚA con ese mismo modelo ya ajustado
                           (predict/score_samples, nunca fit ni refit).

Por qué importa: si haces fit_predict() sobre el mismo lote, el parámetro
`contamination` OBLIGA a que ese % exacto del lote salga marcado como
anómalo, sin importar si hay algo realmente raro ahí o no. Separando
fit (años base) de predict (años de prueba), el % de años de prueba
marcado como anómalo YA NO está forzado a ningún valor -- es evidencia
real de cuánto se aleja ese período de la línea base, no un artefacto
del parámetro. Esto es clave para poder decir algo válido sobre si un
período/sitio se ve "raro" respecto a una referencia.

CONTAMINATION = "auto"
-----------------------
No tenemos evidencia de qué fracción de los años base contiene
condiciones genuinamente atípicas, así que fijar un número arbitrario
(ej. 0.01) no está justificado. "auto" usa el criterio del paper
original de Isolation Forest (un umbral fijo sobre el score, no atado
a un porcentaje asumido de antemano) -- más defendible que inventar un
porcentaje sin evidencia.

ADVERTENCIA DE TIEMPO
----------------------
Este script corre el pipeline COMPLETO (incluyendo la U-Net, patch por
patch) sobre potencialmente miles de imágenes. Con ~7600 imágenes en
todo el dataset, esto puede tardar VARIAS HORAS. Los resultados de
superpíxeles se cachean a CSV por conjunto (entrenamiento/prueba) --
si el script se cae o lo detienes, no hace falta re-extraer todo, solo
borra el CSV del conjunto que quieras recalcular. Para una primera
prueba rápida, sube SALTO (ej. 10) o pon un N_IMAGENES bajo.

USO
---
    python notebooks/detectar_anomalias_pipeline.py
"""

import sys
import time
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.segmentation import slic
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE_PATH = Path(r"C:\Users\u_fabcore\Downloads\proyecto_eutrofizacion")
sys.path.insert(0, str(BASE_PATH / "src"))

from unet.config import UNetConfig
from unet.preprocessing import load_image_rgb, downscale_image
from unet.patches import get_patch_positions
from unet.reconstruction import reconstruct_full_prediction
from unet.augmentation import get_val_transforms
from unet.inference import load_model

from water_masc2 import conectar_graffiti_y_cerrar, suavizar_bordes_contorno

# -----------------------------------------------------------------------
# PARÁMETROS A ELEGIR
# -----------------------------------------------------------------------
ANIOS_ENTRENAMIENTO = [2019, 2020]   # línea base ("normal" del río)
ANIOS_PRUEBA = [2022]          # se evalúan contra esa línea base

SALTO = 10          # 1 = todas las imágenes de cada año; sube para ir más rápido
N_IMAGENES = None  # límite opcional por conjunto (None = sin límite)

N_SEGMENTS = 20    # mismo valor que usaba construir_tabla_superpixeles()
SEED = 42

# Mismos límites de a*/L que en 01_water_segmentation.ipynb / 03_pipeline_combinado
A_MIN, A_MAX = -120, 1
L_MIN, L_MAX = 50, 210

CONTAMINATION = "auto"
N_ESTIMATORS = 100

CARACTERISTICAS_DETECCION = [
    "h_mean", "h_std",
    "s_mean", "s_std",
    "v_mean", "v_std",
    "l_mean",
    "a_mean",
    "b_mean",
    "textura_std",
]

OUTPUT_DIR = BASE_PATH / "data" / "processed"
MODELS_DIR = BASE_PATH / "models"


# -----------------------------------------------------------------------
# Bloques 1-3 de 03_pipeline_combinado.ipynb (U-Net + cerrar + a*/L)
# -----------------------------------------------------------------------

def segmentar_agua_canales_a_l(imagen_bgr, a_min, a_max, l_min, l_max, downscale=4):
    """Igual que en 03_pipeline_combinado.ipynb: máscaras de agua usando
    los canales a* (Lab, rojo-verde) y L (Lab, iluminancia), por separado
    y combinados con AND. Devuelve (mask_a, mask_l, mask_a_l) en la
    resolución ORIGINAL de la imagen (0/255, uint8)."""
    altura_orig, ancho_orig = imagen_bgr.shape[:2]

    if downscale > 1:
        imagen_small = cv2.resize(
            imagen_bgr,
            (ancho_orig // downscale, altura_orig // downscale),
            interpolation=cv2.INTER_AREA,
        )
    else:
        imagen_small = imagen_bgr

    lab = cv2.cvtColor(imagen_small, cv2.COLOR_BGR2Lab)
    l_channel = lab[:, :, 0].astype(np.float32)
    a_channel = lab[:, :, 1].astype(np.float32) - 128

    mask_a_bool = (a_channel >= a_min) & (a_channel <= a_max)
    mask_l_bool = (l_channel >= l_min) & (l_channel <= l_max)

    mask_a = mask_a_bool.astype(np.uint8) * 255
    mask_l = mask_l_bool.astype(np.uint8) * 255
    mask_a_l = (mask_a_bool & mask_l_bool).astype(np.uint8) * 255

    if downscale > 1:
        size = (ancho_orig, altura_orig)
        mask_a = cv2.resize(mask_a, size, interpolation=cv2.INTER_NEAREST)
        mask_l = cv2.resize(mask_l, size, interpolation=cv2.INTER_NEAREST)
        mask_a_l = cv2.resize(mask_a_l, size, interpolation=cv2.INTER_NEAREST)

    return mask_a, mask_l, mask_a_l


def predecir_mascara_final(imagen_bgr, imagen_rgb, model, device, transform, config):
    """Ejecuta los bloques 1-3 de 03_pipeline_combinado.ipynb para UNA
    imagen y devuelve la máscara final (0/255, uint8) en la resolución
    ORIGINAL de la imagen."""
    h, w = imagen_rgb.shape[:2]

    # --- Bloque 1: U-Net ---
    image_ds = downscale_image(imagen_rgb, config.downscale_factor)
    h_ds, w_ds = image_ds.shape[:2]
    positions = get_patch_positions(h_ds, w_ds, config.patch_size, config.stride)
    _, mask_unet = reconstruct_full_prediction(
        model=model,
        image_ds=image_ds,
        positions=positions,
        transform=transform,
        patch_size=config.patch_size,
        threshold=config.threshold,
        batch_size=config.batch_size * 2,
        device=device,
    )

    # --- Bloque 2: Cerrar máscaras ---
    mask_cerrada = conectar_graffiti_y_cerrar(mask_unet, skeleton_kernel=3, dilation_kernel=30)
    mask_cerrada = suavizar_bordes_contorno(mask_cerrada, contour_approx=7)

    # --- Bloque 3: canales a* y L combinados con AND ---
    _, _, mask_a_l_orig = segmentar_agua_canales_a_l(
        imagen_bgr, a_min=A_MIN, a_max=A_MAX, l_min=L_MIN, l_max=L_MAX, downscale=4
    )
    mask_a_l = cv2.resize(mask_a_l_orig, (w_ds, h_ds), interpolation=cv2.INTER_NEAREST)
    mask_final_ds = cv2.bitwise_and(mask_cerrada, mask_a_l)

    # Subimos la máscara final (resolución x2 de la U-Net) a la resolución
    # ORIGINAL -- extraer_caracteristicas_superpixeles() espera imagen +
    # máscara en resolución original (hace su propio downscale x4 adentro).
    mask_final = cv2.resize(mask_final_ds, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask_final


# -----------------------------------------------------------------------
# Extracción de superpíxeles -- IGUAL a 02_superpixels_training.ipynb
# -----------------------------------------------------------------------

def extraer_caracteristicas_superpixeles(imagen, mascara_agua, n_segments=20):
    """Copia exacta de la función de 02_superpixels_training.ipynb."""
    downscale = 2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mascara_agua = cv2.erode(mascara_agua, kernel, iterations=1)

    if downscale > 1:
        h, w = imagen.shape[:2]
        imagen = cv2.resize(imagen, (w // downscale, h // downscale))
        mascara_agua = cv2.resize(mascara_agua, (w // downscale, h // downscale))

    img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2Lab)
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    segments = slic(
        img_rgb, n_segments=n_segments, compactness=10, sigma=1, mask=(mascara_agua > 0)
    )

    filas = []
    for label in np.unique(segments):
        if label == 0:
            continue

        region_mask = (segments == label)
        area = region_mask.sum()
        if area < 10000:
            continue

        h_vals = hsv[:, :, 0][region_mask]
        s_vals = hsv[:, :, 1][region_mask]
        v_vals = hsv[:, :, 2][region_mask]
        l_vals = lab[:, :, 0][region_mask]
        a_vals = lab[:, :, 1][region_mask].astype(np.float32) - 128
        b_vals = lab[:, :, 2][region_mask].astype(np.float32) - 128
        gray_vals = gray[region_mask]
        textura_std = gray_vals.std()

        ys, xs = np.where(region_mask)

        filas.append({
            "h_mean": h_vals.mean(), "h_std": h_vals.std(),
            "s_mean": s_vals.mean(), "s_std": s_vals.std(),
            "v_mean": v_vals.mean(), "v_std": v_vals.std(),
            "l_mean": l_vals.mean(),
            "a_mean": a_vals.mean(),
            "b_mean": b_vals.mean(),
            "textura_std": textura_std,
            "area_px": int(area),
            "centroid_x": float(xs.mean()),
            "centroid_y": float(ys.mean()),
            "superpixel_label": int(label),
        })

    return filas


def procesar_imagen(fila, model, device, transform, config):
    """Pipeline completo para UNA imagen: máscara final (bloques 1-3) +
    superpíxeles. Devuelve una lista de dicts (una fila por superpíxel),
    lista vacía si no había agua suficiente, o None si no se pudo cargar."""
    img_path = BASE_PATH / fila["filepath"]

    imagen_bgr = cv2.imread(str(img_path))
    if imagen_bgr is None:
        return None
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

    mask_final = predecir_mascara_final(imagen_bgr, imagen_rgb, model, device, transform, config)

    # Misma erosión extra que aplicaba construir_tabla_superpixeles() antes
    # de pasar la máscara a SLIC (además de la que ya hace la función misma).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_final = cv2.erode(mask_final, kernel, iterations=2)

    if (mask_final > 0).sum() < 500:
        return []

    filas = extraer_caracteristicas_superpixeles(imagen_bgr, mask_final, n_segments=N_SEGMENTS)
    for f in filas:
        f["foto_origen"] = Path(fila["filepath"]).name
        f["anio"] = fila["group"]

    return filas


# -----------------------------------------------------------------------
# Selección de imágenes por AÑOS (lista, no un solo año) + caché a CSV
# -----------------------------------------------------------------------

def seleccionar_imagenes_por_anios(df, anios, salto=3, n_imagenes=None):
    anios_str = [str(a) for a in anios]
    df_anios = df[df["group"].astype(str).isin(anios_str)].copy().reset_index(drop=True)

    indices = range(0, len(df_anios), salto)
    df_sel = df_anios.iloc[list(indices)].reset_index(drop=True)

    if n_imagenes is not None:
        df_sel = df_sel.head(n_imagenes).reset_index(drop=True)

    return df_sel


def construir_o_cargar_superpixeles(df_imagenes, nombre_cache, model, device, transform, config,
                                     forzar=False):
    ruta_cache = OUTPUT_DIR / f"superpixeles_{nombre_cache}.csv"

    if ruta_cache.exists() and not forzar:
        print(f"Cache encontrado, cargando sin reprocesar: {ruta_cache}")
        return pd.read_csv(ruta_cache)

    todas_las_filas = []
    errores = []
    t0 = time.time()
    n = len(df_imagenes)

    for i, (_, fila) in enumerate(df_imagenes.iterrows()):
        try:
            filas = procesar_imagen(fila, model, device, transform, config)
        except Exception as e:
            errores.append((fila["filepath"], str(e)))
            continue

        if filas is None:
            errores.append((fila["filepath"], "no se pudo cargar la imagen"))
            continue

        todas_las_filas.extend(filas)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            restante = elapsed / (i + 1) * (n - i - 1)
            print(
                f"  [{i + 1}/{n}] {len(todas_las_filas)} superpíxeles acumulados | "
                f"{elapsed:.0f}s transcurridos | ~{restante:.0f}s restantes"
            )

    df_tabla = pd.DataFrame(todas_las_filas)
    if not df_tabla.empty:
        df_tabla.insert(0, "superpixel_id", range(1, len(df_tabla) + 1))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_tabla.to_csv(ruta_cache, index=False)

    print(
        f"\nProcesadas {n - len(errores)}/{n} imágenes | "
        f"{len(todas_las_filas)} superpíxeles | {len(errores)} errores"
    )
    if errores:
        print(f"  Ejemplos de errores: {errores[:5]}")
    print(f"Guardado: {ruta_cache}")

    return df_tabla


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    config = UNetConfig(base_path=BASE_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    transform = get_val_transforms(config.imagenet_mean, config.imagenet_std)

    checkpoint_path = config.get_path(config.checkpoints_dir) / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"No se encontró el checkpoint: {checkpoint_path}")
        sys.exit(1)

    model = load_model(
        checkpoint_path,
        encoder_name=config.encoder_name,
        encoder_weights=None,
        device=device,
    )

    df_river = pd.read_csv(config.get_path(config.csv_path))

    df_train_imgs = seleccionar_imagenes_por_anios(df_river, ANIOS_ENTRENAMIENTO, SALTO, N_IMAGENES)
    df_test_imgs = seleccionar_imagenes_por_anios(df_river, ANIOS_PRUEBA, SALTO, N_IMAGENES)

    print(f"\nImágenes de entrenamiento (línea base) {ANIOS_ENTRENAMIENTO}: {len(df_train_imgs)}")
    print(f"Imágenes de prueba {ANIOS_PRUEBA}: {len(df_test_imgs)}\n")

    print("=== Extrayendo superpíxeles: ENTRENAMIENTO (línea base) ===")
    df_sp_train = construir_o_cargar_superpixeles(
        df_train_imgs, "train_baseline", model, device, transform, config
    )

    print("\n=== Extrayendo superpíxeles: PRUEBA ===")
    df_sp_test = construir_o_cargar_superpixeles(
        df_test_imgs, "test_evaluacion", model, device, transform, config
    )

    print(f"\nSuperpíxeles entrenamiento: {len(df_sp_train)} | prueba: {len(df_sp_test)}\n")

    X_train = df_sp_train[CARACTERISTICAS_DETECCION].copy()
    X_test = df_sp_test[CARACTERISTICAS_DETECCION].copy()

    # El scaler se ajusta SOLO con años base; a prueba se le aplica ese mismo
    # scaler (transform, nunca fit) para que quede en la misma escala.
    scaler = StandardScaler()
    X_train_esc = scaler.fit_transform(X_train)
    X_test_esc = scaler.transform(X_test)

    print(f"=== Entrenando Isolation Forest (contamination={CONTAMINATION!r}) ===")
    iso_forest = IsolationForest(
        contamination=CONTAMINATION,
        random_state=SEED,
        n_estimators=N_ESTIMATORS,
    )
    iso_forest.fit(X_train_esc)  # fit SOLO con años base

    df_sp_train["anomalia"] = iso_forest.predict(X_train_esc)
    df_sp_train["score_anomalia"] = iso_forest.score_samples(X_train_esc)

    # predict/score_samples, NUNCA fit, sobre los años de prueba
    df_sp_test["anomalia"] = iso_forest.predict(X_test_esc)
    df_sp_test["score_anomalia"] = iso_forest.score_samples(X_test_esc)

    n_anom_train = int((df_sp_train["anomalia"] == -1).sum())
    n_anom_test = int((df_sp_test["anomalia"] == -1).sum())

    print(f"\n{'=' * 70}")
    print(f"Anomalías en entrenamiento {ANIOS_ENTRENAMIENTO}: "
          f"{n_anom_train}/{len(df_sp_train)} ({n_anom_train / len(df_sp_train) * 100:.2f}%)")
    print(f"Anomalías en prueba       {ANIOS_PRUEBA}: "
          f"{n_anom_test}/{len(df_sp_test)} ({n_anom_test / len(df_sp_test) * 100:.2f}%)")
    print(f"{'=' * 70}")
    print(
        "\nNota: con contamination='auto' y fit() solo en entrenamiento, el %\n"
        "de prueba NO está forzado a ningún valor -- es evidencia real de\n"
        "cuánto se aleja ese período de la línea base, no un artefacto del\n"
        "parámetro (a diferencia de fit_predict() sobre el mismo lote)."
    )

    df_sp_train.to_csv(OUTPUT_DIR / "superpixeles_train_con_anomalias.csv", index=False)
    df_sp_test.to_csv(OUTPUT_DIR / "superpixeles_test_con_anomalias.csv", index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    modelo_path = MODELS_DIR / "isolation_forest_agua.pkl"
    with open(modelo_path, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "modelo": iso_forest,
            "features": CARACTERISTICAS_DETECCION,
            "anios_entrenamiento": ANIOS_ENTRENAMIENTO,
            "anios_prueba": ANIOS_PRUEBA,
            "contamination": CONTAMINATION,
        }, f)
    print(f"\nModelo (scaler + Isolation Forest) guardado en: {modelo_path}")


if __name__ == "__main__":
    main()
