"""
scripts/detectar_anomalias_pipeline.py
=======================================
Pipeline completo de detección de anomalías sobre RivAIrSet:

    imagen -> ¿muy parecida a la anterior subida? -> se salta (evita
              tomas solapadas/casi repetidas del mismo vuelo)
           -> máscara binaria (U-Net + cerrar máscaras + canales
              a*/L/b* combinados con AND + cerrar más con erosión
              ajustable) -- misma lógica que usa app_pipeline.py,
              reutilizada de ahí para que ambos scripts no se desvíen
           -> superpíxeles (igual que en 02_superpixels_training.ipynb)
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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE_PATH = Path(r"C:\Users\u_fabcore\Downloads\proyecto_eutrofizacion")
sys.path.insert(0, str(BASE_PATH / "src"))
sys.path.insert(0, str(BASE_PATH))

from unet.config import UNetConfig
from unet.augmentation import get_val_transforms
from unet.inference import load_model

# Reutilizamos la máscara (U-Net + cerrar + a*/L/b* + cerrar más) y la
# extracción de superpíxeles y el chequeo de similitud de app_pipeline.py
# en vez de mantener una segunda copia que se desactualiza -- ese módulo
# está pensado justo para reutilizarse fuera de la app (no importa
# streamlit).
import app_pipeline as pipe

# -----------------------------------------------------------------------
# PARÁMETROS A ELEGIR
# -----------------------------------------------------------------------
ANIOS_ENTRENAMIENTO = [2019, 2020]   # línea base ("normal" del río)
ANIOS_PRUEBA = [2022]          # se evalúan contra esa línea base

SALTO = 10          # 1 = todas las imágenes de cada año; sube para ir más rápido
N_IMAGENES = None  # límite opcional por conjunto (None = sin límite)

SEED = 42

CONTAMINATION = "auto"
N_ESTIMATORS = 100

# Features, tamaño de máscara de color, umbral de similitud, etc. vienen
# de app_pipeline.py (CARACTERISTICAS_DETECCION, N_SEGMENTS, AREA_MIN,
# A_MIN/A_MAX/L_MIN/L_MAX/B_MIN/B_MAX, KERNEL_SIZE_CERRAR,
# ITERACIONES_CERRAR, UMBRAL_SIMILITUD) -- así ambos scripts usan
# siempre los mismos valores, sin duplicarlos aquí.
CARACTERISTICAS_DETECCION = pipe.CARACTERISTICAS_DETECCION

OUTPUT_DIR = BASE_PATH / "data" / "processed"
MODELS_DIR = BASE_PATH / "models"


def procesar_imagen(fila, model, device, transform, config):
    """Pipeline completo para UNA imagen: máscara final (U-Net + cerrar +
    a*/L/b* + cerrar más, vía app_pipeline.predecir_mascara_final) +
    superpíxeles. Devuelve una lista de dicts (una fila por superpíxel),
    lista vacía si no había agua suficiente, o None si no se pudo cargar."""
    img_path = BASE_PATH / fila["filepath"]

    imagen_bgr = cv2.imread(str(img_path))
    if imagen_bgr is None:
        return None
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

    mask_final = pipe.predecir_mascara_final(imagen_bgr, imagen_rgb, model, device, transform, config)

    if (mask_final > 0).sum() < 500:
        return []

    filas_ok, _, _ = pipe.extraer_superpixeles(imagen_bgr, mask_final)
    for f in filas_ok:
        f["foto_origen"] = Path(fila["filepath"]).name
        f["anio"] = fila["group"]

    return filas_ok


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
    saltadas_similitud = []
    ruta_anterior = None
    t0 = time.time()
    n = len(df_imagenes)

    for i, (_, fila) in enumerate(df_imagenes.iterrows()):
        ruta_actual = BASE_PATH / fila["filepath"]

        # Si esta imagen es muy parecida a la anterior (toma solapada/casi
        # repetida del mismo vuelo), se salta sin correr el pipeline pesado.
        if ruta_anterior is not None:
            try:
                similitud = pipe.calcular_similitud_imagenes(ruta_anterior, ruta_actual)
            except Exception:
                similitud = 0.0
            if similitud >= pipe.UMBRAL_SIMILITUD:
                saltadas_similitud.append((fila["filepath"], similitud))
                ruta_anterior = ruta_actual
                continue

        ruta_anterior = ruta_actual

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
        f"\nProcesadas {n - len(errores) - len(saltadas_similitud)}/{n} imágenes | "
        f"{len(todas_las_filas)} superpíxeles | {len(errores)} errores | "
        f"{len(saltadas_similitud)} saltadas por similitud (>={pipe.UMBRAL_SIMILITUD})"
    )
    if errores:
        print(f"  Ejemplos de errores: {errores[:5]}")
    if saltadas_similitud:
        print(f"  Ejemplos de saltadas por similitud: {saltadas_similitud[:5]}")
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
