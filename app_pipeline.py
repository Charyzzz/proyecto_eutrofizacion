"""
app_pipeline.py
===============
Lógica del pipeline de detección de anomalías usada por app.py.

Pipeline de máscara (propio de la app, ajustado a partir de
notebooks/detectar_anomalias_pipeline.py):
    U-Net -> cerrar máscaras -> canales a*/L/b* combinados con AND
    -> cerrar más (erosión ajustable, igual que en
    03_pipeline_combinado.ipynb)

Y la misma extracción de superpíxeles de 02_superpixels_training.ipynb
-- pero además clasifica CADA superpíxel como conservado/descartado
(para poder visualizarlo en verde/rojo, igual que
visualizar_superpixeles_descartados()) y separa "entrenar un Isolation
Forest nuevo" de "predecir con uno ya entrenado".

Este módulo NO importa streamlit -- es lógica pura, reutilizable fuera
de la app si hace falta.
"""

import sys
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.segmentation import slic, mark_boundaries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE_PATH = Path(r"D:\proyecto_eutrofizacion")
sys.path.insert(0, str(BASE_PATH / "src"))
sys.path.insert(0, str(BASE_PATH / "notebooks"))

from unet.config import UNetConfig
from unet.inference import load_model
from unet.augmentation import get_val_transforms
from unet.preprocessing import downscale_image
from unet.patches import get_patch_positions
from unet.reconstruction import reconstruct_full_prediction

from water_masc2 import conectar_graffiti_y_cerrar, suavizar_bordes_contorno

# -----------------------------------------------------------------------
# Parámetros de la máscara de color (a*, L, b* -- Lab)
# -----------------------------------------------------------------------
# a* y L: mismos límites que 03_pipeline_combinado.ipynb (última versión).
A_MIN, A_MAX = -120, 1
L_MIN, L_MAX = 50, 210

# b* (Lab, azul-amarillo) centrado va de -128 a 127 (igual que a*).
# Se pidió b_max=260, pero eso no es alcanzable -- se usa 127 (el máximo
# real), que en la práctica no filtra nada por el lado superior; el
# filtro efectivo queda en b* >= -4.
B_MIN, B_MAX = -4, 127

# Erosión final ("cerrar más la máscara"), igual que erosionar_mascara()
# de 03_pipeline_combinado.ipynb. Ajusta estos dos valores si el borde
# sigue dejando pasar sombra/vegetación o si se está comiendo agua real.
KERNEL_SIZE_CERRAR = 21
ITERACIONES_CERRAR = 1

# -----------------------------------------------------------------------
# Parámetros del Isolation Forest (mismos que detectar_anomalias_pipeline.py)
# -----------------------------------------------------------------------
N_SEGMENTS = 20
AREA_MIN = 10000
CONTAMINATION = "auto"
N_ESTIMATORS = 100
SEED = 42

CARACTERISTICAS_DETECCION = [
    "h_mean", "h_std",
    "s_mean", "s_std",
    "v_mean", "v_std",
    "l_mean",
    "a_mean",
    "b_mean",
    "textura_std",
]

MODEL_PATH = BASE_PATH / "models" / "isolation_forest_agua.pkl"


# -----------------------------------------------------------------------
# Carga de modelos
# -----------------------------------------------------------------------

def cargar_unet():
    """Carga la U-Net entrenada + config + transform + device."""
    config = UNetConfig(base_path=BASE_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_val_transforms(config.imagenet_mean, config.imagenet_std)

    checkpoint_path = config.get_path(config.checkpoints_dir) / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint de la U-Net: {checkpoint_path}")

    model = load_model(
        checkpoint_path,
        encoder_name=config.encoder_name,
        encoder_weights=None,
        device=device,
    )
    return model, device, transform, config


def cargar_isolation_forest(ruta=MODEL_PATH):
    """Carga el Isolation Forest + scaler guardados. None si no existe todavía."""
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    with open(ruta, "rb") as f:
        return pickle.load(f)


def guardar_isolation_forest(scaler, modelo, metadata=None, ruta=MODEL_PATH):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    contenido = {
        "scaler": scaler,
        "modelo": modelo,
        "features": CARACTERISTICAS_DETECCION,
        "contamination": CONTAMINATION,
    }
    if metadata:
        contenido.update(metadata)
    with open(ruta, "wb") as f:
        pickle.dump(contenido, f)


# -----------------------------------------------------------------------
# Máscara de color: a* + L + b* combinados con AND
# -----------------------------------------------------------------------

def segmentar_agua_canales_color(imagen_bgr, a_min=A_MIN, a_max=A_MAX, l_min=L_MIN, l_max=L_MAX,
                                  b_min=B_MIN, b_max=B_MAX, downscale=4):
    """
    Máscara de agua usando los canales a* (Lab, rojo-verde), L (Lab,
    iluminancia) y b* (Lab, azul-amarillo), combinados con AND.

    Devuelve la máscara combinada (0/255, uint8) en la resolución
    ORIGINAL de la imagen.
    """
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
    b_channel = lab[:, :, 2].astype(np.float32) - 128

    mask_a_bool = (a_channel >= a_min) & (a_channel <= a_max)
    mask_l_bool = (l_channel >= l_min) & (l_channel <= l_max)
    mask_b_bool = (b_channel >= b_min) & (b_channel <= b_max)

    mask_color = (mask_a_bool & mask_l_bool & mask_b_bool).astype(np.uint8) * 255

    if downscale > 1:
        mask_color = cv2.resize(mask_color, (ancho_orig, altura_orig), interpolation=cv2.INTER_NEAREST)

    return mask_color


def erosionar_mascara(mascara, kernel_size=KERNEL_SIZE_CERRAR, iteraciones=ITERACIONES_CERRAR):
    """
    Reduce el área de una máscara de agua erosionando su borde exterior --
    excluye el anillo donde suele acumularse ruido de vegetación/sombras.
    Igual que erosionar_mascara() de 03_pipeline_combinado.ipynb.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(mascara, kernel, iterations=iteraciones)


def predecir_mascara_final(imagen_bgr, imagen_rgb, model, device, transform, config):
    """
    Pipeline de máscara completo para UNA imagen:
        U-Net -> cerrar máscaras -> canales a*/L/b* (AND) -> cerrar más

    Devuelve la máscara final (0/255, uint8) en la resolución ORIGINAL.
    """
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

    # --- Bloque 3: canales a* + L + b* combinados con AND ---
    mask_color_orig = segmentar_agua_canales_color(imagen_bgr)
    mask_color = cv2.resize(mask_color_orig, (w_ds, h_ds), interpolation=cv2.INTER_NEAREST)
    mask_combinada = cv2.bitwise_and(mask_cerrada, mask_color)

    # --- Bloque 4: cerrar más (erosión ajustable) ---
    mask_erosionada = erosionar_mascara(mask_combinada, KERNEL_SIZE_CERRAR, ITERACIONES_CERRAR)

    # Subimos a la resolución ORIGINAL -- extraer_superpixeles() espera
    # imagen + máscara en resolución original.
    mask_final = cv2.resize(mask_erosionada, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask_final


# -----------------------------------------------------------------------
# Superpíxeles: features de TODOS + clasificación verde/rojo + overlay
# -----------------------------------------------------------------------

def extraer_superpixeles(imagen_bgr, mascara_agua, n_segments=N_SEGMENTS, area_min=AREA_MIN, downscale=4):
    """
    Igual que extraer_caracteristicas_superpixeles() de
    02_superpixels_training.ipynb, pero clasifica TODOS los superpíxeles
    (no solo los que pasan el filtro de área) y arma un overlay
    verde=conservado / rojo=descartado, igual que
    visualizar_superpixeles_descartados().

    Devuelve
    --------
    filas_ok          : list[dict] -- superpíxeles con area >= area_min
                         (los que se usan para el Isolation Forest)
    filas_descartadas : list[dict] -- descartados por área insuficiente
    overlay_rgb       : np.ndarray (H_ds, W_ds, 3) uint8, para mostrar
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mascara_agua = cv2.erode(mascara_agua, kernel, iterations=1)

    if downscale > 1:
        h, w = imagen_bgr.shape[:2]
        imagen = cv2.resize(imagen_bgr, (w // downscale, h // downscale))
        mascara_agua = cv2.resize(mascara_agua, (w // downscale, h // downscale))
    else:
        imagen = imagen_bgr

    img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2Lab)
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    segments = slic(img_rgb, n_segments=n_segments, compactness=10, sigma=1, mask=(mascara_agua > 0))

    filas_ok, filas_descartadas = [], []
    mapa_estado = np.zeros(segments.shape, dtype=np.uint8)  # 0=fondo, 1=conservado, 2=descartado

    for label in np.unique(segments):
        if label == 0:
            continue

        region_mask = (segments == label)
        area = int(region_mask.sum())

        h_vals = hsv[:, :, 0][region_mask]
        s_vals = hsv[:, :, 1][region_mask]
        v_vals = hsv[:, :, 2][region_mask]
        l_vals = lab[:, :, 0][region_mask]
        a_vals = lab[:, :, 1][region_mask].astype(np.float32) - 128
        b_vals = lab[:, :, 2][region_mask].astype(np.float32) - 128
        gray_vals = gray[region_mask]
        ys, xs = np.where(region_mask)

        fila = {
            "h_mean": h_vals.mean(), "h_std": h_vals.std(),
            "s_mean": s_vals.mean(), "s_std": s_vals.std(),
            "v_mean": v_vals.mean(), "v_std": v_vals.std(),
            "l_mean": l_vals.mean(),
            "a_mean": a_vals.mean(),
            "b_mean": b_vals.mean(),
            "textura_std": gray_vals.std(),
            "area_px": area,
            "centroid_x": float(xs.mean()),
            "centroid_y": float(ys.mean()),
            "superpixel_label": int(label),
        }

        if area < area_min:
            mapa_estado[region_mask] = 2
            filas_descartadas.append(fila)
        else:
            mapa_estado[region_mask] = 1
            filas_ok.append(fila)

    overlay = img_rgb.copy()
    overlay[mapa_estado == 1] = (overlay[mapa_estado == 1] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    overlay[mapa_estado == 2] = (overlay[mapa_estado == 2] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    overlay_rgb = (mark_boundaries(overlay, segments, color=(1, 1, 0)) * 255).astype(np.uint8)

    return filas_ok, filas_descartadas, overlay_rgb


def extraer_gps(ruta_imagen):
    """
    Extrae latitud/longitud/altitud del EXIF de una imagen (los drones DJI
    lo incluyen). Devuelve {"lat": float, "lon": float, "alt": float|None}
    o None si la imagen no tiene GPS en su EXIF.
    """
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS

    try:
        with Image.open(ruta_imagen) as img:
            exif = img._getexif()
    except Exception:
        return None

    if exif is None:
        return None

    gps_info = None
    for tag_id, value in exif.items():
        if TAGS.get(tag_id, tag_id) == "GPSInfo":
            gps_info = value
            break

    if not gps_info:
        return None

    gps = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}
    if "GPSLatitude" not in gps or "GPSLongitude" not in gps:
        return None

    def _dms_a_decimal(dms, ref):
        grados, minutos, segundos = (float(v) for v in dms)
        decimal = grados + minutos / 60 + segundos / 3600
        if ref in ("S", "W"):
            decimal = -decimal
        return decimal

    lat = _dms_a_decimal(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
    lon = _dms_a_decimal(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
    alt = float(gps["GPSAltitude"]) if "GPSAltitude" in gps else None

    return {"lat": lat, "lon": lon, "alt": alt}


def procesar_imagen(ruta_imagen, model, device, transform, config):
    """Pipeline completo (máscara + superpíxeles) para UNA imagen."""
    imagen_bgr = cv2.imread(str(ruta_imagen))
    if imagen_bgr is None:
        return None
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

    gps = extraer_gps(ruta_imagen)

    mask_final = predecir_mascara_final(imagen_bgr, imagen_rgb, model, device, transform, config)

    if (mask_final > 0).sum() < 500:
        return {
            "imagen_rgb": imagen_rgb,
            "mask_final": mask_final,
            "filas_ok": [],
            "filas_descartadas": [],
            "overlay_superpixeles": None,
            "sin_agua_suficiente": True,
            "gps": gps,
        }

    filas_ok, filas_descartadas, overlay = extraer_superpixeles(imagen_bgr, mask_final)

    return {
        "imagen_rgb": imagen_rgb,
        "mask_final": mask_final,
        "filas_ok": filas_ok,
        "filas_descartadas": filas_descartadas,
        "overlay_superpixeles": overlay,
        "sin_agua_suficiente": False,
        "gps": gps,
    }


# -----------------------------------------------------------------------
# Combinar resultados de varias imágenes + Isolation Forest
# -----------------------------------------------------------------------

def combinar_filas(resultados_por_imagen):
    """resultados_por_imagen: dict {nombre_imagen: resultado de procesar_imagen()}.
    Devuelve un DataFrame con todos los superpíxeles "ok" de todas las
    imágenes, con columna foto_origen."""
    filas = []
    for nombre, r in resultados_por_imagen.items():
        for f in r["filas_ok"]:
            fila = dict(f)
            fila["foto_origen"] = nombre
            filas.append(fila)

    df = pd.DataFrame(filas)
    if not df.empty:
        df.insert(0, "superpixel_id", range(1, len(df) + 1))
    return df


def entrenar_isolation_forest(df_superpixeles, features=CARACTERISTICAS_DETECCION):
    """Ajusta (fit) un StandardScaler + IsolationForest nuevos sobre
    df_superpixeles. Devuelve (scaler, modelo, df_con_anomalias)."""
    X = df_superpixeles[features].copy()

    scaler = StandardScaler()
    X_esc = scaler.fit_transform(X)

    modelo = IsolationForest(contamination=CONTAMINATION, random_state=SEED, n_estimators=N_ESTIMATORS)
    modelo.fit(X_esc)

    df = df_superpixeles.copy()
    df["anomalia"] = modelo.predict(X_esc)
    df["score_anomalia"] = modelo.score_samples(X_esc)

    return scaler, modelo, df


def predecir_anomalias(df_superpixeles, scaler, modelo, features=CARACTERISTICAS_DETECCION):
    """Aplica un scaler + Isolation Forest YA entrenados (transform/predict,
    nunca fit) sobre df_superpixeles."""
    df = df_superpixeles.copy()
    X = df[features].copy()
    X_esc = scaler.transform(X)
    df["anomalia"] = modelo.predict(X_esc)
    df["score_anomalia"] = modelo.score_samples(X_esc)
    return df


# -----------------------------------------------------------------------
# Visualización de anomalías (círculo rojo), igual que
# visualizar_anomalias_en_imagen() de 02_superpixels_training.ipynb
# -----------------------------------------------------------------------

def dibujar_anomalias(imagen_rgb, df_superpixeles_imagen, downscale=4):
    """Genera una figura de matplotlib (original | máscara de anomalías |
    superposición), marcando con círculos rojos los superpíxeles anómalos."""
    import matplotlib.pyplot as plt

    h, w = imagen_rgb.shape[:2]
    anomalias = df_superpixeles_imagen[df_superpixeles_imagen["anomalia"] == -1]

    mascara_anomalias = np.zeros((h, w, 3), dtype=np.uint8)
    for _, row in anomalias.iterrows():
        cx = int(row["centroid_x"] * downscale)
        cy = int(row["centroid_y"] * downscale)
        cx = min(max(cx, 0), w - 1)
        cy = min(max(cy, 0), h - 1)
        area = int(row["area_px"] * downscale * downscale)
        radio = max(8, int(np.sqrt(area / np.pi)))
        cv2.circle(mascara_anomalias, (cx, cy), radio, (255, 0, 0), -1)

    imagen_resultado = cv2.addWeighted(imagen_rgb, 0.75, mascara_anomalias, 0.5, 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(imagen_rgb)
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    axes[1].imshow(mascara_anomalias)
    axes[1].set_title(f"Anomalías ({len(anomalias)})")
    axes[1].axis("off")

    axes[2].imshow(imagen_resultado)
    axes[2].set_title("Superposición")
    axes[2].axis("off")

    plt.tight_layout()
    return fig
