"""
preprocessing.py
================
Carga de imagen RGB y máscara binaria desde el formato del proyecto,
downscale ×2 y conversión a tensores.

Reutiliza cargar_mascara_desde_labeling() de water_masc1.py para
mantener consistencia con el pipeline existente.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
import sys

logger = logging.getLogger(__name__)


def load_image_rgb(image_path: Path) -> np.ndarray:
    """
    Carga una imagen en formato RGB (H, W, 3), dtype uint8.

    Parameters
    ----------
    image_path : Path
        Ruta a la imagen JPG/PNG

    Returns
    -------
    np.ndarray
        Imagen RGB (H, W, 3), uint8
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask_binary(
    mask_path: Path,
    image_height: int,
    image_width: int,
    base_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Carga la máscara binaria de agua desde el archivo JSON de anotación.
    Utiliza cargar_mascara_desde_labeling() del proyecto existente.

    Parameters
    ----------
    mask_path : Path
        Ruta al archivo JSON de anotación
    image_height, image_width : int
        Dimensiones de la imagen original (para rasterizar la máscara)
    base_path : Path, optional
        Ruta base del proyecto (para agregar src/ al sys.path si hace falta)

    Returns
    -------
    np.ndarray
        Máscara binaria (H, W), dtype uint8, valores {0, 255}
        255 = agua, 0 = no-agua
    """
    # Asegurar que src/ está en el path para importar water_masc1
    if base_path is not None:
        src_path = str(base_path / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

    try:
        from water_masc1 import cargar_mascara_desde_labeling
    except ImportError as e:
        raise ImportError(
            "No se pudo importar cargar_mascara_desde_labeling de water_masc1.py. "
            f"Verifica que BASE_PATH/src/ está en sys.path. Error: {e}"
        )

    mask = cargar_mascara_desde_labeling(
        str(mask_path), image_height, image_width
    )

    if mask is None:
        logger.warning(f"Máscara vacía o inválida para: {mask_path}")
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Normalizar a valores {0, 255}
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)

    return mask.astype(np.uint8)


def downscale_image(
    image: np.ndarray,
    factor: int = 2,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Reduce la resolución de una imagen RGB por un factor.

    Usa INTER_AREA por defecto: interpolación apropiada para reducción
    de imágenes (antialiasing sin artefactos, preserva colores).

    Parameters
    ----------
    image : np.ndarray
        Imagen RGB (H, W, 3)
    factor : int
        Factor de reducción (2 = mitad)
    interpolation : int
        Interpolación de OpenCV. INTER_AREA para imágenes.

    Returns
    -------
    np.ndarray
        Imagen reducida (H//factor, W//factor, 3)
    """
    h, w = image.shape[:2]
    new_w, new_h = w // factor, h // factor
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def downscale_mask(
    mask: np.ndarray,
    factor: int = 2,
) -> np.ndarray:
    """
    Reduce la resolución de una máscara binaria por un factor.

    SIEMPRE usa INTER_NEAREST para preservar valores discretos {0, 255}.
    No usar bilinear ni bicúbica: crearían valores intermedios que
    corrompen la máscara binaria.

    Parameters
    ----------
    mask : np.ndarray
        Máscara binaria (H, W), valores {0, 255}
    factor : int
        Factor de reducción

    Returns
    -------
    np.ndarray
        Máscara reducida (H//factor, W//factor), valores {0, 255}
    """
    h, w = mask.shape[:2]
    new_w, new_h = w // factor, h // factor
    mask_small = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    # Forzar valores exactamente binarios después del resize
    return np.where(mask_small > 127, 255, 0).astype(np.uint8)


def load_and_downscale(
    image_path: Path,
    mask_path: Path,
    downscale_factor: int = 2,
    base_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pipeline completo: carga imagen + máscara y aplica downscale.

    Parameters
    ----------
    image_path : Path
    mask_path : Path
    downscale_factor : int
    base_path : Path, optional

    Returns
    -------
    image_ds : np.ndarray
        Imagen RGB downscaled (H_ds, W_ds, 3), uint8
    mask_ds : np.ndarray
        Máscara binaria downscaled (H_ds, W_ds), uint8, {0, 255}
    """
    # Cargar imagen original
    image = load_image_rgb(image_path)
    h_orig, w_orig = image.shape[:2]

    # Cargar máscara usando la función del proyecto existente
    mask = load_mask_binary(mask_path, h_orig, w_orig, base_path)

    # Downscale
    image_ds = downscale_image(image, factor=downscale_factor)
    mask_ds  = downscale_mask(mask,  factor=downscale_factor)

    return image_ds, mask_ds


def mask_to_float(mask: np.ndarray) -> np.ndarray:
    """
    Convierte máscara {0, 255} a float32 {0.0, 1.0}.
    Formato requerido por la loss function.
    """
    return (mask / 255.0).astype(np.float32)


def upscale_mask_to_original(
    mask_ds: np.ndarray,
    original_size: Tuple[int, int],
) -> np.ndarray:
    """
    Reescala una máscara downscaled de vuelta al tamaño original.

    IMPORTANTE: esto NO recupera detalle perdido durante el downscale.
    Solo sirve para visualización/comparación sobre la imagen original.

    Parameters
    ----------
    mask_ds : np.ndarray
        Máscara en resolución reducida (H_ds, W_ds)
    original_size : Tuple[int, int]
        (W_original, H_original) - tamaño destino

    Returns
    -------
    np.ndarray
        Máscara en tamaño original, valores {0, 255}
    """
    w_orig, h_orig = original_size
    mask_up = cv2.resize(
        mask_ds.astype(np.uint8),
        (w_orig, h_orig),
        interpolation=cv2.INTER_NEAREST,  # preservar clases binarias
    )
    return np.where(mask_up > 127, 255, 0).astype(np.uint8)
