"""
reconstruction.py
=================
Reconstrucción de la predicción completa (2640×1978) a partir de
los patches solapados producidos por la U-Net.

CONCEPTO
--------
Con 50% de overlap, cada píxel en el interior de la imagen es visto
por múltiples patches. Por ejemplo, un píxel en el centro puede
aparecer en 4 patches diferentes.

Estrategia de fusión: promedio de probabilidades.

    prediction_final(x, y) = sum(pred_patches(x, y)) / count(x, y)

donde count(x, y) es el número de patches que contienen al píxel (x, y).

Después del promedio se aplica el threshold para binarizar.

VENTAJAS DEL PROMEDIO
---------------------
1. Elimina artefactos en las costuras entre patches.
2. Aprovecha múltiples "opiniones" del modelo sobre el mismo píxel.
3. Los píxeles en el centro del patch (donde el modelo es más seguro)
   se promedian con predicciones del borde de otros patches (menos seguros),
   pero el resultado sigue siendo mejor que cualquier predicción individual.

IMPLEMENTACIÓN
--------------
Se usan dos arrays del tamaño de la imagen completa:
  - prob_sum[H, W]: acumula la suma de probabilidades de cada predicción
  - count[H, W]:    acumula cuántas veces fue predicho cada píxel

Al final:
  prob_avg = prob_sum / count   (donde count > 0)
  mask_bin = prob_avg > threshold
"""

import numpy as np
from typing import List, Tuple
import torch


def create_reconstruction_buffers(
    image_h: int,
    image_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inicializa los buffers de acumulación para la reconstrucción.

    Parameters
    ----------
    image_h, image_w : int
        Dimensiones de la imagen completa (tras downscale)

    Returns
    -------
    prob_sum : np.ndarray (H, W) float32
        Buffer para acumular probabilidades
    count : np.ndarray (H, W) float32
        Buffer para contar predicciones por píxel
    """
    prob_sum = np.zeros((image_h, image_w), dtype=np.float32)
    count    = np.zeros((image_h, image_w), dtype=np.float32)
    return prob_sum, count


def accumulate_patch_prediction(
    prob_sum: np.ndarray,
    count: np.ndarray,
    patch_proba: np.ndarray,
    y_start: int,
    x_start: int,
    patch_size: int = 512,
):
    """
    Acumula la predicción de un patch en los buffers globales.

    Parameters
    ----------
    prob_sum, count : np.ndarray
        Buffers de acumulación (modificados in-place)
    patch_proba : np.ndarray
        Probabilidades del patch (patch_size, patch_size), valores [0,1]
    y_start, x_start : int
        Posición del patch en la imagen completa
    patch_size : int
    """
    y_end = y_start + patch_size
    x_end = x_start + patch_size

    # Asegurar que el patch no se salga de la imagen
    actual_h = min(y_end, prob_sum.shape[0]) - y_start
    actual_w = min(x_end, prob_sum.shape[1]) - x_start

    prob_sum[y_start:y_start + actual_h, x_start:x_start + actual_w] += \
        patch_proba[:actual_h, :actual_w]

    count[y_start:y_start + actual_h, x_start:x_start + actual_w] += 1.0


def finalize_reconstruction(
    prob_sum: np.ndarray,
    count: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finaliza la reconstrucción calculando el promedio y aplicando el threshold.

    Parameters
    ----------
    prob_sum, count : np.ndarray
        Buffers acumulados
    threshold : float
        Umbral de binarización

    Returns
    -------
    prob_avg : np.ndarray (H, W) float32
        Mapa de probabilidades promediadas [0, 1]
    mask_bin : np.ndarray (H, W) uint8
        Máscara binaria {0, 255}
        255 = agua, 0 = no-agua
    """
    # Evitar división por cero (píxeles no cubiertos, si los hubiera)
    safe_count = np.where(count > 0, count, 1.0)
    prob_avg   = prob_sum / safe_count
    prob_avg   = np.clip(prob_avg, 0.0, 1.0)

    # Binarizar
    mask_bin = np.where(prob_avg >= threshold, 255, 0).astype(np.uint8)

    return prob_avg, mask_bin


def reconstruct_full_prediction(
    model,
    image_ds: np.ndarray,
    positions: List[Tuple[int, int]],
    transform,
    patch_size: int = 512,
    threshold: float = 0.5,
    batch_size: int = 16,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pipeline completo de reconstrucción para una imagen.

    1. Itera sobre todos los patches de la imagen en orden determinista.
    2. Pasa cada patch (en batches) por la U-Net.
    3. Acumula probabilidades con el buffer de suma.
    4. Calcula el promedio y binariza.

    Parameters
    ----------
    model : nn.Module
        Modelo U-Net en modo eval
    image_ds : np.ndarray
        Imagen RGB downscaled (H, W, 3)
    positions : List[Tuple[int, int]]
        Lista de (y, x) de cada patch (determinista)
    transform : A.Compose
        Transform de validación (sin augmentation)
    patch_size : int
    threshold : float
    batch_size : int
        Patches a procesar en paralelo en GPU
    device : torch.device

    Returns
    -------
    prob_avg : np.ndarray (H, W) float32
    mask_bin : np.ndarray (H, W) uint8
    """
    import torch
    from .preprocessing import mask_to_float
    from .augmentation import apply_transform_pair
    import numpy as np

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h_ds, w_ds = image_ds.shape[:2]
    prob_sum, count = create_reconstruction_buffers(h_ds, w_ds)

    # Procesar patches en mini-batches para eficiencia en GPU
    model.eval()
    n = len(positions)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_positions = positions[start:start + batch_size]
            batch_tensors   = []

            for (y, x) in batch_positions:
                patch = image_ds[y:y + patch_size, x:x + patch_size]
                # Dummy mask para el transform (no se usa en inferencia)
                dummy_mask = np.zeros((patch_size, patch_size), dtype=np.float32)
                img_tensor, _ = apply_transform_pair(transform, patch, dummy_mask)
                batch_tensors.append(img_tensor)

            # Stack en batch
            batch = torch.stack(batch_tensors, dim=0).to(device)

            # Inferencia
            logits = model(batch)
            probas = torch.sigmoid(logits).cpu().numpy()  # (B, 1, H, W)

            # Acumular cada patch
            for i, (y, x) in enumerate(batch_positions):
                patch_proba = probas[i, 0]  # (H, W)
                accumulate_patch_prediction(prob_sum, count, patch_proba, y, x, patch_size)

    prob_avg, mask_bin = finalize_reconstruction(prob_sum, count, threshold)
    return prob_avg, mask_bin
