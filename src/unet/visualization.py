"""
visualization.py
================
Funciones de visualización para:
  - Imagen original vs downscaled
  - Patches seleccionados
  - Predicción vs ground truth
  - Overlay de máscara sobre imagen
  - Análisis de errores (FP, FN, bordes)
"""

import sys
import numpy as np
import matplotlib

# Solo forzamos el backend "Agg" (sin pantalla) fuera de Jupyter. Así
# train_unet.py puede seguir guardando figuras aunque no haya display, pero
# un notebook (que corre dentro de ipykernel) conserva su backend
# interactivo/inline normal y plt.show() funciona como se espera.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, Tuple
import cv2
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FUNCIONES BASE
# ---------------------------------------------------------------------------

def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 100, 255),
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Superpone una máscara binaria sobre una imagen RGB con transparencia.

    Parameters
    ----------
    image : np.ndarray (H, W, 3) RGB uint8
    mask : np.ndarray (H, W) valores {0, 255} o {0, 1}
    color : Tuple[int, int, int]  RGB del color del overlay
    alpha : float  opacidad del overlay [0, 1]

    Returns
    -------
    np.ndarray (H, W, 3) imagen con overlay
    """
    result = image.copy()
    if mask.max() <= 1:
        mask_bin = (mask > 0.5).astype(np.uint8) * 255
    else:
        mask_bin = (mask > 127).astype(np.uint8) * 255

    overlay = np.zeros_like(image)
    overlay[mask_bin > 0] = color

    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
    return result


def colorize_error_map(
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> np.ndarray:
    """
    Genera un mapa de colores de errores:
      Verde (0, 200, 0): Verdadero Positivo (agua detectada correctamente)
      Rojo (255, 0, 0): Falso Positivo (tierra detectada como agua)
      Azul (0, 0, 255): Falso Negativo (agua no detectada)
      Gris (40, 40, 40): Verdadero Negativo (tierra correctamente ignorada)

    Parameters
    ----------
    true_mask, pred_mask : np.ndarray (H, W)
        Máscaras binarias {0, 1} o {0, 255}

    Returns
    -------
    np.ndarray (H, W, 3) RGB uint8
    """
    if true_mask.max() > 1:
        true_bin = (true_mask > 127).astype(np.uint8)
    else:
        true_bin = (true_mask > 0.5).astype(np.uint8)

    if pred_mask.max() > 1:
        pred_bin = (pred_mask > 127).astype(np.uint8)
    else:
        pred_bin = (pred_mask > 0.5).astype(np.uint8)

    h, w   = true_bin.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    tp_mask = (true_bin == 1) & (pred_bin == 1)
    fp_mask = (true_bin == 0) & (pred_bin == 1)
    fn_mask = (true_bin == 1) & (pred_bin == 0)
    tn_mask = (true_bin == 0) & (pred_bin == 0)

    result[tp_mask] = [0, 200, 0]    # Verde: TP
    result[fp_mask] = [255, 50, 50]  # Rojo: FP
    result[fn_mask] = [50, 50, 255]  # Azul: FN
    result[tn_mask] = [40, 40, 40]   # Gris oscuro: TN

    return result


# ---------------------------------------------------------------------------
# VISUALIZACIONES PRINCIPALES
# ---------------------------------------------------------------------------

def plot_image_vs_downscaled(
    image_original: np.ndarray,
    image_ds: np.ndarray,
    save_path: Optional[Path] = None,
):
    """Muestra la imagen original vs la versión downscaled."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    h_o, w_o = image_original.shape[:2]
    h_d, w_d = image_ds.shape[:2]

    axes[0].imshow(image_original)
    axes[0].set_title(f"Original: {w_o}×{h_o} px")
    axes[0].axis("off")

    axes[1].imshow(image_ds)
    axes[1].set_title(f"Downscaled (×2): {w_d}×{h_d} px")
    axes[1].axis("off")

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_patch_sample(
    image_ds: np.ndarray,
    mask_ds: np.ndarray,
    y: int,
    x: int,
    patch_size: int = 512,
    save_path: Optional[Path] = None,
):
    """
    Muestra un patch seleccionado dentro de la imagen completa.
    Dibuja un rectángulo en la imagen completa indicando la posición del patch.
    """
    img_patch  = image_ds[y:y + patch_size, x:x + patch_size]
    mask_patch = mask_ds[y:y + patch_size, x:x + patch_size]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Imagen completa con rectángulo
    img_with_rect = image_ds.copy()
    cv2.rectangle(img_with_rect, (x, y), (x + patch_size, y + patch_size), (255, 0, 0), 6)
    axes[0].imshow(img_with_rect)
    axes[0].set_title(f"Imagen completa\n(rojo = patch seleccionado en y={y}, x={x})")
    axes[0].axis("off")

    axes[1].imshow(img_patch)
    axes[1].set_title(f"Patch imagen {patch_size}×{patch_size}")
    axes[1].axis("off")

    axes[2].imshow(mask_patch, cmap="gray")
    water_pct = (mask_patch > 127).mean() * 100
    axes[2].set_title(f"Patch máscara\n({water_pct:.1f}% agua)")
    axes[2].axis("off")

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_prediction(
    image_ds: np.ndarray,
    true_mask: np.ndarray,
    prob_avg: np.ndarray,
    pred_mask: np.ndarray,
    metrics=None,
    title: str = "",
    save_path: Optional[Path] = None,
):
    """
    Visualización completa de una predicción:
      Col 1: Imagen downscaled
      Col 2: Máscara real (ground truth)
      Col 3: Mapa de probabilidades de la U-Net
      Col 4: Predicción binarizada
      Col 5: Mapa de errores (TP/FP/FN/TN)
    """
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))

    axes[0].imshow(image_ds)
    axes[0].set_title("Imagen downscaled")
    axes[0].axis("off")

    axes[1].imshow(true_mask, cmap="gray")
    axes[1].set_title("Máscara real (GT)")
    axes[1].axis("off")

    im = axes[2].imshow(prob_avg, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Probabilidades U-Net")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    axes[3].imshow(pred_mask, cmap="gray")
    if metrics:
        axes[3].set_title(
            f"Predicción binarizada\n"
            f"Dice={metrics.dice:.3f} | IoU={metrics.iou:.3f}\n"
            f"Prec={metrics.precision:.3f} | Rec={metrics.recall:.3f}"
        )
    else:
        axes[3].set_title("Predicción binarizada")
    axes[3].axis("off")

    # Mapa de errores
    error_map = colorize_error_map(true_mask, pred_mask)
    axes[4].imshow(error_map)
    axes[4].set_title("Mapa de errores\nVerde=TP | Rojo=FP | Azul=FN")
    axes[4].axis("off")

    # Leyenda
    legend_patches = [
        mpatches.Patch(color=[0, 200/255, 0],   label="TP (agua correcta)"),
        mpatches.Patch(color=[255/255, 50/255, 50/255], label="FP (tierra→agua)"),
        mpatches.Patch(color=[50/255, 50/255, 255/255], label="FN (agua→tierra)"),
    ]
    axes[4].legend(handles=legend_patches, loc="lower right", fontsize=7)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_overlay(
    image_ds: np.ndarray,
    pred_mask: np.ndarray,
    true_mask: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
):
    """
    Overlay de la predicción sobre la imagen.
    Si se provee true_mask, muestra también el overlay del GT para comparar.
    """
    n_cols = 3 if true_mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 8, 6))

    axes[0].imshow(image_ds)
    axes[0].set_title("Imagen original (downscaled)")
    axes[0].axis("off")

    pred_overlay = overlay_mask(image_ds, pred_mask, color=(0, 100, 255), alpha=0.45)
    axes[1].imshow(pred_overlay)
    axes[1].set_title("Predicción U-Net (azul = agua)")
    axes[1].axis("off")

    if true_mask is not None:
        gt_overlay = overlay_mask(image_ds, true_mask, color=(0, 200, 0), alpha=0.45)
        axes[2].imshow(gt_overlay)
        axes[2].set_title("Ground Truth (verde = agua)")
        axes[2].axis("off")

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_error_analysis(
    image_ds: np.ndarray,
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
    title: str = "",
    save_path: Optional[Path] = None,
):
    """
    Análisis detallado de errores:
      - Falsos Positivos: tierra clasificada como agua
        (puede indicar reflejos, sombras, o zonas húmedas)
      - Falsos Negativos: agua no detectada
        (puede indicar zonas oscuras, turbias o de baja saturación)
      - Errores en bordes (dilación del borde real ∩ errores)
    """
    if true_mask.max() > 1:
        true_bin = (true_mask > 127).astype(np.uint8)
    else:
        true_bin = (true_mask > 0.5).astype(np.uint8)

    if pred_mask.max() > 1:
        pred_bin = (pred_mask > 127).astype(np.uint8)
    else:
        pred_bin = (pred_mask > 0.5).astype(np.uint8)

    fp_mask = ((pred_bin == 1) & (true_bin == 0)).astype(np.uint8) * 255
    fn_mask = ((pred_bin == 0) & (true_bin == 1)).astype(np.uint8) * 255

    # Detectar errores en bordes: dilatar borde real y ver si FP/FN están ahí
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    border_true = cv2.dilate(true_bin.astype(np.uint8), kernel) - cv2.erode(true_bin.astype(np.uint8), kernel)
    border_fp = (fp_mask > 0) & (border_true > 0)
    border_fn = (fn_mask > 0) & (border_true > 0)

    n_fp = fp_mask.sum() // 255
    n_fn = fn_mask.sum() // 255
    n_border_fp = border_fp.sum()
    n_border_fn = border_fn.sum()

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    axes[0, 0].imshow(image_ds)
    axes[0, 0].set_title("Imagen")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(overlay_mask(image_ds, fp_mask, color=(255, 50, 50), alpha=0.6))
    axes[0, 1].set_title(f"Falsos Positivos (tierra→agua)\n{n_fp:,} píxeles")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(overlay_mask(image_ds, fn_mask, color=(50, 50, 255), alpha=0.6))
    axes[0, 2].set_title(f"Falsos Negativos (agua→tierra)\n{n_fn:,} píxeles")
    axes[0, 2].axis("off")

    error_map = colorize_error_map(true_mask, pred_mask)
    axes[1, 0].imshow(error_map)
    axes[1, 0].set_title("Mapa de errores completo")
    axes[1, 0].axis("off")

    # FP en bordes
    fp_border_overlay = overlay_mask(image_ds, border_fp.astype(np.uint8) * 255, color=(255, 165, 0), alpha=0.7)
    axes[1, 1].imshow(fp_border_overlay)
    axes[1, 1].set_title(f"FP en zona de borde\n{n_border_fp:,} píxeles\n(posible reflejo/sombra)")
    axes[1, 1].axis("off")

    # FN en bordes
    fn_border_overlay = overlay_mask(image_ds, border_fn.astype(np.uint8) * 255, color=(180, 0, 255), alpha=0.7)
    axes[1, 2].imshow(fn_border_overlay)
    axes[1, 2].set_title(f"FN en zona de borde\n{n_border_fn:,} píxeles\n(agua perdida en borde)")
    axes[1, 2].axis("off")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_training_history(history: dict, save_path: Optional[Path] = None):
    """Visualiza el historial de entrenamiento (loss, métricas, LR)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    axes[0].set_title("Loss de entrenamiento")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_dice"],      "g-o", markersize=3, label="Dice")
    axes[1].plot(epochs, history["val_iou"],       "b-s", markersize=3, label="IoU")
    axes[1].plot(epochs, history["val_precision"], "r-^", markersize=3, label="Precision")
    axes[1].plot(epochs, history["val_recall"],    "m-v", markersize=3, label="Recall")
    axes[1].set_title("Métricas de validación")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    axes[2].semilogy(epochs, history["lr"], "k-o", markersize=3)
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Época")
    axes[2].set_ylabel("LR")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------

def _save_or_show(fig, save_path: Optional[Path]):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
