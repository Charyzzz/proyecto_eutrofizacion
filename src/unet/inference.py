"""
inference.py
============
Inferencia completa sobre imágenes nuevas.

PIPELINE
--------
5280×3956
  → downscale ×2
  → 2640×1978
  → patches 512×512
  → U-Net
  → reconstrucción con overlap averaging
  → máscara 2640×1978

Opcionalmente:
  → upscale máscara a 5280×3956 (SOLO para visualización)

NOTA IMPORTANTE SOBRE EL UPSCALE
---------------------------------
Al devolver la máscara a 5280×3956, NO se recupera ningún detalle
perdido durante el downscale. La resolución de la segmentación real
es 2640×1978. El upscale es únicamente para facilitar la comparación
visual sobre la imagen original.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

from .preprocessing import (
    load_image_rgb,
    downscale_image,
    upscale_mask_to_original,
)
from .patches import get_patch_positions
from .reconstruction import reconstruct_full_prediction
from .augmentation import get_val_transforms

logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: Path,
    encoder_name: str = "resnet34",
    encoder_weights: str = None,  # None al cargar desde checkpoint
    device: torch.device = None,
) -> torch.nn.Module:
    """
    Carga el modelo U-Net desde un checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
    encoder_name : str
    encoder_weights : str
        None para no cargar pesos de ImageNet (se cargarán del checkpoint)
    device : torch.device

    Returns
    -------
    nn.Module en modo eval
    """
    from .model import build_unet

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        num_classes=1,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    dice  = ckpt.get("val_dice", "?")
    logger.info(
        f"Modelo cargado: {checkpoint_path.name} "
        f"(epoch={epoch}, val_dice={dice})"
    )

    return model


def run_inference(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    patch_size: int = 512,
    stride: int = 256,
    downscale_factor: int = 2,
    threshold: float = 0.5,
    batch_size: int = 16,
    return_upscaled: bool = True,
    imagenet_mean=(0.485, 0.456, 0.406),
    imagenet_std=(0.229, 0.224, 0.225),
) -> dict:
    """
    Inferencia completa sobre una imagen.

    Parameters
    ----------
    image_path : Path
        Imagen original (5280×3956)
    model : nn.Module
        Modelo U-Net cargado
    device : torch.device
    patch_size, stride, downscale_factor, threshold, batch_size : config
    return_upscaled : bool
        Si True, incluye la máscara upscalada a la resolución original
    imagenet_mean, imagenet_std : tuple
        Para normalización

    Returns
    -------
    dict con:
        'image_ds': imagen downscaled RGB (H_ds, W_ds, 3)
        'prob_avg': probabilidades (H_ds, W_ds) float32
        'mask_ds': máscara binaria (H_ds, W_ds) uint8 {0, 255}
        'mask_original': [opcional] máscara upscaled (H_orig, W_orig) uint8
        'image_size_original': (W, H) de la imagen original
        'image_size_downscaled': (W, H) tras downscale
    """
    image_path = Path(image_path)
    logger.info(f"Inferencia sobre: {image_path.name}")

    # 1. Cargar imagen
    image = load_image_rgb(image_path)
    h_orig, w_orig = image.shape[:2]

    # 2. Downscale
    image_ds = downscale_image(image, downscale_factor)
    h_ds, w_ds = image_ds.shape[:2]
    logger.info(f"  Original: {w_orig}×{h_orig} → Downscaled: {w_ds}×{h_ds}")

    # 3. Generar posiciones de patches
    positions = get_patch_positions(h_ds, w_ds, patch_size, stride)
    logger.info(f"  Patches a procesar: {len(positions)}")

    # 4. Transform (sin augmentation)
    transform = get_val_transforms(imagenet_mean, imagenet_std)

    # 5. Reconstrucción completa
    prob_avg, mask_ds = reconstruct_full_prediction(
        model=model,
        image_ds=image_ds,
        positions=positions,
        transform=transform,
        patch_size=patch_size,
        threshold=threshold,
        batch_size=batch_size,
        device=device,
    )

    result = {
        "image_ds":            image_ds,
        "prob_avg":            prob_avg,
        "mask_ds":             mask_ds,
        "image_size_original":   (w_orig, h_orig),
        "image_size_downscaled": (w_ds,   h_ds),
    }

    # 6. Opcional: upscale máscara a resolución original
    if return_upscaled:
        mask_orig = upscale_mask_to_original(mask_ds, (w_orig, h_orig))
        result["mask_original"] = mask_orig
        logger.info(
            f"  ⚠ mask_original es SOLO para visualización. "
            f"No recupera detalle perdido en el downscale."
        )

    water_pct = (mask_ds > 127).mean() * 100
    logger.info(f"  Agua detectada: {water_pct:.1f}% de la imagen")

    return result


def batch_inference(
    image_paths: list,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    **inference_kwargs,
) -> list:
    """
    Inferencia sobre múltiples imágenes con guardado automático.

    Parameters
    ----------
    image_paths : list of Path
    model : nn.Module
    device : torch.device
    output_dir : Path
        Directorio donde guardar las máscaras predichas
    **inference_kwargs
        Parámetros para run_inference

    Returns
    -------
    list of dict con resultados por imagen
    """
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    n = len(image_paths)

    for i, img_path in enumerate(image_paths):
        logger.info(f"\n[{i+1}/{n}] {Path(img_path).name}")
        try:
            res = run_inference(img_path, model, device, **inference_kwargs)

            # Guardar máscara downscaled
            stem = Path(img_path).stem
            mask_path = output_dir / f"{stem}_mask_ds.png"
            cv2.imwrite(str(mask_path), res["mask_ds"])

            if "mask_original" in res:
                mask_orig_path = output_dir / f"{stem}_mask_original.png"
                cv2.imwrite(str(mask_orig_path), res["mask_original"])

            results.append({"path": img_path, "result": res, "error": None})

        except Exception as e:
            logger.error(f"Error en {img_path}: {e}")
            results.append({"path": img_path, "result": None, "error": str(e)})

    n_ok  = sum(1 for r in results if r["error"] is None)
    n_err = sum(1 for r in results if r["error"] is not None)
    logger.info(f"\nBatch completado: {n_ok} OK | {n_err} errores")

    return results
