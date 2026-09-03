"""
validator.py
============
Validación determinista con reconstrucción completa de la imagen.

FLUJO POR IMAGEN
----------------
1. Cargar imagen original (5280×3956)
2. Downscale ×2 → 2640×1978
3. Generar patches 512×512 con stride 256 (posiciones deterministas)
4. Pasar cada patch por la U-Net → probabilidades
5. Acumular probabilidades con overlap averaging
6. Reconstruir mapa completo 2640×1978
7. Comparar con máscara real completa
8. Calcular Dice, IoU, Precision, Recall por imagen y global

NO se usan patches aleatorios ni augmentation en validación.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
import logging

from .preprocessing import (
    load_image_rgb,
    load_mask_binary,
    downscale_image,
    downscale_mask,
    mask_to_float,
)
from .patches import get_patch_positions
from .reconstruction import reconstruct_full_prediction
from .metrics import compute_metrics, MetricAccumulator, SegMetrics
from .augmentation import get_val_transforms

logger = logging.getLogger(__name__)


class Validator:
    """
    Validación completa imagen por imagen con reconstrucción.

    Parameters
    ----------
    df_val : pd.DataFrame
        DataFrame del split de validación
    base_path : Path
    config : UNetConfig
    max_images : int, optional
        Limitar número de imágenes evaluadas (útil durante desarrollo)
    """

    def __init__(
        self,
        df_val: pd.DataFrame,
        base_path: Path,
        patch_size: int = 512,
        stride: int = 256,
        downscale_factor: int = 2,
        threshold: float = 0.5,
        batch_size: int = 16,
        imagenet_mean=(0.485, 0.456, 0.406),
        imagenet_std=(0.229, 0.224, 0.225),
        max_images: Optional[int] = None,
        viz_dir: Optional[Path] = None,
        num_viz_samples: int = 5,
    ):
        self.df              = df_val.reset_index(drop=True)
        self.base_path       = Path(base_path)
        self.patch_size      = patch_size
        self.stride          = stride
        self.downscale       = downscale_factor
        self.threshold       = threshold
        self.batch_size      = batch_size
        self.max_images      = max_images
        self.viz_dir         = Path(viz_dir) if viz_dir else None
        self.num_viz_samples = num_viz_samples

        self.transform = get_val_transforms(imagenet_mean, imagenet_std)

        if self.viz_dir:
            self.viz_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        model: torch.nn.Module,
        device: torch.device,
        threshold: Optional[float] = None,
        save_viz: bool = False,
    ) -> SegMetrics:
        """
        Evalúa el modelo sobre todo el conjunto de validación.

        Parameters
        ----------
        model : nn.Module
        device : torch.device
        threshold : float, optional
            Si None, usa self.threshold
        save_viz : bool
            Si True, guarda visualizaciones representativas

        Returns
        -------
        SegMetrics globales (micro-averaged sobre todas las imágenes)
        """
        if threshold is None:
            threshold = self.threshold

        accumulator = MetricAccumulator()
        n_images = len(self.df) if self.max_images is None else min(self.max_images, len(self.df))

        # Imágenes a guardar: las primeras num_viz_samples
        viz_indices = set(range(min(self.num_viz_samples, n_images))) if save_viz else set()

        model.eval()

        for i in range(n_images):
            row = self.df.iloc[i]
            img_path  = self.base_path / row["filepath"]
            mask_path = self.base_path / row["segmentation_mask_path"]

            try:
                # 1. Cargar y downscalear
                image = load_image_rgb(img_path)
                h, w  = image.shape[:2]
                mask  = load_mask_binary(mask_path, h, w, self.base_path)

                image_ds = downscale_image(image, self.downscale)
                mask_ds  = downscale_mask(mask,   self.downscale)
                h_ds, w_ds = image_ds.shape[:2]

                # 2. Generar posiciones deterministas
                positions = get_patch_positions(h_ds, w_ds, self.patch_size, self.stride)

                # 3. Reconstruir predicción completa
                prob_avg, mask_pred = reconstruct_full_prediction(
                    model=model,
                    image_ds=image_ds,
                    positions=positions,
                    transform=self.transform,
                    patch_size=self.patch_size,
                    threshold=threshold,
                    batch_size=self.batch_size,
                    device=device,
                )

                # 4. Calcular métricas sobre la imagen completa
                true_mask_float = (mask_ds > 127).astype(np.float32)
                metrics_img = compute_metrics(prob_avg, true_mask_float, threshold=threshold)
                accumulator.update(metrics_img)

                if i % 10 == 0:
                    logger.info(
                        f"  Val [{i+1}/{n_images}] "
                        f"{Path(row['filepath']).name}: {metrics_img}"
                    )

                # 5. Guardar visualizaciones
                if i in viz_indices and self.viz_dir:
                    self._save_visualization(
                        image_ds, mask_ds, prob_avg, mask_pred,
                        metrics_img, Path(row["filepath"]).stem, i
                    )

            except Exception as e:
                logger.warning(f"Error evaluando {img_path}: {e}")
                continue

        global_metrics = accumulator.compute_global()
        mean_metrics   = accumulator.compute_mean()

        logger.info(f"\nValidación completa ({n_images} imágenes):")
        logger.info(f"  Global (micro): {global_metrics}")
        logger.info(f"  Media (macro):  {mean_metrics}")

        return global_metrics

    def evaluate_thresholds(
        self,
        model: torch.nn.Module,
        device: torch.device,
        candidates: List[float] = None,
        n_images: int = 50,
    ) -> dict:
        """
        Evalúa múltiples thresholds sobre un subconjunto de validación.

        IMPORTANTE: Solo usar sobre VALIDATION, nunca sobre TEST.

        Returns
        -------
        dict con 'best_threshold', 'best_dice', 'results_per_threshold'
        """
        if candidates is None:
            candidates = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

        n_eval = min(n_images, len(self.df))
        results = {t: MetricAccumulator() for t in candidates}

        model.eval()
        logger.info(f"Buscando mejor threshold sobre {n_eval} imágenes de validación...")

        for i in range(n_eval):
            row = self.df.iloc[i]
            img_path  = self.base_path / row["filepath"]
            mask_path = self.base_path / row["segmentation_mask_path"]

            try:
                image = load_image_rgb(img_path)
                h, w  = image.shape[:2]
                mask  = load_mask_binary(mask_path, h, w, self.base_path)

                image_ds = downscale_image(image, self.downscale)
                mask_ds  = downscale_mask(mask,   self.downscale)
                h_ds, w_ds = image_ds.shape[:2]

                positions = get_patch_positions(h_ds, w_ds, self.patch_size, self.stride)

                # Reconstruir probabilidades (sin threshold aún)
                prob_avg, _ = reconstruct_full_prediction(
                    model=model,
                    image_ds=image_ds,
                    positions=positions,
                    transform=self.transform,
                    patch_size=self.patch_size,
                    threshold=0.5,  # threshold dummy, usamos prob_avg directamente
                    batch_size=self.batch_size,
                    device=device,
                )

                true_mask = (mask_ds > 127).astype(np.float32)

                # Evaluar cada threshold sobre el mismo prob_avg
                for t in candidates:
                    m = compute_metrics(prob_avg, true_mask, threshold=t)
                    results[t].update(m)

            except Exception as e:
                logger.warning(f"Error: {e}")
                continue

        # Comparar thresholds
        threshold_scores = {
            t: acc.compute_global().dice
            for t, acc in results.items()
        }

        best_t = max(threshold_scores, key=threshold_scores.get)

        logger.info("Resultados por threshold:")
        for t, score in sorted(threshold_scores.items()):
            marker = " ← MEJOR" if t == best_t else ""
            logger.info(f"  threshold={t:.2f} → Dice={score:.4f}{marker}")

        return {
            "best_threshold": best_t,
            "best_dice": threshold_scores[best_t],
            "results_per_threshold": threshold_scores,
        }

    def _save_visualization(
        self, image_ds, mask_ds, prob_avg, mask_pred, metrics, stem, idx
    ):
        """Guarda una visualización de la predicción vs ground truth."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 4, figsize=(24, 6))

            axes[0].imshow(image_ds)
            axes[0].set_title("Imagen downscaled")
            axes[0].axis("off")

            axes[1].imshow(mask_ds, cmap="gray")
            axes[1].set_title("Máscara real")
            axes[1].axis("off")

            axes[2].imshow(prob_avg, cmap="hot", vmin=0, vmax=1)
            axes[2].set_title("Probabilidades U-Net")
            axes[2].axis("off")

            axes[3].imshow(mask_pred, cmap="gray")
            axes[3].set_title(
                f"Predicción binarizada\nDice={metrics.dice:.3f} IoU={metrics.iou:.3f}"
            )
            axes[3].axis("off")

            plt.tight_layout()
            out_path = self.viz_dir / f"val_{idx:03d}_{stem}.png"
            plt.savefig(out_path, dpi=100, bbox_inches="tight")
            plt.close()

        except Exception as e:
            logger.warning(f"Error guardando visualización: {e}")
