"""
scripts/00_precompute_water_stats.py
=====================================
Script one-time: calcula la fracción de agua de cada imagen del dataset
y la guarda en un CSV cache.

Esto acelera el entrenamiento: en lugar de cargar la máscara en cada
__getitem__ solo para decidir si un patch tiene agua, se consulta el
cache precomputado.

Ejecutar UNA VEZ antes del entrenamiento:
    python scripts/00_precompute_water_stats.py

Runtime estimado: ~2-5 min para 15,000 imágenes (depende del disco).
"""

import sys
import logging
import time
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Setup
BASE_PATH = Path(r"D:\proyecto_eutrofizacion")
sys.path.insert(0, str(BASE_PATH / "src"))

from unet.config import UNetConfig
from unet.preprocessing import load_mask_binary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_stats_for_row(row, base_path: Path, config: UNetConfig) -> dict:
    """Calcula estadísticas de agua para una imagen."""
    try:
        mask_path = base_path / row["segmentation_mask_path"]
        img_path  = base_path / row["filepath"]

        # Necesitamos dimensiones de la imagen para rasterizar la máscara
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"filepath": row["filepath"], "water_fraction": -1.0, "error": "img_not_found"}

        h, w = img.shape[:2]
        mask = load_mask_binary(mask_path, h, w, base_path)

        # Fracción de agua en imagen completa (downscale para velocidad)
        scale = config.downscale_factor
        mask_small = cv2.resize(
            mask, (w // scale, h // scale), interpolation=cv2.INTER_NEAREST
        )

        water_fraction = float((mask_small > 127).mean())
        pixel_count    = int(mask_small.size)
        water_pixels   = int((mask_small > 127).sum())

        return {
            "filepath":       row["filepath"],
            "water_fraction": water_fraction,
            "water_pixels":   water_pixels,
            "total_pixels":   pixel_count,
            "error":          None,
        }

    except Exception as e:
        return {
            "filepath":       row["filepath"],
            "water_fraction": -1.0,
            "water_pixels":   0,
            "total_pixels":   0,
            "error":          str(e),
        }


def main():
    config = UNetConfig(base_path=BASE_PATH)
    config.setup_dirs()

    cache_path = config.get_path(config.water_stats_cache)

    if cache_path.exists():
        logger.info(f"Cache ya existe: {cache_path}")
        logger.info("Para regenerar, borra el archivo y vuelve a ejecutar.")
        df_cache = pd.read_csv(cache_path)
        logger.info(f"  {len(df_cache)} imágenes en cache")
        logger.info(f"  Fracción media de agua: {df_cache['water_fraction'].mean():.3f}")
        return

    # Cargar CSV completo
    csv_path = config.get_path(config.csv_path)
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["segmentation_mask_path"])
    logger.info(f"Procesando {len(df)} imágenes...")

    t0 = time.time()
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculando stats"):
        stats = compute_stats_for_row(row, BASE_PATH, config)
        results.append(stats)

    df_stats = pd.DataFrame(results)

    # Resumen
    n_ok  = (df_stats["water_fraction"] >= 0).sum()
    n_err = (df_stats["water_fraction"] < 0).sum()
    mean_water = df_stats.loc[df_stats["water_fraction"] >= 0, "water_fraction"].mean()

    logger.info(f"\nCompletado en {time.time() - t0:.1f}s")
    logger.info(f"  OK:     {n_ok} imágenes")
    logger.info(f"  Errores: {n_err}")
    logger.info(f"  Fracción media de agua: {mean_water:.3f} ({mean_water*100:.1f}%)")

    # Distribución
    w = df_stats.loc[df_stats["water_fraction"] >= 0, "water_fraction"]
    logger.info(f"  Sin agua (<5%): {(w < 0.05).sum()} imágenes")
    logger.info(f"  Poca agua (5-30%): {((w >= 0.05) & (w < 0.30)).sum()}")
    logger.info(f"  Mucha agua (>30%): {(w >= 0.30).sum()}")

    # Guardar cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(cache_path, index=False)
    logger.info(f"\nCache guardado: {cache_path}")


if __name__ == "__main__":
    main()
