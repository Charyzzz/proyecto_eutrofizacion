"""
dataset.py
==========
Datasets PyTorch para train, val y test.

ESTRATEGIA GENERAL
------------------

TRAINING (WaterTrainDataset):
  - Generación de patches on-the-fly: no se almacenan patches en disco.
  - len() = len(train_images) × patches_per_image
  - Para cada __getitem__:
      1. Determina qué imagen corresponde (idx // patches_per_image)
      2. Carga imagen + máscara (con cache LRU si hay RAM disponible)
      3. Downscale ×2
      4. Calcula fracciones de agua por patch (una vez por imagen)
      5. Samplea un patch con probabilidad ponderada (más agua y bordes)
      6. Aplica augmentation
  - Cada época puede samplear patches diferentes de cada imagen.

VALIDATION / TEST (WaterEvalDataset):
  - COMPLETAMENTE DETERMINISTA.
  - Para cada imagen: todos los patches en orden fijo (sliding window).
  - __getitem__ devuelve patches individuales junto con sus metadatos.
  - La reconstrucción completa se hace en validator.py.
  - Sin augmentation, sin aleatoriedad.

POR QUÉ NO SE GUARDAN LOS PATCHES EN DISCO
-------------------------------------------
Con 15,000 imágenes × 70 patches = 1,050,000 patches de 512×512×3.
Almacenarlos sería ~1,050,000 × 786KB ≈ 825 GB. Inviable.
La generación on-the-fly a la velocidad de la GPU es perfectamente
factible con num_workers > 0.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict
import torch
from torch.utils.data import Dataset
import albumentations as A
import logging

from .preprocessing import (
    load_image_rgb,
    load_mask_binary,
    downscale_image,
    downscale_mask,
    mask_to_float,
)
from .patches import (
    get_patch_positions,
    extract_patch,
    compute_water_fractions,
    classify_patches,
    sample_patch_indices,
)
from .augmentation import apply_transform_pair

logger = logging.getLogger(__name__)


class WaterTrainDataset(Dataset):
    """
    Dataset de entrenamiento con generación on-the-fly y sampling ponderado.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'filepath', 'segmentation_mask_path'
    base_path : Path
        Ruta base del proyecto (D:\\proyecto_eutrofizacion)
    transform : A.Compose
        Pipeline de augmentation para training
    config : UNetConfig
        Configuración del pipeline
    """

    def __init__(
        self,
        df: pd.DataFrame,
        base_path: Path,
        transform: A.Compose,
        patch_size: int = 512,
        stride: int = 256,
        downscale_factor: int = 2,
        patches_per_image: int = 8,
        water_ratio: float = 0.50,
        boundary_ratio: float = 0.35,
        no_water_ratio: float = 0.15,
        min_water_fraction: float = 0.05,
        boundary_max_water: float = 0.70,
        seed: int = 42,
    ):
        self.df               = df.reset_index(drop=True)
        self.base_path        = Path(base_path)
        self.transform        = transform
        self.patch_size       = patch_size
        self.stride           = stride
        self.downscale_factor = downscale_factor
        self.patches_per_img  = patches_per_image
        self.water_ratio      = water_ratio
        self.boundary_ratio   = boundary_ratio
        self.no_water_ratio   = no_water_ratio
        self.min_water        = min_water_fraction
        self.boundary_max     = boundary_max_water
        self.seed             = seed

        # Cache de posiciones y fracciones de agua (se llena bajo demanda)
        # Evita recalcular posiciones cada vez
        self._pos_cache: Dict[int, list] = {}
        self._frac_cache: Dict[int, np.ndarray] = {}

        # RNG base por worker (se reinicializa en worker_init_fn)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df) * self.patches_per_img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Determinar qué imagen usar
        img_idx = idx // self.patches_per_img

        row = self.df.iloc[img_idx]
        img_path  = self.base_path / row["filepath"]
        mask_path = self.base_path / row["segmentation_mask_path"]

        # Cargar y downscalear
        try:
            image = load_image_rgb(img_path)
            h, w  = image.shape[:2]
            mask  = load_mask_binary(mask_path, h, w, self.base_path)
        except Exception as e:
            logger.warning(f"Error cargando imagen {img_path}: {e}. Usando dummy.")
            return self._dummy_sample()

        image_ds = downscale_image(image, self.downscale_factor)
        mask_ds  = downscale_mask(mask,   self.downscale_factor)
        h_ds, w_ds = image_ds.shape[:2]

        # Obtener posiciones (con cache)
        if img_idx not in self._pos_cache:
            positions = get_patch_positions(h_ds, w_ds, self.patch_size, self.stride)
            self._pos_cache[img_idx]  = positions
            fractions = compute_water_fractions(mask_ds, positions, self.patch_size)
            self._frac_cache[img_idx] = fractions
        else:
            positions = self._pos_cache[img_idx]
            fractions = self._frac_cache[img_idx]

        # Clasificar patches por contenido de agua
        water_idx, boundary_idx, no_water_idx = classify_patches(
            fractions, self.min_water, self.boundary_max
        )

        # Samplear UN patch con la estrategia ponderada
        sampled = sample_patch_indices(
            water_idx, boundary_idx, no_water_idx,
            n_patches=1,
            water_ratio=self.water_ratio,
            boundary_ratio=self.boundary_ratio,
            no_water_ratio=self.no_water_ratio,
            rng=self._rng,
        )

        if len(sampled) == 0:
            # Fallback: patch aleatorio si el sampling falla
            sampled = [self._rng.integers(0, len(positions))]

        patch_pos_idx = int(sampled[0])
        y, x = positions[patch_pos_idx]

        # Extraer patch
        img_patch  = extract_patch(image_ds, y, x, self.patch_size)
        mask_patch = extract_patch(mask_ds,  y, x, self.patch_size)

        # Máscara a float [0, 1]
        mask_float = mask_to_float(mask_patch)

        # Augmentation
        img_tensor, mask_tensor = apply_transform_pair(
            self.transform, img_patch, mask_float
        )

        # Añadir dimensión de canal a la máscara: (H, W) → (1, H, W)
        return img_tensor, mask_tensor.unsqueeze(0)

    def _dummy_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample de relleno cuando hay error al cargar una imagen."""
        img  = torch.zeros(3, self.patch_size, self.patch_size)
        mask = torch.zeros(1, self.patch_size, self.patch_size)
        return img, mask


def worker_init_fn(worker_id: int):
    """
    Inicializa el RNG de cada worker con una semilla única.
    Necesario para que workers diferentes generen patches diferentes
    (sin esto todos los workers usarían la misma semilla y generarían
    los mismos patches en paralelo).
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


class WaterEvalDataset(Dataset):
    """
    Dataset determinista para validation y test.

    Para cada imagen, genera todos los patches del sliding window en orden fijo.
    NO usa augmentation ni aleatoriedad.

    __getitem__ devuelve:
      - patch de imagen normalizado (tensor)
      - patch de máscara (tensor)
      - metadatos del patch (imagen_idx, y, x) para reconstrucción

    La reconstrucción completa de la predicción se hace en validator.py.

    Parameters
    ----------
    df : pd.DataFrame
    base_path : Path
    transform : A.Compose
        Transform de validación (solo normalización)
    patch_size, stride, downscale_factor : int
    """

    def __init__(
        self,
        df: pd.DataFrame,
        base_path: Path,
        transform: A.Compose,
        patch_size: int = 512,
        stride: int = 256,
        downscale_factor: int = 2,
    ):
        self.df               = df.reset_index(drop=True)
        self.base_path        = Path(base_path)
        self.transform        = transform
        self.patch_size       = patch_size
        self.stride           = stride
        self.downscale_factor = downscale_factor

        # Construir índice plano: (image_idx, y, x) para cada patch
        # Esto es DETERMINISTA y reproducible
        logger.info("Construyendo índice de patches de evaluación...")
        self._build_patch_index()
        logger.info(f"  Total patches de eval: {len(self._index)}")

    def _build_patch_index(self):
        """Pre-calcula las posiciones de todos los patches de forma determinista."""
        from .preprocessing import load_image_rgb
        import cv2

        # Calculamos posiciones usando las dimensiones esperadas del downscale
        # Para evitar cargar todas las imágenes, asumimos dimensión uniforme.
        # Si hay variación de tamaño, se calcula por imagen.
        h_ds = 1978  # 3956 // 2
        w_ds = 2640  # 5280 // 2

        positions = get_patch_positions(h_ds, w_ds, self.patch_size, self.stride)

        self._index = []
        for img_idx in range(len(self.df)):
            for (y, x) in positions:
                self._index.append((img_idx, y, x))

        self._positions_per_image = positions
        self._n_positions = len(positions)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        img_idx, y, x = self._index[idx]
        row = self.df.iloc[img_idx]

        img_path  = self.base_path / row["filepath"]
        mask_path = self.base_path / row["segmentation_mask_path"]

        # Cargar imagen y máscara
        try:
            image = load_image_rgb(img_path)
            h, w  = image.shape[:2]
            mask  = load_mask_binary(mask_path, h, w, self.base_path)
        except Exception as e:
            logger.warning(f"Error cargando {img_path}: {e}")
            img_patch  = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
            mask_patch = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        else:
            image_ds = downscale_image(image, self.downscale_factor)
            mask_ds  = downscale_mask(mask,   self.downscale_factor)
            img_patch  = extract_patch(image_ds, y, x, self.patch_size)
            mask_patch = extract_patch(mask_ds,  y, x, self.patch_size)

        mask_float = mask_to_float(mask_patch)
        img_tensor, mask_tensor = apply_transform_pair(
            self.transform, img_patch, mask_float
        )

        meta = {
            "img_idx": img_idx,
            "y": y,
            "x": x,
        }

        return img_tensor, mask_tensor.unsqueeze(0), meta
