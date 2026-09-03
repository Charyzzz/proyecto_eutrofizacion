"""
patches.py
==========
Extracción de patches 512×512 con 50% overlap sobre imágenes 2640×1978.

MANEJO DE BORDES
----------------
Las dimensiones 2640×1978 no son múltiplos exactos de 512 ni del stride 256.

Estrategia implementada:
  Para cada fila/columna de patches, se generan posiciones con stride 256.
  Cuando una posición haría que el patch sobresalga de la imagen, el último
  patch se desplaza hacia atrás para quedar pegado al borde:

    última_posición_x = ancho_imagen - patch_size   (= 2640 - 512 = 2128)
    última_posición_y = alto_imagen  - patch_size   (= 1978 - 512 = 1466)

  De esta manera TODOS los patches son exactamente 512×512 sin padding.
  El solapamiento del último patch con el penúltimo puede ser mayor al 50%
  en los bordes, pero eso es aceptable y correcto.

CUÁNTOS PATCHES SE GENERAN (imagen 2640×1978, patch 512, stride 256)
---------------------------------------------------------------------
  Posiciones X: 0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2128 → 10
  Posiciones Y: 0, 256, 512, 768, 1024, 1280, 1466              → 7
  Total por imagen: 10 × 7 = 70 patches
"""

import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

# Tipos de alias
PatchCoord = Tuple[int, int]  # (y_start, x_start)


def get_patch_positions(
    image_h: int,
    image_w: int,
    patch_size: int = 512,
    stride: int = 256,
) -> List[PatchCoord]:
    """
    Genera las posiciones (y, x) de la esquina superior-izquierda de cada patch.

    Todos los patches son exactamente patch_size × patch_size.
    El manejo de bordes se realiza desplazando el último patch hacia adentro.

    Parameters
    ----------
    image_h, image_w : int
        Dimensiones de la imagen (tras downscale)
    patch_size : int
        Tamaño del patch cuadrado
    stride : int
        Desplazamiento entre patches adyacentes

    Returns
    -------
    List[Tuple[int, int]]
        Lista de (y_start, x_start) para cada patch
    """
    positions = []

    def _get_starts(dim_size: int) -> List[int]:
        """Genera posiciones de inicio en una dimensión."""
        starts = list(range(0, dim_size - patch_size + 1, stride))
        # Asegurar que el último patch llegue exactamente al borde
        last = dim_size - patch_size
        if not starts or starts[-1] < last:
            starts.append(last)
        return starts

    y_starts = _get_starts(image_h)
    x_starts = _get_starts(image_w)

    for y in y_starts:
        for x in x_starts:
            positions.append((y, x))

    return positions


def extract_patch(
    image: np.ndarray,
    y_start: int,
    x_start: int,
    patch_size: int = 512,
) -> np.ndarray:
    """
    Extrae un patch de una imagen en la posición (y_start, x_start).

    Parameters
    ----------
    image : np.ndarray
        Imagen (H, W) o (H, W, C)
    y_start, x_start : int
        Coordenadas de la esquina superior-izquierda
    patch_size : int

    Returns
    -------
    np.ndarray
        Patch de dimensiones (patch_size, patch_size) o (patch_size, patch_size, C)
    """
    return image[y_start:y_start + patch_size, x_start:x_start + patch_size]


def extract_all_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 512,
    stride: int = 256,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[PatchCoord]]:
    """
    Extrae todos los patches de una imagen y su máscara.

    Parameters
    ----------
    image : np.ndarray
        Imagen RGB (H, W, 3)
    mask : np.ndarray
        Máscara binaria (H, W), valores {0, 255} o {0.0, 1.0}
    patch_size : int
    stride : int

    Returns
    -------
    image_patches : List[np.ndarray]
        Patches de imagen (patch_size, patch_size, 3)
    mask_patches : List[np.ndarray]
        Patches de máscara (patch_size, patch_size)
    positions : List[Tuple[int, int]]
        Coordenadas (y_start, x_start) de cada patch
    """
    h, w = image.shape[:2]
    positions = get_patch_positions(h, w, patch_size, stride)

    image_patches = []
    mask_patches  = []

    for (y, x) in positions:
        img_patch  = extract_patch(image, y, x, patch_size)
        mask_patch = extract_patch(mask,  y, x, patch_size)

        # Verificación de dimensiones
        assert img_patch.shape[:2]  == (patch_size, patch_size), \
            f"Patch de imagen con forma incorrecta: {img_patch.shape}"
        assert mask_patch.shape[:2] == (patch_size, patch_size), \
            f"Patch de máscara con forma incorrecta: {mask_patch.shape}"

        image_patches.append(img_patch)
        mask_patches.append(mask_patch)

    return image_patches, mask_patches, positions


def compute_water_fractions(
    mask: np.ndarray,
    positions: List[PatchCoord],
    patch_size: int = 512,
) -> np.ndarray:
    """
    Calcula la fracción de píxeles de agua en cada patch de forma vectorizada.

    Este cálculo se usa para la estrategia de sampling ponderado.
    Se realiza UNA vez por imagen (no en cada acceso del DataLoader).

    Parameters
    ----------
    mask : np.ndarray
        Máscara binaria (H, W), valores {0, 255}
    positions : List[Tuple[int, int]]
    patch_size : int

    Returns
    -------
    np.ndarray
        Array (N,) con la fracción de agua [0.0, 1.0] de cada patch
    """
    mask_binary = (mask > 127).astype(np.float32)
    fractions = np.zeros(len(positions), dtype=np.float32)

    for i, (y, x) in enumerate(positions):
        patch = mask_binary[y:y + patch_size, x:x + patch_size]
        fractions[i] = patch.mean()

    return fractions


def classify_patches(
    water_fractions: np.ndarray,
    min_water: float = 0.05,
    max_water_for_boundary: float = 0.70,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Clasifica los patches en tres categorías para sampling ponderado.

    Categorías:
    - water: fracción > min_water (tiene agua suficiente)
    - boundary: fracción en (min_water, max_water_for_boundary) (frontera agua/tierra)
    - no_water: fracción <= min_water (sin agua significativa)

    Los patches de frontera son críticos para que la U-Net aprenda
    bien los bordes agua/tierra, que son el caso más difícil.

    Parameters
    ----------
    water_fractions : np.ndarray
        Array (N,) con fracciones de agua
    min_water : float
        Umbral mínimo para considerar un patch como "con agua"
    max_water_for_boundary : float
        Fracción máxima para considerar "frontera" (> esta es principalmente agua)

    Returns
    -------
    water_idx, boundary_idx, no_water_idx : np.ndarray
        Índices de patches en cada categoría
    """
    water_mask    = water_fractions > min_water
    boundary_mask = water_mask & (water_fractions <= max_water_for_boundary)
    no_water_mask = ~water_mask

    water_idx    = np.where(water_mask)[0]
    boundary_idx = np.where(boundary_mask)[0]
    no_water_idx = np.where(no_water_mask)[0]

    return water_idx, boundary_idx, no_water_idx


def sample_patch_indices(
    water_idx: np.ndarray,
    boundary_idx: np.ndarray,
    no_water_idx: np.ndarray,
    n_patches: int,
    water_ratio: float = 0.50,
    boundary_ratio: float = 0.35,
    no_water_ratio: float = 0.15,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Samplea índices de patches respetando las proporciones deseadas.

    Si alguna categoría tiene menos patches de los requeridos,
    se hace sampling con reemplazo para esa categoría.

    Parameters
    ----------
    water_idx, boundary_idx, no_water_idx : np.ndarray
        Índices de patches por categoría
    n_patches : int
        Total de patches a samplear
    water_ratio, boundary_ratio, no_water_ratio : float
        Proporciones deseadas (deben sumar 1.0)
    rng : np.random.Generator
        Generador de números aleatorios (para reproducibilidad)

    Returns
    -------
    np.ndarray
        Array de índices sampleados (N,)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sorteo categórico: reparte exactamente n_patches entre las 3 categorías
    # de forma aleatoria respetando las proporciones. A diferencia de usar
    # max(1, int(n_patches * ratio)) por categoría, esto nunca da conteos
    # negativos y funciona igual de bien para n_patches=1 (caso real de uso,
    # un patch por __getitem__) que para n_patches grandes.
    n_water, n_boundary, n_no_water = rng.multinomial(
        n_patches, [water_ratio, boundary_ratio, no_water_ratio]
    )

    def _sample(pool: np.ndarray, n: int) -> np.ndarray:
        if n == 0 or len(pool) == 0:
            return np.array([], dtype=int)
        replace = len(pool) < n
        return rng.choice(pool, size=n, replace=replace)

    sampled = np.concatenate([
        _sample(water_idx,    n_water),
        _sample(boundary_idx, n_boundary),
        _sample(no_water_idx, n_no_water),
    ])

    rng.shuffle(sampled)
    return sampled
