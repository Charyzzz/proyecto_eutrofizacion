"""
config.py
=========
Configuración centralizada del pipeline U-Net.
Todos los parámetros son modificables aquí sin tocar el resto del código.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional


@dataclass
class UNetConfig:
    # ------------------------------------------------------------------
    # PATHS (ajustar BASE_PATH según el entorno)
    # ------------------------------------------------------------------
    base_path: Path = Path(r"D:\proyecto_eutrofizacion")

    # Archivos de datos
    csv_path: str = "data/metadata/river_water_index.csv"

    # Splits (se crean si no existen; se respetan si ya existen)
    splits_dir: str = "data/splits/unet"

    # Outputs
    checkpoints_dir: str = "outputs/unet/checkpoints"
    logs_dir: str = "outputs/unet/logs"
    viz_dir: str = "outputs/unet/visualizations"

    # Cache de estadísticas de agua por imagen (one-time precompute)
    water_stats_cache: str = "data/metadata/water_stats_cache.csv"

    # ------------------------------------------------------------------
    # ESPECIFICACIONES DE IMAGEN
    # ------------------------------------------------------------------
    # Resolución original de las imágenes DJI
    image_size_original: Tuple[int, int] = (5280, 3956)   # (W, H)

    # Factor de reducción (×2 = mitad de resolución)
    # Justificación: reduce costo computacional de imágenes 4K manteniendo
    # suficiente detalle para segmentación robusta de cuerpos de agua.
    downscale_factor: int = 2

    # Resolución tras el downscale
    image_size_downscaled: Tuple[int, int] = (2640, 1978)  # (W, H)

    # ------------------------------------------------------------------
    # EXTRACCIÓN DE PATCHES
    # ------------------------------------------------------------------
    # Tamaño del patch cuadrado.
    # Justificación: 512×512 cabe en GPU de 8GB con batch ≥ 4,
    # y el receptive field de ResNet34 es suficiente para capturar
    # contexto de bordes agua/tierra.
    patch_size: int = 512

    # Overlap entre patches adyacentes.
    # Justificación 50%: cada píxel del borde aparece en ≥ 2 patches,
    # lo que estabiliza la reconstrucción por promedio y reduce artefactos.
    overlap: float = 0.50

    # Stride = patch_size * (1 - overlap)
    # Se calcula automáticamente en __post_init__
    stride: int = field(init=False)

    # ------------------------------------------------------------------
    # DIVISIÓN DEL DATASET (a nivel de IMAGEN, no de patch)
    # Justificación: previene data leakage; patches de la misma imagen
    # nunca se mezclan entre splits.
    # ------------------------------------------------------------------
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # ------------------------------------------------------------------
    # ESTRATEGIA DE SAMPLING (entrenamiento)
    # ------------------------------------------------------------------
    # Proporción de patches con agua significativa
    water_patch_ratio: float = 0.50

    # Proporción de patches con frontera agua/no-agua
    boundary_patch_ratio: float = 0.35

    # Proporción de patches sin agua (necesarios para aprender clase negativa)
    no_water_patch_ratio: float = 0.15

    # Un patch se considera "con agua" si tiene al menos este % de píxeles de agua
    min_water_fraction: float = 0.05

    # Un patch se considera "frontera" si tiene agua pero no es mayoritariamente agua
    boundary_max_water_fraction: float = 0.70

    # Número de patches a samplear por imagen por época durante training
    # Con 15000 imágenes × 8 patches = 120,000 muestras/época
    patches_per_image_train: int = 8

    # ------------------------------------------------------------------
    # MODELO
    # ------------------------------------------------------------------
    # Encoder preentrenado.
    # Justificación ResNet34:
    # - Preentrenado en ImageNet (acelera convergencia, mejora features)
    # - 4 bloques con skip connections bien definidas para U-Net
    # - Más ligero que ResNet50 pero más potente que ResNet18
    # - Ampliamente validado en segmentación de imágenes de drones
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"

    # Normalización ImageNet (para encoder preentrenado)
    # Se aplica SOLO a las imágenes, nunca a las máscaras
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    num_classes: int = 1  # segmentación binaria agua/no-agua

    # ------------------------------------------------------------------
    # ENTRENAMIENTO
    # ------------------------------------------------------------------
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # El encoder se congela inicialmente para no destruir los pesos
    # preentrenados con un LR alto. Se descongela después de N épocas
    # con un LR más bajo para fine-tuning suave.
    freeze_encoder_epochs: int = 5
    encoder_lr_factor: float = 0.1  # LR del encoder = lr * factor

    # ------------------------------------------------------------------
    # LOSS FUNCTION
    # ------------------------------------------------------------------
    # Combinación BCE + Dice.
    # BCE: gradientes estables, funciona bien con logits.
    # Dice: directamente maximiza el overlap (F1), robusto al desbalance.
    # Justificación: solo BCE tiende a ignorar píxeles de agua si son minoría;
    # solo Dice puede ser inestable al inicio del entrenamiento.
    bce_weight: float = 0.5
    dice_weight: float = 0.5

    # ------------------------------------------------------------------
    # THRESHOLD
    # ------------------------------------------------------------------
    # Threshold inicial para binarizar probabilidades.
    # Será optimizado sobre validation (NUNCA sobre test).
    threshold: float = 0.5

    # Thresholds a evaluar en validation para optimización
    threshold_candidates: List[float] = field(
        default_factory=lambda: [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    )

    # ------------------------------------------------------------------
    # SCHEDULER Y REGULARIZACIÓN
    # ------------------------------------------------------------------
    scheduler_patience: int = 5          # para ReduceLROnPlateau
    scheduler_factor: float = 0.5
    min_lr: float = 1e-7

    # ------------------------------------------------------------------
    # EARLY STOPPING Y CHECKPOINTING
    # ------------------------------------------------------------------
    early_stopping_patience: int = 12
    monitor_metric: str = "dice"  # métrica para seleccionar mejor modelo

    # ------------------------------------------------------------------
    # MIXED PRECISION
    # ------------------------------------------------------------------
    use_amp: bool = True  # AMP si CUDA disponible

    # ------------------------------------------------------------------
    # REPRODUCIBILIDAD
    # ------------------------------------------------------------------
    seed: int = 42

    # ------------------------------------------------------------------
    # DATALOADER
    # ------------------------------------------------------------------
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # ------------------------------------------------------------------
    # VISUALIZACIÓN
    # ------------------------------------------------------------------
    num_viz_samples: int = 10  # imágenes a guardar durante val/test

    def __post_init__(self):
        # Calcular stride automáticamente
        self.stride = int(self.patch_size * (1 - self.overlap))

        # Convertir paths a Path objects
        self.base_path = Path(self.base_path)

        # Verificar que los ratios sumen 1
        total = self.train_ratio + self.val_ratio + self.test_ratio
        assert abs(total - 1.0) < 1e-6, f"Los ratios deben sumar 1.0, suman {total}"

        # Verificar ratios de sampling
        total_sampling = (
            self.water_patch_ratio + self.boundary_patch_ratio + self.no_water_patch_ratio
        )
        assert abs(total_sampling - 1.0) < 1e-6, \
            f"Los ratios de sampling deben sumar 1.0, suman {total_sampling}"

    def get_path(self, relative_path: str) -> Path:
        """Devuelve una ruta absoluta combinando base_path con la ruta relativa."""
        return self.base_path / relative_path

    def setup_dirs(self):
        """Crea todos los directorios de output necesarios."""
        for d in [self.splits_dir, self.checkpoints_dir, self.logs_dir, self.viz_dir]:
            self.get_path(d).mkdir(parents=True, exist_ok=True)


# Instancia por defecto (importable directamente)
DEFAULT_CONFIG = UNetConfig()
