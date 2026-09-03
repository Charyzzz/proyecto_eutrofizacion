"""
augmentation.py
===============
Augmentations para el pipeline de entrenamiento.

REGLAS FUNDAMENTALES
--------------------
1. Transformaciones GEOMÉTRICAS (flip, rotación): se aplican igual a
   imagen Y máscara usando la API de Albumentations con pares (image, mask).

2. Transformaciones de COLOR (brillo, contraste, saturación): se aplican
   SOLO a la imagen, nunca a la máscara.

3. NO augmentation en validation ni test.

4. No se usan transformaciones excesivamente agresivas que puedan
   producir imágenes irreales para el dominio de ríos/agua.

DECISIONES DE DISEÑO
--------------------
- HorizontalFlip: Los ríos no tienen orientación preferida. Dobla el dataset.
- VerticalFlip: Las imágenes de dron tienen perspectiva top-down,
  el flip vertical es plausible.
- Rotaciones pequeñas (≤15°): El dron puede tener ligera inclinación.
  Rotaciones grandes producirían bordes negros no realistas.
- Brillo/contraste leves: Simula variación de iluminación (hora del día,
  nubes). Se mantienen moderados para no alterar el color del agua.
- No se usa elastic transform: puede distorsionar la frontera agua/tierra
  de forma no realista.
"""

import numpy as np
from typing import Dict, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    patch_size: int = 512,
    imagenet_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    imagenet_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Pipeline de augmentation para TRAINING.

    Combina transformaciones geométricas (imagen + máscara)
    y de color (solo imagen).

    Parameters
    ----------
    patch_size : int
        Tamaño del patch (para RandomCrop si se necesita)
    imagenet_mean, imagenet_std : Tuple
        Media y std de ImageNet para normalización

    Returns
    -------
    A.Compose
        Pipeline de Albumentations con soporte para pares (image, mask)
    """
    return A.Compose([
        # ----------------------------------------------------------
        # TRANSFORMACIONES GEOMÉTRICAS (imagen + máscara)
        # ----------------------------------------------------------

        # Flip horizontal: muy común y seguro para imágenes de ríos
        A.HorizontalFlip(p=0.5),

        # Flip vertical: plausible en imágenes de dron top-down
        A.VerticalFlip(p=0.3),

        # Rotación pequeña: simula ligera inclinación del dron.
        # border_mode=REFLECT_101 evita bordes negros artificiales.
        A.Rotate(
            limit=15,
            border_mode=4,  # cv2.BORDER_REFLECT_101
            p=0.4,
        ),

        # ----------------------------------------------------------
        # TRANSFORMACIONES DE COLOR (solo imagen, no máscara)
        # ----------------------------------------------------------

        # Brillo y contraste leves: simula variación de iluminación
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5,
        ),

        # Variación de matiz, saturación y valor (HSV)
        # Moderado para no alterar el color característico del agua
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=15,
            p=0.3,
        ),

        # Ligero blur: simula movimiento del dron o desenfoque
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # Ruido leve: simula ruido del sensor de cámara.
        # std_range es fracción de 255 (API de Albumentations >=2.0);
        # (0.01, 0.02) equivale a la std del var_limit=(5, 20) usado antes.
        A.GaussNoise(std_range=(0.01, 0.02), p=0.2),

        # ----------------------------------------------------------
        # NORMALIZACIÓN (siempre al final, solo imagen)
        # Usa estadísticas de ImageNet porque el encoder ResNet34
        # fue preentrenado con esas estadísticas.
        # ----------------------------------------------------------
        A.Normalize(mean=imagenet_mean, std=imagenet_std),

        # Convierte a tensor PyTorch (C, H, W)
        ToTensorV2(),
    ])


def get_val_transforms(
    imagenet_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    imagenet_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Pipeline para VALIDATION y TEST.

    Solo normalización. Sin augmentation aleatorio.
    Determinista y reproducible.

    Parameters
    ----------
    imagenet_mean, imagenet_std : Tuple
        Media y std de ImageNet

    Returns
    -------
    A.Compose
    """
    return A.Compose([
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2(),
    ])


def get_inference_transform(
    imagenet_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    imagenet_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Transform para inferencia sobre patches individuales.
    Idéntico al de validación.
    """
    return get_val_transforms(imagenet_mean, imagenet_std)


def apply_transform_pair(
    transform: A.Compose,
    image: np.ndarray,
    mask: np.ndarray,
) -> Tuple:
    """
    Aplica un transform de Albumentations a un par (imagen, máscara).

    Parameters
    ----------
    transform : A.Compose
    image : np.ndarray
        Imagen RGB (H, W, 3), uint8
    mask : np.ndarray
        Máscara (H, W), valores {0, 1} o {0, 255}

    Returns
    -------
    image_tensor : torch.Tensor
        Tensor (3, H, W), float32, normalizado
    mask_tensor : torch.Tensor
        Tensor (1, H, W) o (H, W), float32, valores [0, 1]
    """
    # Albumentations espera máscara uint8 o float32
    if mask.dtype != np.uint8:
        mask_in = (mask * 255).astype(np.uint8)
    else:
        mask_in = mask

    result = transform(image=image, mask=mask_in)

    img_tensor  = result["image"]         # torch.Tensor (3, H, W)
    mask_tensor = result["mask"].float()  # torch.Tensor (H, W)

    # Normalizar máscara a [0, 1] si estaba en [0, 255]
    if mask_tensor.max() > 1.0:
        mask_tensor = mask_tensor / 255.0

    return img_tensor, mask_tensor
