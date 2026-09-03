"""
model.py
========
U-Net con encoder ResNet34 preentrenado (ImageNet) via segmentation_models_pytorch.

DECISIONES DE ARQUITECTURA
---------------------------

Encoder: ResNet34
  - 4 bloques residuales con skip connections bien definidas
  - Preentrenado en ImageNet: las features iniciales son mucho mejores
    que random init, especialmente para imágenes de drones en exteriores
  - Más ligero que ResNet50 (21M vs 25M parámetros)
  - Ampliamente validado en segmentación de imágenes aéreas

Skip Connections (ResNet34 → U-Net decoder):
  Block 1 → 64 channels,  stride 1  (resolución: patch/2)
  Block 2 → 128 channels, stride 2  (resolución: patch/4)
  Block 3 → 256 channels, stride 4  (resolución: patch/8)
  Block 4 → 512 channels, stride 8  (resolución: patch/16)

El decoder de smp reconstruye la resolución original 512×512 usando
las skip connections en orden inverso.

Estrategia de congelamiento:
  - Épocas 0..freeze_epochs-1: encoder congelado, solo decoder entrena.
    Esto evita destruir los pesos de ImageNet con un LR alto inicial.
  - Épocas freeze_epochs en adelante: encoder descongelado con LR×0.1.
    Fine-tuning suave del encoder para adaptar al dominio de ríos.

Salida:
  - Logits crudos (sin sigmoid) de forma (B, 1, H, W)
  - Durante inferencia/evaluación: sigmoid(logits) → probabilidades [0,1]
  - Justificación: BCEWithLogitsLoss es numéricamente más estable
    que aplicar sigmoid antes de la BCE.
"""

import torch
import torch.nn as nn
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def build_unet(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    num_classes: int = 1,
    in_channels: int = 3,
) -> nn.Module:
    """
    Construye U-Net con encoder preentrenado usando segmentation_models_pytorch.

    Parameters
    ----------
    encoder_name : str
        Nombre del encoder (resnet34, resnet50, efficientnet-b3, etc.)
    encoder_weights : str
        Pesos preentrenados ('imagenet' o None)
    num_classes : int
        Número de clases de salida (1 para segmentación binaria)
    in_channels : int
        Canales de entrada (3 para RGB)

    Returns
    -------
    nn.Module
        Modelo U-Net
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "segmentation_models_pytorch no está instalado.\n"
            "Instálalo con: pip install segmentation-models-pytorch"
        )

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,          # Logits crudos; sigmoid se aplica en inferencia
        decoder_attention_type=None,  # Sin attention gates (más ligero)
    )

    logger.info(
        f"Modelo construido: U-Net + {encoder_name} "
        f"(weights={encoder_weights}, classes={num_classes})"
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"  Parámetros totales: {n_params:.1f}M | Entrenables: {n_train:.1f}M")

    return model


def freeze_encoder(model: nn.Module):
    """
    Congela los pesos del encoder.
    Solo el decoder y la cabeza de clasificación entrenan.
    """
    for param in model.encoder.parameters():
        param.requires_grad = False
    n_frozen = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    logger.info(f"Encoder congelado ({n_frozen:.1f}M parámetros)")


def unfreeze_encoder(model: nn.Module):
    """
    Descongela el encoder para fine-tuning.
    """
    for param in model.encoder.parameters():
        param.requires_grad = True
    n_unfrozen = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    logger.info(f"Encoder descongelado para fine-tuning ({n_unfrozen:.1f}M parámetros)")


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    encoder_lr_factor: float = 0.1,
    encoder_frozen: bool = True,
) -> torch.optim.Optimizer:
    """
    Crea el optimizador AdamW con grupos de parámetros diferenciados.

    Durante el congelamiento: solo parámetros del decoder.
    Durante el fine-tuning: encoder con LR reducido, decoder con LR normal.

    Parameters
    ----------
    model : nn.Module
    learning_rate : float
        LR base para decoder/head
    weight_decay : float
    encoder_lr_factor : float
        El encoder se entrena con LR = learning_rate * encoder_lr_factor
    encoder_frozen : bool
        Si True, el encoder no se incluye en el optimizador

    Returns
    -------
    torch.optim.AdamW
    """
    if encoder_frozen:
        params = [
            {"params": model.decoder.parameters(), "lr": learning_rate},
            {"params": model.segmentation_head.parameters(), "lr": learning_rate},
        ]
    else:
        params = [
            {"params": model.encoder.parameters(),
             "lr": learning_rate * encoder_lr_factor},
            {"params": model.decoder.parameters(),
             "lr": learning_rate},
            {"params": model.segmentation_head.parameters(),
             "lr": learning_rate},
        ]

    optimizer = torch.optim.AdamW(params, weight_decay=weight_decay)
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    patience: int = 5,
    factor: float = 0.5,
    min_lr: float = 1e-7,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """
    Scheduler ReduceLROnPlateau: reduce el LR cuando la métrica deja
    de mejorar. Robusto y no requiere especificar epochs de antemano.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",      # maximizar Dice
        patience=patience,
        factor=factor,
        min_lr=min_lr,
    )


def predict_proba(
    model: nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Inferencia: devuelve probabilidades [0,1] para un batch de patches.

    Parameters
    ----------
    model : nn.Module
    img_tensor : torch.Tensor
        (B, 3, H, W) normalizado
    device : torch.device

    Returns
    -------
    torch.Tensor
        (B, 1, H, W) probabilidades en [0, 1]
    """
    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        logits = model(img_tensor)
        proba  = torch.sigmoid(logits)
    return proba.cpu()
