"""
loss.py
=======
Función de pérdida combinada: BCE + Dice.

JUSTIFICACIÓN
-------------
Problema: Segmentación binaria agua/no-agua con posible desbalance
          (en zonas de poca agua, la mayoría de píxeles son "no agua").

BCE (BCEWithLogitsLoss):
  - Estable numéricamente (fusiona sigmoid + BCE en un paso)
  - Gradientes bien definidos en toda la curva
  - Problema: si hay desbalance, el modelo puede aprender a predecir
    siempre "no agua" y obtener buena loss. No penaliza bien el olvido
    de píxeles de agua que son minoría.

Dice Loss:
  - Directamente optimiza el coeficiente de Dice (= F1 para segmentación)
  - Intrínsecamente robusto al desbalance (opera sobre overlaps)
  - Problema: puede ser inestable al inicio del entrenamiento cuando
    las predicciones son muy ruidosas.

Combinación BCE + Dice (0.5 cada una):
  - BCE proporciona gradientes estables al inicio
  - Dice empuja hacia maximizar el overlap real agua/predicción
  - Esta combinación es estándar en literatura de segmentación médica
    y ha demostrado funcionar bien en segmentación de agua en drones.

La loss opera sobre LOGITS (sin sigmoid), que es lo que produce el modelo.
El sigmoid se aplica internamente por BCEWithLogitsLoss y explícitamente
en DiceLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss para segmentación binaria.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    DiceLoss = 1 - Dice

    smooth: pequeño valor para estabilidad numérica (evita división por cero).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor
            Logits crudos (B, 1, H, W)
        targets : torch.Tensor
            Máscaras binarias (B, 1, H, W), valores {0.0, 1.0}

        Returns
        -------
        torch.Tensor
            Dice Loss escalar
        """
        proba = torch.sigmoid(logits)

        # Aplanar para calcular Dice sobre toda la imagen
        proba_flat   = proba.view(-1)
        targets_flat = targets.view(-1)

        intersection = (proba_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            proba_flat.sum() + targets_flat.sum() + self.smooth
        )

        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combinación lineal de BCE + Dice.

    loss = bce_weight * BCE(logits, targets) + dice_weight * Dice(logits, targets)

    Parameters
    ----------
    bce_weight : float
        Peso de la BCE (default 0.5)
    dice_weight : float
        Peso de la Dice Loss (default 0.5)
    smooth : float
        Suavizado para Dice
    pos_weight : torch.Tensor, optional
        Peso para la clase positiva en BCE. Útil si el agua es muy escasa.
        Ej: si hay 10× más no-agua que agua, usar pos_weight=torch.tensor([10.])
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: torch.Tensor = None,
    ):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth=smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor (B, 1, H, W)
        targets : torch.Tensor (B, 1, H, W), valores {0.0, 1.0}

        Returns
        -------
        torch.Tensor escalar
        """
        bce_loss  = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        total = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total

    def forward_with_components(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Igual que forward pero devuelve también los componentes individuales.
        Útil para logging durante entrenamiento.
        """
        bce_loss  = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total     = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total, bce_loss.item(), dice_loss.item()
