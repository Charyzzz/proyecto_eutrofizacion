"""
metrics.py
==========
Métricas de segmentación binaria.

MÉTRICAS IMPLEMENTADAS
----------------------
- Dice / F1: 2×TP / (2×TP + FP + FN)  — métrica principal
- IoU / Jaccard: TP / (TP + FP + FN)   — métrica de overlap
- Precision: TP / (TP + FP)            — ¿cuánto del predicho es correcto?
- Recall: TP / (TP + FN)               — ¿cuánto del real se detectó?

JUSTIFICACIÓN
-------------
- Accuracy sola es inútil si hay desbalance (predecir todo "no agua" da
  alta accuracy si el agua es minoría).
- Dice e IoU son directamente interpretables como "qué tan bien se superpone
  la predicción con la máscara real". Son las métricas de referencia
  en segmentación semántica.
- Precision y Recall permiten diagnosticar el tipo de error:
  * Precision baja → muchos falsos positivos (confunde tierra con agua)
  * Recall bajo → muchos falsos negativos (pierde agua real)

Se calculan tanto por imagen individual como globalmente (acumulando
TP/FP/FN en todo el split de evaluación).
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SegMetrics:
    """Contenedor de métricas de segmentación."""
    dice:      float = 0.0
    iou:       float = 0.0
    precision: float = 0.0
    recall:    float = 0.0
    tp:        int   = 0
    fp:        int   = 0
    fn:        int   = 0
    tn:        int   = 0

    def __repr__(self):
        return (
            f"Dice={self.dice:.4f} | IoU={self.iou:.4f} | "
            f"Prec={self.precision:.4f} | Rec={self.recall:.4f}"
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "dice":      self.dice,
            "iou":       self.iou,
            "precision": self.precision,
            "recall":    self.recall,
        }


def compute_metrics(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> SegMetrics:
    """
    Calcula métricas de segmentación binaria.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicción: probabilidades [0,1] o binario {0,1}
        Forma: (H, W) o (1, H, W)
    true_mask : np.ndarray
        Máscara real binaria {0, 1} o {0, 255}
        Forma: (H, W) o (1, H, W)
    threshold : float
        Umbral para binarizar pred_mask
    eps : float
        Epsilon para estabilidad numérica

    Returns
    -------
    SegMetrics
    """
    # Normalizar formas
    pred = pred_mask.squeeze()
    true = true_mask.squeeze()

    # Binarizar predicción
    if pred.max() > 1.0 + eps:
        pred = pred / 255.0
    pred_bin = (pred >= threshold).astype(np.uint8)

    # Normalizar máscara real
    if true.max() > 1.0 + eps:
        true_bin = (true > 127).astype(np.uint8)
    else:
        true_bin = (true >= threshold).astype(np.uint8)

    # Calcular TP, FP, FN, TN
    tp = int(((pred_bin == 1) & (true_bin == 1)).sum())
    fp = int(((pred_bin == 1) & (true_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (true_bin == 1)).sum())
    tn = int(((pred_bin == 0) & (true_bin == 0)).sum())

    # Métricas
    dice      = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou       = (tp + eps)     / (tp + fp + fn + eps)
    precision = (tp + eps)     / (tp + fp + eps)
    recall    = (tp + eps)     / (tp + fn + eps)

    return SegMetrics(
        dice=float(dice),
        iou=float(iou),
        precision=float(precision),
        recall=float(recall),
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


class MetricAccumulator:
    """
    Acumula TP/FP/FN/TN sobre múltiples imágenes para calcular
    métricas GLOBALES al final del epoch de validación.

    Las métricas globales (micro-averaged) son más robustas que el
    promedio de métricas por imagen cuando las imágenes tienen tamaños
    de máscara muy diferentes.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0
        self.per_image_metrics: List[SegMetrics] = []

    def update(self, metrics: SegMetrics):
        """Añade métricas de una imagen."""
        self.total_tp += metrics.tp
        self.total_fp += metrics.fp
        self.total_fn += metrics.fn
        self.total_tn += metrics.tn
        self.per_image_metrics.append(metrics)

    def compute_global(self, eps: float = 1e-7) -> SegMetrics:
        """
        Calcula métricas globales (micro-averaged) acumulando todos los
        TP/FP/FN de todas las imágenes evaluadas.
        """
        tp, fp, fn, tn = self.total_tp, self.total_fp, self.total_fn, self.total_tn

        dice      = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou       = (tp + eps)     / (tp + fp + fn + eps)
        precision = (tp + eps)     / (tp + fp + eps)
        recall    = (tp + eps)     / (tp + fn + eps)

        return SegMetrics(
            dice=float(dice),
            iou=float(iou),
            precision=float(precision),
            recall=float(recall),
            tp=tp, fp=fp, fn=fn, tn=tn,
        )

    def compute_mean(self) -> SegMetrics:
        """
        Calcula la MEDIA de las métricas por imagen (macro-averaged).
        Complementa a compute_global().
        """
        if not self.per_image_metrics:
            return SegMetrics()

        dices  = [m.dice      for m in self.per_image_metrics]
        ious   = [m.iou       for m in self.per_image_metrics]
        precs  = [m.precision for m in self.per_image_metrics]
        recs   = [m.recall    for m in self.per_image_metrics]

        return SegMetrics(
            dice=float(np.mean(dices)),
            iou=float(np.mean(ious)),
            precision=float(np.mean(precs)),
            recall=float(np.mean(recs)),
        )


def find_best_threshold(
    pred_proba: np.ndarray,
    true_mask: np.ndarray,
    candidates: List[float] = None,
    metric: str = "dice",
) -> Dict:
    """
    Evalúa múltiples thresholds sobre un conjunto de predicciones y
    devuelve el que maximiza la métrica indicada.

    IMPORTANTE: Solo usar sobre VALIDATION, nunca sobre TEST.

    Parameters
    ----------
    pred_proba : np.ndarray
        Probabilidades predichas (H, W) o stack (N, H, W)
    true_mask : np.ndarray
        Máscaras reales correspondientes
    candidates : List[float]
        Thresholds a evaluar
    metric : str
        Métrica a optimizar ('dice' o 'iou')

    Returns
    -------
    dict con 'best_threshold', 'best_score', 'all_results'
    """
    if candidates is None:
        candidates = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    results = {}
    for t in candidates:
        m = compute_metrics(pred_proba, true_mask, threshold=t)
        results[t] = getattr(m, metric)

    best_t = max(results, key=results.get)
    return {
        "best_threshold": best_t,
        "best_score": results[best_t],
        "all_results": results,
    }
