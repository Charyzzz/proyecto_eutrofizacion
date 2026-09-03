"""
trainer.py
==========
Loop de entrenamiento completo con:
  - Mixed Precision (AMP) si CUDA disponible
  - Congelamiento/descongelamiento de encoder
  - Checkpointing del mejor modelo
  - Early stopping
  - Logging de métricas por época
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Optional, Dict
import logging
import json
import time

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Para el entrenamiento si la métrica de validación no mejora
    durante `patience` épocas consecutivas.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.stop       = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True

        return self.stop


class Trainer:
    """
    Encapsula el loop de entrenamiento completo.

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer
    loss_fn : nn.Module
    scheduler : opcional
    device : torch.device
    checkpoints_dir : Path
    use_amp : bool
    freeze_encoder_epochs : int
    early_stopping_patience : int
    monitor_metric : str
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        scheduler=None,
        device: torch.device = None,
        checkpoints_dir: Path = Path("outputs/unet/checkpoints"),
        use_amp: bool = True,
        freeze_encoder_epochs: int = 5,
        early_stopping_patience: int = 12,
        monitor_metric: str = "dice",
    ):
        self.model      = model
        self.optimizer  = optimizer
        self.loss_fn    = loss_fn
        self.scheduler  = scheduler
        self.device     = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.ckpt_dir   = Path(checkpoints_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # AMP: solo si CUDA disponible
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler  = GradScaler(enabled=self.use_amp)

        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.early_stop = EarlyStopping(patience=early_stopping_patience)
        self.monitor    = monitor_metric

        self.model.to(self.device)

        # Historial de métricas
        self.history: Dict[str, list] = {
            "train_loss": [], "val_dice": [], "val_iou": [],
            "val_precision": [], "val_recall": [], "lr": [],
        }

        self.best_val_score = 0.0
        self.best_epoch     = 0

        logger.info(f"Trainer inicializado | Device: {self.device} | AMP: {self.use_amp}")

    def train_one_epoch(self, loader: DataLoader) -> float:
        """Entrena una época. Devuelve la loss promedio."""
        self.model.train()
        total_loss = 0.0
        n_batches  = 0
        t0 = time.time()

        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device,  non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss   = self.loss_fn(logits, masks)

            self.scaler.scale(loss).backward()
            # Gradient clipping: previene explosión de gradientes
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches  += 1

            if batch_idx % 50 == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"  [{batch_idx}/{len(loader)}] "
                    f"loss={loss.item():.4f} | {elapsed:.1f}s"
                )

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        validator,          # instancia de Validator
        num_epochs: int = 50,
        start_epoch: int = 0,
    ):
        """
        Loop principal de entrenamiento.

        Parameters
        ----------
        train_loader : DataLoader
        validator : Validator
            Objeto con método evaluate(model, device, threshold) → SegMetrics
        num_epochs : int
        start_epoch : int
            Para reanudar desde un checkpoint
        """
        from .model import freeze_encoder, unfreeze_encoder, get_optimizer

        logger.info(f"Iniciando entrenamiento: {num_epochs} épocas en {self.device}")

        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch+1}/{num_epochs}")
            logger.info(f"{'='*60}")

            # Gestión del encoder (congelar / descongelar)
            if epoch == 0 and self.freeze_encoder_epochs > 0:
                freeze_encoder(self.model)
                logger.info(f"  [Estrategia] Encoder congelado por {self.freeze_encoder_epochs} épocas")

            if epoch == self.freeze_encoder_epochs and self.freeze_encoder_epochs > 0:
                unfreeze_encoder(self.model)
                logger.info("  [Estrategia] Encoder descongelado para fine-tuning")

            # Training
            train_loss = self.train_one_epoch(train_loader)
            logger.info(f"Train loss: {train_loss:.4f}")

            # Validation completa (con reconstrucción de imagen completa)
            val_metrics = validator.evaluate(self.model, self.device)
            logger.info(f"Val: {val_metrics}")

            # Scheduler
            current_score = getattr(val_metrics, self.monitor)
            if self.scheduler is not None:
                self.scheduler.step(current_score)

            # Logging
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_dice"].append(val_metrics.dice)
            self.history["val_iou"].append(val_metrics.iou)
            self.history["val_precision"].append(val_metrics.precision)
            self.history["val_recall"].append(val_metrics.recall)
            self.history["lr"].append(current_lr)

            # Checkpointing
            if current_score > self.best_val_score:
                self.best_val_score = current_score
                self.best_epoch     = epoch
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(
                    f"  ★ Nuevo mejor modelo: {self.monitor}={current_score:.4f} "
                    f"(epoch {epoch+1})"
                )

            # Checkpoint periódico (cada 5 épocas)
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)

            elapsed = time.time() - epoch_start
            logger.info(f"Época completada en {elapsed:.1f}s | LR={current_lr:.2e}")

            # Guardar historial
            self._save_history()

            # Early stopping
            if self.early_stop(current_score):
                logger.info(
                    f"Early stopping en época {epoch+1}. "
                    f"Mejor {self.monitor}={self.best_val_score:.4f} "
                    f"en época {self.best_epoch+1}"
                )
                break

        logger.info(f"\nEntrenamiento completado.")
        logger.info(
            f"Mejor modelo: epoch={self.best_epoch+1} | "
            f"{self.monitor}={self.best_val_score:.4f}"
        )

    def _save_checkpoint(self, epoch: int, val_metrics, is_best: bool = False):
        """Guarda el estado del modelo."""
        ckpt = {
            "epoch":      epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict":    self.scaler.state_dict(),
            "val_dice":    val_metrics.dice,
            "val_iou":     val_metrics.iou,
            "best_score":  self.best_val_score,
        }

        if is_best:
            path = self.ckpt_dir / "best_model.pth"
        else:
            path = self.ckpt_dir / f"checkpoint_epoch_{epoch+1:03d}.pth"

        torch.save(ckpt, path)

    def _save_history(self):
        """Guarda el historial de métricas en JSON."""
        path = self.ckpt_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self, checkpoint_path: Path):
        """Carga un checkpoint para reanudar el entrenamiento."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Checkpoint cargado: epoch={ckpt['epoch']+1} | Reanudando desde {start_epoch}")
        return start_epoch
