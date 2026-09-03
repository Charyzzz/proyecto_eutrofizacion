"""
scripts/train_unet.py
=====================
Script principal de entrenamiento del pipeline U-Net.

USO
---
    python scripts/train_unet.py

Para reanudar desde un checkpoint:
    python scripts/train_unet.py --resume outputs/unet/checkpoints/checkpoint_epoch_010.pth

Para evaluar thresholds al final:
    python scripts/train_unet.py --find-threshold
"""

import sys
import argparse
import logging
import random
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------
# Setup de paths
# -----------------------------------------------------------------------
BASE_PATH = Path(r"D:\proyecto_eutrofizacion")
sys.path.insert(0, str(BASE_PATH / "src"))

# -----------------------------------------------------------------------
# Imports del pipeline
# -----------------------------------------------------------------------
from unet.config      import UNetConfig
from unet.splits      import load_or_create_splits, verify_no_leakage
from unet.augmentation import get_train_transforms, get_val_transforms
from unet.dataset     import WaterTrainDataset, WaterEvalDataset, worker_init_fn
from unet.model       import build_unet, freeze_encoder, get_optimizer, get_scheduler
from unet.loss        import BCEDiceLoss
from unet.trainer     import Trainer
from unet.validator   import Validator
from unet.visualization import plot_training_history


# -----------------------------------------------------------------------
# Reproducibilidad
# -----------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# -----------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------
def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    return logging.getLogger(__name__)


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def main(args):
    # ---- Configuración ------------------------------------------------
    config = UNetConfig(base_path=BASE_PATH)
    config.setup_dirs()

    logger = setup_logging(config.get_path(config.logs_dir))
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    logger.info("\n=== CONFIGURACIÓN ===")
    logger.info(f"  Encoder: {config.encoder_name} (weights={config.encoder_weights})")
    logger.info(f"  Patch: {config.patch_size} | Stride: {config.stride} | Overlap: {config.overlap}")
    logger.info(f"  Batch size: {config.batch_size} | Epochs: {config.num_epochs}")
    logger.info(f"  LR: {config.learning_rate} | Weight decay: {config.weight_decay}")
    logger.info(f"  Loss: BCE×{config.bce_weight} + Dice×{config.dice_weight}")
    logger.info(f"  AMP: {config.use_amp} | Seed: {config.seed}")

    # ---- Splits -------------------------------------------------------
    logger.info("\n=== SPLITS ===")
    df_train, df_val, df_test = load_or_create_splits(
        csv_path  = config.get_path(config.csv_path),
        splits_dir= config.get_path(config.splits_dir),
        train_ratio=config.train_ratio,
        val_ratio  =config.val_ratio,
        test_ratio =config.test_ratio,
        seed       =config.seed,
    )
    verify_no_leakage(df_train, df_val, df_test)
    logger.info(f"  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # Guardar resumen de splits
    split_summary = {
        "n_train": len(df_train),
        "n_val":   len(df_val),
        "n_test":  len(df_test),
        "seed":    config.seed,
        "patch_size": config.patch_size,
        "stride":     config.stride,
    }
    with open(config.get_path(config.splits_dir) / "split_summary.json", "w") as f:
        json.dump(split_summary, f, indent=2)

    # ---- Transforms ---------------------------------------------------
    train_transform = get_train_transforms(
        patch_size=config.patch_size,
        imagenet_mean=config.imagenet_mean,
        imagenet_std =config.imagenet_std,
    )
    val_transform = get_val_transforms(
        imagenet_mean=config.imagenet_mean,
        imagenet_std =config.imagenet_std,
    )

    # ---- Datasets y DataLoaders ---------------------------------------
    logger.info("\n=== DATASETS ===")

    train_dataset = WaterTrainDataset(
        df               = df_train,
        base_path        = BASE_PATH,
        transform        = train_transform,
        patch_size       = config.patch_size,
        stride           = config.stride,
        downscale_factor = config.downscale_factor,
        patches_per_image= config.patches_per_image_train,
        water_ratio      = config.water_patch_ratio,
        boundary_ratio   = config.boundary_patch_ratio,
        no_water_ratio   = config.no_water_patch_ratio,
        min_water_fraction=config.min_water_fraction,
        boundary_max_water=config.boundary_max_water_fraction,
        seed             = config.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = config.batch_size,
        shuffle     = True,
        num_workers = config.num_workers,
        pin_memory  = config.pin_memory,
        worker_init_fn=worker_init_fn,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0,
        drop_last   = True,
    )

    logger.info(f"  Train dataset: {len(train_dataset):,} samples")
    logger.info(f"  Train loader:  {len(train_loader):,} batches/epoch")

    # ---- Modelo -------------------------------------------------------
    logger.info("\n=== MODELO ===")
    model = build_unet(
        encoder_name   = config.encoder_name,
        encoder_weights= config.encoder_weights,
        num_classes    = config.num_classes,
    )

    # ---- Loss ---------------------------------------------------------
    loss_fn = BCEDiceLoss(
        bce_weight  = config.bce_weight,
        dice_weight = config.dice_weight,
    )

    # ---- Optimizer (encoder congelado inicialmente) -------------------
    optimizer = get_optimizer(
        model           = model,
        learning_rate   = config.learning_rate,
        weight_decay    = config.weight_decay,
        encoder_lr_factor=config.encoder_lr_factor,
        encoder_frozen  = config.freeze_encoder_epochs > 0,
    )
    if config.freeze_encoder_epochs > 0:
        freeze_encoder(model)

    # ---- Scheduler ----------------------------------------------------
    scheduler = get_scheduler(
        optimizer = optimizer,
        patience  = config.scheduler_patience,
        factor    = config.scheduler_factor,
        min_lr    = config.min_lr,
    )

    # ---- Validator ----------------------------------------------------
    validator = Validator(
        df_val           = df_val,
        base_path        = BASE_PATH,
        patch_size       = config.patch_size,
        stride           = config.stride,
        downscale_factor = config.downscale_factor,
        threshold        = config.threshold,
        batch_size       = config.batch_size * 2,  # más patches en val (sin gradientes)
        imagenet_mean    = config.imagenet_mean,
        imagenet_std     = config.imagenet_std,
        max_images       = None,  # evaluar TODO el val set
        viz_dir          = config.get_path(config.viz_dir),
        num_viz_samples  = config.num_viz_samples,
    )

    # ---- Trainer ------------------------------------------------------
    trainer = Trainer(
        model                   = model,
        optimizer               = optimizer,
        loss_fn                 = loss_fn,
        scheduler               = scheduler,
        device                  = device,
        checkpoints_dir         = config.get_path(config.checkpoints_dir),
        use_amp                 = config.use_amp,
        freeze_encoder_epochs   = config.freeze_encoder_epochs,
        early_stopping_patience = config.early_stopping_patience,
        monitor_metric          = config.monitor_metric,
    )

    # ---- Reanudar desde checkpoint ------------------------------------
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_epoch = trainer.load_checkpoint(resume_path)
        else:
            logger.error(f"Checkpoint no encontrado: {resume_path}")
            sys.exit(1)

    # ---- Entrenamiento ------------------------------------------------
    logger.info("\n=== ENTRENAMIENTO ===")
    trainer.train(
        train_loader = train_loader,
        validator    = validator,
        num_epochs   = config.num_epochs,
        start_epoch  = start_epoch,
    )

    # ---- Curvas de entrenamiento --------------------------------------
    history_path = config.get_path(config.checkpoints_dir) / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        plot_training_history(
            history,
            save_path=config.get_path(config.viz_dir) / "training_curves.png",
        )

    # ---- Búsqueda del mejor threshold (sobre VAL, nunca TEST) ---------
    if args.find_threshold:
        logger.info("\n=== BÚSQUEDA DE THRESHOLD ÓPTIMO (sobre val) ===")
        best_model_path = config.get_path(config.checkpoints_dir) / "best_model.pth"

        if best_model_path.exists():
            from unet.inference import load_model
            best_model = load_model(
                best_model_path,
                encoder_name=config.encoder_name,
                encoder_weights=None,
                device=device,
            )
            threshold_result = validator.evaluate_thresholds(
                model=best_model,
                device=device,
                candidates=config.threshold_candidates,
                n_images=min(100, len(df_val)),
            )

            logger.info(
                f"\nMejor threshold: {threshold_result['best_threshold']} "
                f"(Dice={threshold_result['best_dice']:.4f})"
            )

            with open(config.get_path(config.logs_dir) / "best_threshold.json", "w") as f:
                json.dump(threshold_result, f, indent=2)
        else:
            logger.warning("No se encontró best_model.pth para evaluar thresholds.")

    # ---- Evaluación FINAL en TEST -------------------------------------
    logger.info("\n=== EVALUACIÓN FINAL EN TEST SET ===")
    logger.info("IMPORTANTE: El threshold usado aquí fue seleccionado sobre VAL, no TEST.")

    best_model_path = config.get_path(config.checkpoints_dir) / "best_model.pth"
    if best_model_path.exists():
        from unet.inference import load_model
        best_model = load_model(
            best_model_path,
            encoder_name=config.encoder_name,
            encoder_weights=None,
            device=device,
        )

        # Cargar mejor threshold si fue calculado
        threshold_path = config.get_path(config.logs_dir) / "best_threshold.json"
        if threshold_path.exists():
            with open(threshold_path) as f:
                best_threshold = json.load(f)["best_threshold"]
        else:
            best_threshold = config.threshold

        test_validator = Validator(
            df_val           = df_test,
            base_path        = BASE_PATH,
            patch_size       = config.patch_size,
            stride           = config.stride,
            downscale_factor = config.downscale_factor,
            threshold        = best_threshold,
            batch_size       = config.batch_size * 2,
            imagenet_mean    = config.imagenet_mean,
            imagenet_std     = config.imagenet_std,
            viz_dir          = config.get_path(config.viz_dir) / "test",
            num_viz_samples  = config.num_viz_samples,
        )

        test_metrics = test_validator.evaluate(
            model=best_model,
            device=device,
            threshold=best_threshold,
            save_viz=True,
        )

        logger.info(f"\n{'='*60}")
        logger.info("RESULTADOS FINALES (TEST SET)")
        logger.info(f"{'='*60}")
        logger.info(f"  Threshold:  {best_threshold}")
        logger.info(f"  Dice:       {test_metrics.dice:.4f}")
        logger.info(f"  IoU:        {test_metrics.iou:.4f}")
        logger.info(f"  Precision:  {test_metrics.precision:.4f}")
        logger.info(f"  Recall:     {test_metrics.recall:.4f}")
        logger.info(f"{'='*60}")

        # Guardar resultados de test
        test_results = {
            "threshold": best_threshold,
            **test_metrics.to_dict(),
        }
        with open(config.get_path(config.logs_dir) / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)


# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento U-Net segmentación de agua")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Ruta a checkpoint para reanudar entrenamiento"
    )
    parser.add_argument(
        "--find-threshold", action="store_true",
        help="Buscar threshold óptimo en val después del entrenamiento"
    )
    args = parser.parse_args()
    main(args)
