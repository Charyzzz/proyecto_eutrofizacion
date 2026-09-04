"""
scripts/visualize_predictions.py
=================================
Visualiza predicciones de la U-Net entrenada: imagen original, máscara
real (ground truth), mapa de probabilidades, predicción binarizada y
mapa de errores (TP/FP/FN), junto con sus métricas (Dice/IoU/Prec/Rec).

USO
---
Ver N imágenes aleatorias del test set:
    python notebooks/visualize_predictions.py --n 8

Ver imágenes de otro split:
    python notebooks/visualize_predictions.py --n 8 --split val

Ver imágenes específicas por nombre (ej. casos buenos/malos que viste en el log):
    python notebooks/visualize_predictions.py --imagenes DJI_1935.JPG DJI_5725.JPG DJI_12059.JPG

Guardar las figuras en vez de solo mostrarlas:
    python notebooks/visualize_predictions.py --n 8 --guardar

Usar un checkpoint distinto al best_model.pth:
    python notebooks/visualize_predictions.py --n 8 --checkpoint outputs/unet/checkpoints/checkpoint_epoch_010.pth
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE_PATH = Path(r"D:\proyecto_eutrofizacion")
sys.path.insert(0, str(BASE_PATH / "src"))

from unet.config import UNetConfig
from unet.splits import load_or_create_splits
from unet.preprocessing import load_image_rgb, load_mask_binary, downscale_image, downscale_mask
from unet.patches import get_patch_positions
from unet.reconstruction import reconstruct_full_prediction
from unet.augmentation import get_val_transforms
from unet.metrics import compute_metrics
from unet.inference import load_model
from unet.visualization import plot_prediction


def predecir_y_visualizar_imagen(row, base_path, model, device, config, transform, save_dir=None):
    """Corre el pipeline completo (downscale + patches + reconstrucción) sobre
    UNA imagen y la muestra junto a su ground truth y el mapa de errores."""
    img_path = base_path / row["filepath"]
    mask_path = base_path / row["segmentation_mask_path"]
    nombre = Path(row["filepath"]).name

    image = load_image_rgb(img_path)
    h, w = image.shape[:2]
    mask = load_mask_binary(mask_path, h, w, base_path)

    image_ds = downscale_image(image, config.downscale_factor)
    mask_ds = downscale_mask(mask, config.downscale_factor)
    h_ds, w_ds = image_ds.shape[:2]

    positions = get_patch_positions(h_ds, w_ds, config.patch_size, config.stride)

    prob_avg, mask_pred = reconstruct_full_prediction(
        model=model,
        image_ds=image_ds,
        positions=positions,
        transform=transform,
        patch_size=config.patch_size,
        threshold=config.threshold,
        batch_size=config.batch_size * 2,
        device=device,
    )

    true_mask_float = (mask_ds > 127).astype(np.float32)
    metrics = compute_metrics(prob_avg, true_mask_float, threshold=config.threshold)

    save_path = (Path(save_dir) / f"pred_{Path(nombre).stem}.png") if save_dir else None

    plot_prediction(
        image_ds=image_ds,
        true_mask=mask_ds,
        prob_avg=prob_avg,
        pred_mask=mask_pred,
        metrics=metrics,
        title=nombre,
        save_path=save_path,
    )

    print(f"{nombre}: {metrics}")
    return metrics


def buscar_por_nombre(nombre, df_train, df_val, df_test):
    """Busca una imagen por nombre de archivo en los tres splits y devuelve
    (fila, nombre_split), o (None, None) si no la encuentra."""
    for split_name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        coincidencias = df[df["filepath"].str.contains(nombre)]
        if not coincidencias.empty:
            return coincidencias.iloc[0], split_name
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Visualiza predicciones de la U-Net entrenada")
    parser.add_argument("--n", type=int, default=5, help="Número de imágenes aleatorias a visualizar")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--imagenes", type=str, nargs="+", default=None,
                         help="Nombres de archivo específicos a visualizar (ignora --n y --split)")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Ruta al checkpoint (default: outputs/unet/checkpoints/best_model.pth)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guardar", action="store_true",
                         help="Guarda las figuras en outputs/unet/visualizations/predicciones_manual/")
    args = parser.parse_args()

    config = UNetConfig(base_path=BASE_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_val_transforms(config.imagenet_mean, config.imagenet_std)

    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint
        else config.get_path(config.checkpoints_dir) / "best_model.pth"
    )
    if not checkpoint_path.exists():
        print(f"No se encontró el checkpoint: {checkpoint_path}")
        sys.exit(1)

    model = load_model(
        checkpoint_path,
        encoder_name=config.encoder_name,
        encoder_weights=None,
        device=device,
    )

    save_dir = None
    if args.guardar:
        save_dir = config.get_path(config.viz_dir) / "predicciones_manual"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Guardando figuras en: {save_dir}")

    df_train, df_val, df_test = load_or_create_splits(
        csv_path=config.get_path(config.csv_path),
        splits_dir=config.get_path(config.splits_dir),
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

    filas_a_visualizar = []

    if args.imagenes:
        for nombre in args.imagenes:
            fila, split_encontrado = buscar_por_nombre(nombre, df_train, df_val, df_test)
            if fila is None:
                print(f"⚠ No se encontró '{nombre}' en ningún split, se omite")
                continue
            print(f"{nombre} -> split: {split_encontrado}")
            filas_a_visualizar.append(fila)
    else:
        df_split = {"train": df_train, "val": df_val, "test": df_test}[args.split]
        rng = np.random.default_rng(args.seed)
        n = min(args.n, len(df_split))
        indices = rng.choice(len(df_split), size=n, replace=False)
        filas_a_visualizar = [df_split.iloc[i] for i in indices]
        print(f"Visualizando {n} imágenes aleatorias del split '{args.split}' (seed={args.seed})")

    todas_las_metricas = []
    for fila in filas_a_visualizar:
        m = predecir_y_visualizar_imagen(fila, BASE_PATH, model, device, config, transform, save_dir)
        todas_las_metricas.append(m)

    if todas_las_metricas:
        dices = [m.dice for m in todas_las_metricas]
        ious = [m.iou for m in todas_las_metricas]
        print(f"\n{'='*60}")
        print(f"Resumen de {len(todas_las_metricas)} imágenes")
        print(f"  Dice: media={np.mean(dices):.4f} | min={np.min(dices):.4f} | max={np.max(dices):.4f}")
        print(f"  IoU:  media={np.mean(ious):.4f} | min={np.min(ious):.4f} | max={np.max(ious):.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
