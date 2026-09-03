"""
splits.py
=========
División train/val/test a nivel de IMAGEN ORIGINAL.

CRÍTICO: la división se hace ANTES de extraer patches.
Esto previene data leakage: patches de una misma imagen nunca
aparecen en splits diferentes.

Si ya existen archivos de split en data/splits/unet/, se respetan.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_or_create_splits(
    csv_path: Path,
    splits_dir: Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    mask_format_filter: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga splits existentes o los crea si no existen.

    La división se realiza a nivel de imagen, no de patch.

    Parameters
    ----------
    csv_path : Path
        Ruta al CSV de índice del dataset (river_water_index.csv)
    splits_dir : Path
        Directorio donde guardar/leer los splits
    train_ratio, val_ratio, test_ratio : float
        Proporción de cada split (deben sumar 1.0)
    seed : int
        Semilla para reproducibilidad
    mask_format_filter : str, optional
        Si se especifica, filtra por mask_format (ej: 'json')

    Returns
    -------
    df_train, df_val, df_test : pd.DataFrame
        Cada DataFrame contiene las filas del CSV correspondientes al split.
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.csv"
    val_path   = splits_dir / "val.csv"
    test_path  = splits_dir / "test.csv"

    # ----------------------------------------------------------------
    # Si ya existen los splits, cargarlos y respetar la división
    # ----------------------------------------------------------------
    if train_path.exists() and val_path.exists() and test_path.exists():
        logger.info("Splits existentes encontrados. Cargando sin modificar.")
        df_train = pd.read_csv(train_path)
        df_val   = pd.read_csv(val_path)
        df_test  = pd.read_csv(test_path)

        logger.info(
            f"  Train: {len(df_train)} imágenes | "
            f"Val: {len(df_val)} | "
            f"Test: {len(df_test)}"
        )
        return df_train, df_val, df_test

    # ----------------------------------------------------------------
    # Crear splits desde el CSV
    # ----------------------------------------------------------------
    logger.info(f"Creando splits desde: {csv_path}")
    df = pd.read_csv(csv_path)

    # Filtrar solo imágenes que tienen máscara de segmentación
    df = df.dropna(subset=["segmentation_mask_path"])
    df = df[df["segmentation_mask_path"].astype(str).str.strip() != ""]

    if mask_format_filter:
        df = df[df["mask_format"].astype(str).str.lower() == mask_format_filter.lower()]

    logger.info(f"Imágenes con máscara disponible: {len(df)}")

    if len(df) == 0:
        raise ValueError("No se encontraron imágenes con máscaras. Verifica el CSV.")

    # Shuffle con semilla fija para reproducibilidad
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # ----------------------------------------------------------------
    # División a nivel de imagen
    # Ningún patch de una imagen puede aparecer en dos splits diferentes
    # ----------------------------------------------------------------
    n = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    # El resto va a test (así no se pierden imágenes por redondeo)
    n_test  = n - n_train - n_val

    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val   = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test  = df.iloc[n_train + n_val:].reset_index(drop=True)

    # Añadir columna de split para trazabilidad
    df_train["split"] = "train"
    df_val["split"]   = "val"
    df_test["split"]  = "test"

    # Guardar para que próximas ejecuciones sean consistentes
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    logger.info(
        f"Splits creados y guardados en {splits_dir}\n"
        f"  Train: {len(df_train)} imágenes ({len(df_train)/n*100:.1f}%)\n"
        f"  Val:   {len(df_val)} imágenes ({len(df_val)/n*100:.1f}%)\n"
        f"  Test:  {len(df_test)} imágenes ({len(df_test)/n*100:.1f}%)"
    )

    return df_train, df_val, df_test


def verify_no_leakage(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    id_column: str = "filepath",
) -> bool:
    """
    Verifica que no haya solapamiento de imágenes entre splits.
    Lanza un AssertionError si hay data leakage.
    """
    train_ids = set(df_train[id_column].tolist())
    val_ids   = set(df_val[id_column].tolist())
    test_ids  = set(df_test[id_column].tolist())

    train_val_overlap  = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap   = val_ids & test_ids

    assert len(train_val_overlap) == 0, \
        f"DATA LEAKAGE: {len(train_val_overlap)} imágenes compartidas entre train y val"
    assert len(train_test_overlap) == 0, \
        f"DATA LEAKAGE: {len(train_test_overlap)} imágenes compartidas entre train y test"
    assert len(val_test_overlap) == 0, \
        f"DATA LEAKAGE: {len(val_test_overlap)} imágenes compartidas entre val y test"

    logger.info("✓ Verificación de data leakage: sin solapamiento entre splits.")
    return True
