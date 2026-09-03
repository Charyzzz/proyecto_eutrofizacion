"""
src/unet/__init__.py
Pipeline U-Net para segmentación binaria de agua en imágenes de drones.
"""

from .config        import UNetConfig
from .splits        import load_or_create_splits, verify_no_leakage
from .preprocessing import load_image_rgb, load_mask_binary, downscale_image, downscale_mask
from .patches       import get_patch_positions, extract_patch, compute_water_fractions
from .augmentation  import get_train_transforms, get_val_transforms
from .dataset       import WaterTrainDataset, WaterEvalDataset, worker_init_fn
from .model         import build_unet, freeze_encoder, unfreeze_encoder, get_optimizer, get_scheduler
from .loss          import BCEDiceLoss, DiceLoss
from .metrics       import compute_metrics, MetricAccumulator, SegMetrics
from .reconstruction import reconstruct_full_prediction
from .trainer       import Trainer
from .validator     import Validator
from .inference     import load_model, run_inference, batch_inference
from .visualization import plot_prediction, plot_overlay, plot_error_analysis, plot_training_history
