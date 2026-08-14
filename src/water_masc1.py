import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt


def segmentar_agua(imagen,
                   a_min=-120,
                   a_max=-1,
                   b_min=-120,
                   b_max=14,
                   l_min=50,
                   l_max=210,
                   h_min=22,
                   h_max=180,
                   s_min=0,
                   s_max=110,
                   downscale=4,
                   verbose=False):
    """
    Segmentación de agua usando Lab + HSV combinados.
    
    ESTRATEGIA:
    1. Usa a* (Lab) como discriminador principal (rojo-verde):
       - Agua turbia verde: a* negativo (-120 a -10)
       - Vegetación: a* positivo o cercano a 0
    
    2. Usa b* (Lab) como discriminador secundario (azul-amarillo):
       - Agua azul/turbia: b* negativo (-80 a -10)
       - Grava clara: b* positivo (+10 a +80)
       - CRUCIAL para eliminar falsos positivos
    
    3. Usa L (Iluminancia) como filtro:
       - Agua es típicamente oscura (L < 100)
       - Excluye cielo brillante (L > 200)
    
    4. Usa H (Hue) para refinar:
       - Agua: Hue azul-verde (30-180)
       - Vegetación: Hue verde (60-90)
    
    5. Usa S (Saturación) como ajuste fino:
       - Baja influencia pero ayuda a descartar ruido
    Retorna:
    --------
    np.ndarray : Máscara binaria (255=agua, 0=no-agua)
    """
    
    # PASO 1: Obtén dimensiones originales
    altura_orig, ancho_orig = imagen.shape[:2]
    
    # PASO 2: Redimensiona (downsampling)
    if downscale > 1:
        altura_new = altura_orig // downscale
        ancho_new = ancho_orig // downscale
        imagen_small = cv2.resize(
            imagen,
            (ancho_new, altura_new),
            interpolation=cv2.INTER_AREA
        )
    else:
        imagen_small = imagen
    
    # PASO 3: Convierte a Lab color space
    lab = cv2.cvtColor(imagen_small, cv2.COLOR_BGR2Lab)
    l_channel = lab[:, :, 0].astype(np.float32)
    a_channel = lab[:, :, 1].astype(np.float32) - 128  # Centra en 0
    b_channel = lab[:, :, 2].astype(np.float32) - 128  # Centra en 0
    
    # PASO 4: Convierte a HSV para H y S
    hsv = cv2.cvtColor(imagen_small, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    
    if verbose:
        print(f"\nESTADÍSTICAS DE CANALES:")
        print(f"  a* (rojo-verde): [{a_channel.min():.1f}, {a_channel.max():.1f}], "
              f"mean={a_channel.mean():.1f}")
        print(f"  b* (azul-amarillo): [{b_channel.min():.1f}, {b_channel.max():.1f}], "
              f"mean={b_channel.mean():.1f}")
        print(f"  L (iluminancia): [{l_channel.min():.1f}, {l_channel.max():.1f}], "
              f"mean={l_channel.mean():.1f}")
        print(f"  H (tonalidad): [{h_channel.min()}, {h_channel.max()}], "
              f"mean={h_channel.mean():.1f}")
        print(f"  S (saturación): [{s_channel.min()}, {s_channel.max()}], "
              f"mean={s_channel.mean():.1f}")
    
    # PASO 5: Threshold en cada canal
    mask_a = (a_channel >= a_min) & (a_channel <= a_max)
    mask_b = (b_channel >= b_min) & (b_channel <= b_max)
    mask_l = (l_channel >= l_min) & (l_channel <= l_max)
    mask_h = (h_channel >= h_min) & (h_channel <= h_max)
    mask_s = (s_channel >= s_min) & (s_channel <= s_max)
    
    # PASO 6: Combina máscaras (todas deben cumplirse = AND lógico)
    mascara_combined = (mask_a & mask_b & mask_l & mask_h & mask_s).astype(np.uint8) * 255

    # PASO 8: Restaura a resolución original
    if downscale > 1:
        mascara_final = cv2.resize(
            mascara_combined,
            (ancho_orig, altura_orig),
            interpolation=cv2.INTER_NEAREST
        )
    else:
        mascara_final = mascara_combined
    
    return mascara_final


def visualizar_comparacion_mascaras(imagen,
                               mascara_gt,
                               a_min=-128,
                               a_max=-5,
                               b_min=-128,      # ← NUEVO
                               b_max=127,       # ← NUEVO
                               l_min=10,
                               l_max=220,
                               h_min=30,
                               h_max=180,
                               s_min=0,
                               s_max=255,
                               downscale=4):
    """
    Visualiza la segmentación en detalle con canales individuales.
    Útil para diagnosticar y ajustar parámetros.
    Ahora incluye visualización de b* (eje azul-amarillo).
    """
    
    altura_orig, ancho_orig = imagen.shape[:2]
    
    if downscale > 1:
        altura_new = altura_orig // downscale
        ancho_new = ancho_orig // downscale
        imagen_small = cv2.resize(
            imagen,
            (ancho_new, altura_new),
            interpolation=cv2.INTER_AREA
        )
    else:
        imagen_small = imagen
    
    lab = cv2.cvtColor(imagen_small, cv2.COLOR_BGR2Lab)
    hsv = cv2.cvtColor(imagen_small, cv2.COLOR_BGR2HSV)
    
    l_channel = lab[:, :, 0].astype(np.float32)
    a_channel = lab[:, :, 1].astype(np.float32) - 128
    b_channel = lab[:, :, 2].astype(np.float32) - 128  # [NUEVO]
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    
    # Máscaras por canal
    mask_a = ((a_channel >= a_min) & (a_channel <= a_max)).astype(np.uint8) * 255
    mask_b = ((b_channel >= b_min) & (b_channel <= b_max)).astype(np.uint8) * 255  # [NUEVO]
    mask_l = ((l_channel >= l_min) & (l_channel <= l_max)).astype(np.uint8) * 255
    mask_h = ((h_channel >= h_min) & (h_channel <= h_max)).astype(np.uint8) * 255
    mask_s = ((s_channel >= s_min) & (s_channel <= s_max)).astype(np.uint8) * 255
    
    # Restaura para visualizar
    mask_a_full = cv2.resize(mask_a, (ancho_orig, altura_orig), interpolation=cv2.INTER_NEAREST)
    mask_b_full = cv2.resize(mask_b, (ancho_orig, altura_orig), interpolation=cv2.INTER_NEAREST)  # [NUEVO]
    mask_l_full = cv2.resize(mask_l, (ancho_orig, altura_orig), interpolation=cv2.INTER_NEAREST)
    mask_h_full = cv2.resize(mask_h, (ancho_orig, altura_orig), interpolation=cv2.INTER_NEAREST)
    mask_s_full = cv2.resize(mask_s, (ancho_orig, altura_orig), interpolation=cv2.INTER_NEAREST)
    
    # Máscara final
    mascara_final = segmentar_agua(
        imagen,
        a_min=a_min,
        a_max=a_max,
        b_min=b_min,      # [NUEVO]
        b_max=b_max,      # [NUEVO]
        l_min=l_min,
        l_max=l_max,
        h_min=h_min,
        h_max=h_max,
        s_min=s_min,
        s_max=s_max,
        downscale=downscale
    )
    
    # Cálcula métricas
    metricas = calcular_metricas(mascara_final, mascara_gt)
    
    # Visualiza en 3 filas, 3 columnas (9 subgráficos)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    # FILA 1: Imagen original y primeras máscaras
    img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Imagen Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_a_full, cmap='gray')
    axes[0, 1].set_title(f'Máscara a* ({a_min} a {a_max})\n[rojo-verde]')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(mask_b_full, cmap='gray')
    axes[0, 2].set_title(f'Máscara b* ({b_min} a {b_max})\n[azul-amarillo] [NUEVO]')
    axes[0, 2].axis('off')
    
    # FILA 2: Máscaras de L, H, S
    axes[1, 0].imshow(mask_l_full, cmap='gray')
    axes[1, 0].set_title(f'Máscara L ({l_min} a {l_max})\n[iluminancia]')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask_h_full, cmap='gray')
    axes[1, 1].set_title(f'Máscara H ({h_min} a {h_max})\n[tonalidad]')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(mask_s_full, cmap='gray')
    axes[1, 2].set_title(f'Máscara S ({s_min} a {s_max})\n[saturación - bajo peso]')
    axes[1, 2].axis('off')
    
    # FILA 3: Resultados finales y análisis
    axes[2, 0].imshow(mascara_final, cmap='gray')
    axes[2, 0].set_title(
        f'FINAL (a* ∩ b* ∩ L ∩ H ∩ S)\n'
        f'IoU: {metricas["iou"]:.4f} | F1: {metricas["f1"]:.4f}'
    )
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(mascara_gt, cmap='gray')
    axes[2, 1].set_title('Ground Truth (Referencia)')
    axes[2, 1].axis('off')
    
    # Análisis de errores
    comparacion = np.zeros((imagen.shape[0], imagen.shape[1], 3), dtype=np.uint8)
    tp = np.logical_and(mascara_final > 0, mascara_gt > 0)
    fp = np.logical_and(mascara_final > 0, mascara_gt == 0)
    fn = np.logical_and(mascara_final == 0, mascara_gt > 0)
    comparacion[tp] = [0, 255, 0]    # Verde = TP
    comparacion[fp] = [0, 0, 255]    # Rojo = FP
    comparacion[fn] = [255, 0, 0]    # Azul = FN
    
    axes[2, 2].imshow(comparacion)
    axes[2, 2].set_title(
        f'Análisis de Errores\n'
        f'Verde=TP, Rojo=FP, Azul=FN\n'
        f'Precision: {metricas["precision"]:.4f} | Recall: {metricas["recall"]:.4f}'
    )
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    return fig, metricas


def visualizar_mascara_final(imagen, mascara_final, mascara_gt, metricas=None):
    """
    Visualiza en una sola figura:
    1. Imagen original
    2. Máscara final (predicción)
    3. Máscara ground truth (referencia)
    4. Análisis de errores (TP, FP, FN)
    
    Parámetros:
    -----------
    imagen : np.ndarray
        Imagen original en formato BGR (como la lee cv2.imread)
    mascara_final : np.ndarray
        Máscara predicha por el modelo (255=agua, 0=no-agua)
    mascara_gt : np.ndarray
        Máscara ground truth anotada (255=agua, 0=no-agua)
    metricas : dict (opcional)
        Diccionario con 'iou', 'precision', 'recall', 'f1'
        Si no se pasa, se calcula automáticamente
    
    Retorna:
    --------
    fig : matplotlib figure
        La figura creada
    metricas : dict
        Diccionario con métricas (IoU, Precision, Recall, F1)
    """
    
    # Calcula métricas si no se proporcionan
    if metricas is None:
        pred = (mascara_final > 0).astype(np.uint8)
        gt = (mascara_gt > 0).astype(np.uint8)
        
        tp = np.logical_and(pred == 1, gt == 1).sum()
        fp = np.logical_and(pred == 1, gt == 0).sum()
        fn = np.logical_and(pred == 0, gt == 1).sum()
        
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metricas = {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    # Crea figura con 2x2 subgráficos
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # [0, 0] Imagen original
    img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Imagen Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # [0, 1] Máscara final predicha
    axes[0, 1].imshow(mascara_final, cmap='gray')
    axes[0, 1].set_title(
        f'Máscara Final (Predicción)\nIoU: {metricas["iou"]:.4f}',
        fontsize=12, fontweight='bold'
    )
    axes[0, 1].axis('off')
    
    # [1, 0] Ground truth
    axes[1, 0].imshow(mascara_gt, cmap='gray')
    axes[1, 0].set_title(
        'Máscara Ground Truth (Referencia)',
        fontsize=12, fontweight='bold'
    )
    axes[1, 0].axis('off')
    
    # [1, 1] Análisis de errores
    # Verde = TP (True Positive - aciertos)
    # Rojo = FP (False Positive - falsos positivos)
    # Azul = FN (False Negative - falsos negativos)
    
    comparacion = np.zeros((imagen.shape[0], imagen.shape[1], 3), dtype=np.uint8)
    
    tp_mask = np.logical_and(mascara_final > 0, mascara_gt > 0)
    fp_mask = np.logical_and(mascara_final > 0, mascara_gt == 0)
    fn_mask = np.logical_and(mascara_final == 0, mascara_gt > 0)
    
    comparacion[tp_mask] = [0, 255, 0]      # Verde = TP (aciertos)
    comparacion[fp_mask] = [0, 0, 255]      # Rojo = FP (falsos positivos)
    comparacion[fn_mask] = [255, 0, 0]      # Azul = FN (falsos negativos)
    
    axes[1, 1].imshow(comparacion)
    axes[1, 1].set_title(
        f'Análisis de Errores\n'
        f'Verde=TP (✓), Rojo=FP, Azul=FN\n'
        f'Precision: {metricas["precision"]:.4f} | Recall: {metricas["recall"]:.4f} | F1: {metricas["f1"]:.4f}',
        fontsize=11, fontweight='bold'
    )
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    return fig, metricas

# ============================================================================
# GROUND TRUTH Y MÉTRICAS
# ============================================================================

def cargar_mascara_desde_labeling(
    ruta_json,
    alto_img,
    ancho_img
):
    """
    Carga la máscara ground truth desde JSON.

    Solo utiliza las regiones etiquetadas como 'water'.
    """

    mascara = np.zeros(
        (alto_img, ancho_img),
        dtype=np.uint8
    )

    try:

        with open(ruta_json, 'r') as f:
            data = json.load(f)

    except (FileNotFoundError, json.JSONDecodeError) as e:

        print(
            f"Error al leer {ruta_json}: {e}"
        )

        return mascara

    for shape in data.get('shapes', []):

        if shape.get(
            'label',
            ''
        ).lower() == 'water':

            points = np.array(
                shape['points'],
                dtype=np.int32
            )

            cv2.fillPoly(
                mascara,
                [points],
                255
            )

    return mascara


def calcular_metricas(mascara_predicha, mascara_ground_truth):
    """
    Calcula IoU, Precision, Recall, F1-score.
    """

    pred = (
        mascara_predicha > 0
    ).astype(np.uint8)

    gt = (
        mascara_ground_truth > 0
    ).astype(np.uint8)

    tp = np.logical_and(
        pred == 1,
        gt == 1
    ).sum()

    fp = np.logical_and(
        pred == 1,
        gt == 0
    ).sum()

    fn = np.logical_and(
        pred == 0,
        gt == 1
    ).sum()

    tn = np.logical_and(
        pred == 0,
        gt == 0
    ).sum()

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }