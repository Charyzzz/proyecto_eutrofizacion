"""
ETAPA 2 — SEGMENTACIÓN INTERNA DEL AGUA CON SUPERPÍXELES SLIC

Entrada: Máscara de agua (de Etapa 1)
Salida: Superpíxeles (regiones pequeñas coherentes dentro del agua)

Función: Dividir el agua en parches pequeños y coherentes (superpíxeles)
que servirán como unidad de análisis para la Etapa 3 (extracción de features)
y Etapa 4 (detección de anomalías).
"""

import os
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from collections import defaultdict

# ================================================================
# FUNCIÓN PRINCIPAL: APLICAR SLIC SOLO DENTRO DEL AGUA
# ================================================================

def aplicar_slic(imagen_bgr, mascara_agua, n_segments=300, compactness=10, downscale=4):
    """
    Aplica segmentación SLIC solo dentro del área de agua.
    
    SLIC = Simple Linear Iterative Clustering
    
    Algoritmo: Itera sobre la imagen dividiendo en superpíxeles coherentes
    (parches donde los píxeles son similares en color y cercanos espacialmente).
    
    Parámetros
    ----------
    imagen_bgr : np.ndarray
        Imagen original en BGR (de cv2.imread).
    mascara_agua : np.ndarray
        Máscara binaria del agua (255=agua, 0=no-agua).
        Puede venir de Etapa 1 (HSV u otro método).
    n_segments : int
        Número de superpíxeles deseados (~300 es balance entre detalle y velocidad).
        - Valores bajos (100): superpíxeles grandes, menos detalle
        - Valores altos (500): superpíxeles pequeños, más detalle pero más ruido
    compactness : int
        Qué tan compactos son los superpíxeles (10 es balance, rango típico 1-20).
        - Valores bajos (1-5): superpíxeles pueden ser muy irregulares
        - Valores altos (15-20): superpíxeles casi cuadrados, poco detalle
    downscale : int
        Factor de reducción antes de aplicar SLIC (4 = procesa a 1/4 del tamaño).
        Reduce ruido y acelera cálculo.
    
    Retorna
    -------
    dict
        Diccionario con:
        - 'superpixels': array donde cada píxel tiene ID del superpíxel (0-n_segments)
        - 'superpixels_original': superpíxeles en resolución original
        - 'centroides': dict {id_superpixel: (x, y)} centroide de cada superpíxel
        - 'n_superpixels': número real de superpíxeles creados
    """
    
    print(f"Aplicando SLIC (n_segments={n_segments}, compactness={compactness})...")
    
    altura_orig, ancho_orig = imagen_bgr.shape[:2]
    
    # Downscale para acelerar
    imagen_small = cv2.resize(
        imagen_bgr,
        (ancho_orig // downscale, altura_orig // downscale),
        interpolation=cv2.INTER_AREA
    )
    mascara_small = cv2.resize(
        mascara_agua,
        (ancho_orig // downscale, altura_orig // downscale),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Convertir a float [0, 1] para SLIC
    imagen_float = img_as_float(imagen_small)
    
    # Aplicar SLIC a toda la imagen (no solo agua)
    superpixels = slic(
        imagen_float,
        n_segments=n_segments,
        compactness=compactness,
        start_label=1  # Comienza IDs en 1, 0 es background
    )
    
    # Enmascarar: marcar superpíxeles fuera del agua como 0
    superpixels[mascara_small == 0] = 0
    
    # Restaurar a resolución original
    superpixels_original = cv2.resize(
        superpixels.astype(np.float32),
        (ancho_orig, altura_orig),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)
    
    # Calcular centroides de cada superpíxel
    centroides = calcular_centroides(superpixels_original, mascara_agua)
    
    n_superpix_reales = len([sp for sp in np.unique(superpixels_original) if sp > 0])
    
    print(f"✓ SLIC completado")
    print(f"  Superpíxeles creados: {n_superpix_reales}")
    print(f"  Tamaño promedio: {(mascara_agua > 0).sum() / max(n_superpix_reales, 1):.0f} píxeles/superpíxel")
    
    return {
        'superpixels': superpixels,
        'superpixels_original': superpixels_original,
        'centroides': centroides,
        'n_superpixels': n_superpix_reales,
        'imagen_original': imagen_bgr,
        'mascara_agua': mascara_agua
    }

# ================================================================
# FUNCIÓN AUXILIAR: CALCULAR CENTROIDES
# ================================================================

def calcular_centroides(superpixels, mascara_agua):
    """
    Calcula el centroide (x, y) de cada superpíxel.
    
    Retorna
    -------
    dict
        {id_superpixel: (x, y)} en coordenadas de la imagen original
    """
    centroides = {}
    
    for sp_id in np.unique(superpixels):
        if sp_id == 0:  # Skip background
            continue
        
        # Encontrar todos los píxeles de este superpíxel
        y_coords, x_coords = np.where(superpixels == sp_id)
        
        if len(x_coords) > 0:
            x_centroide = np.mean(x_coords)
            y_centroide = np.mean(y_coords)
            centroides[int(sp_id)] = (x_centroide, y_centroide)
    
    return centroides

# ================================================================
# FUNCIÓN: VISUALIZAR SUPERPÍXELES
# ================================================================

def visualizar_superpixeles(resultado_slic, figsize=(16, 6)):
    """
    Visualiza la segmentación en superpíxeles.
    
    Parámetros
    ----------
    resultado_slic : dict
        Diccionario retornado por aplicar_slic().
    figsize : tuple
        Tamaño de la figura (ancho, alto) en inches.
    
    Retorna
    -------
    fig, axes
        Figura y ejes de matplotlib.
    """
    
    imagen = resultado_slic['imagen_original']
    superpixels = resultado_slic['superpixels_original']
    centroides = resultado_slic['centroides']
    mascara_agua = resultado_slic['mascara_agua']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Subplot 1: Imagen original
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    axes[0].imshow(imagen_rgb)
    axes[0].set_title('Imagen original')
    axes[0].axis('off')
    
    # Subplot 2: Máscara de agua
    axes[1].imshow(mascara_agua, cmap='gray')
    axes[1].set_title('Máscara de agua (Etapa 1)')
    axes[1].axis('off')
    
    # Subplot 3: Superpíxeles
    superpixels_colored = colorear_superpixeles(superpixels)
    axes[2].imshow(superpixels_colored)
    
    # Dibujar centroides
    for sp_id, (x, y) in centroides.items():
        axes[2].plot(x, y, 'r.', markersize=4)
    
    axes[2].set_title(f'Superpíxeles SLIC (n={len(centroides)})')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig, axes

# ================================================================
# FUNCIÓN AUXILIAR: COLOREAR SUPERPÍXELES
# ================================================================

def colorear_superpixeles(superpixels, max_colors=256):
    """
    Asigna colores aleatorios a cada superpíxel para visualización.
    
    Parámetros
    ----------
    superpixels : np.ndarray
        Array con IDs de superpíxeles.
    max_colors : int
        Número máximo de colores distintos a usar.
    
    Retorna
    -------
    np.ndarray
        Imagen RGB donde cada superpíxel tiene un color único.
    """
    
    altura, ancho = superpixels.shape
    imagen_colored = np.zeros((altura, ancho, 3), dtype=np.uint8)
    
    # Mapeo de ID -> color aleatorio
    colores = {}
    np.random.seed(42)  # Para reproducibilidad
    
    for sp_id in np.unique(superpixels):
        if sp_id == 0:  # Background (no-agua) en negro
            colores[sp_id] = (0, 0, 0)
        else:
            colores[sp_id] = tuple(np.random.randint(0, 256, 3))
    
    # Asignar colores
    for sp_id, color in colores.items():
        imagen_colored[superpixels == sp_id] = color
    
    return imagen_colored

# ================================================================
# FUNCIÓN: EXTRAER PÍXELES DE UN SUPERPÍXEL
# ================================================================

def obtener_pixeles_superpixel(imagen, superpixels, sp_id):
    """
    Extrae todos los píxeles de un superpíxel específico.
    
    Parámetros
    ----------
    imagen : np.ndarray
        Imagen original (cualquier número de canales).
    superpixels : np.ndarray
        Array de IDs de superpíxeles.
    sp_id : int
        ID del superpíxel a extraer.
    
    Retorna
    -------
    np.ndarray
        Array de píxeles del superpíxel (n_pixeles, n_canales).
    """
    
    mask = (superpixels == sp_id)
    return imagen[mask]

# ================================================================
# FUNCIÓN: ESTADÍSTICAS DE SUPERPÍXEL
# ================================================================

def estadisticas_superpixel(imagen, superpixels, sp_id):
    """
    Calcula estadísticas básicas de un superpíxel.
    
    Retorna
    -------
    dict
        Diccionario con media, std, min, max por canal.
    """
    
    pixeles = obtener_pixeles_superpixel(imagen, superpixels, sp_id)
    
    return {
        'n_pixeles': len(pixeles),
        'media': pixeles.mean(axis=0),
        'std': pixeles.std(axis=0),
        'min': pixeles.min(axis=0),
        'max': pixeles.max(axis=0),
    }

# ================================================================
# FUNCIÓN: EXPORTAR INFORMACIÓN DE SUPERPÍXELES
# ================================================================

def exportar_superpixeles_csv(resultado_slic, imagen_hsv, imagen_lab, output_path):
    """
    Exporta información de cada superpíxel a CSV para análisis posterior.
    
    Para cada superpíxel calcula: posición, tamaño, color promedio (HSV, Lab), etc.
    
    Parámetros
    ----------
    resultado_slic : dict
        Diccionario de aplicar_slic().
    imagen_hsv : np.ndarray
        Imagen en espacio HSV.
    imagen_lab : np.ndarray
        Imagen en espacio Lab.
    output_path : str
        Ruta de salida del CSV.
    """
    
    import pandas as pd
    
    superpixels = resultado_slic['superpixels_original']
    imagen_bgr = resultado_slic['imagen_original']
    centroides = resultado_slic['centroides']
    
    datos = []
    
    for sp_id in np.unique(superpixels):
        if sp_id == 0:
            continue
        
        mask = (superpixels == sp_id)
        n_pixeles = mask.sum()
        
        # Color en BGR
        bgr_pixels = imagen_bgr[mask]
        b_mean, g_mean, r_mean = bgr_pixels.mean(axis=0)
        
        # Color en HSV
        hsv_pixels = imagen_hsv[mask]
        h_mean, s_mean, v_mean = hsv_pixels.mean(axis=0)
        
        # Color en Lab
        lab_pixels = imagen_lab[mask]
        l_mean, a_mean, b_mean_lab = lab_pixels.mean(axis=0)
        
        # Posición
        x, y = centroides[sp_id]
        
        datos.append({
            'superpixel_id': sp_id,
            'n_pixeles': n_pixeles,
            'centroide_x': x,
            'centroide_y': y,
            'bgr_b': b_mean,
            'bgr_g': g_mean,
            'bgr_r': r_mean,
            'hsv_h': h_mean,
            'hsv_s': s_mean,
            'hsv_v': v_mean,
            'lab_l': l_mean,
            'lab_a': a_mean,
            'lab_b': b_mean_lab,
        })
    
    df = pd.DataFrame(datos)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Datos exportados: {output_path}")
    print(f"  Total de superpíxeles: {len(df)}")
    
    return df

# ================================================================
# EJEMPLO DE USO
# ================================================================

if __name__ == '__main__':
    """
    Ejemplo mínimo de uso de Etapa 2.
    
    Requiere: imagen y máscara de agua de Etapa 1.
    """
    
    print("="*80)
    print("ETAPA 2 — SEGMENTACIÓN INTERNA DEL AGUA CON SUPERPÍXELES SLIC")
    print("="*80)
    print()
    
    # Crear datos de ejemplo (en tu caso, cargarías imagen y máscara reales)
    print("Nota: Este es un ejemplo de uso.")
    print("En tu notebook, usa:")
    print()
    print("  from etapa_2_superpixeles_slic import aplicar_slic, visualizar_superpixeles")
    print("  ")
    print("  resultado = aplicar_slic(")
    print("      imagen_bgr,          # De cv2.imread()")
    print("      mascara_agua,        # De Etapa 1")
    print("      n_segments=300,")
    print("      compactness=10")
    print("  )")
    print("  ")
    print("  fig, axes = visualizar_superpixeles(resultado)")
    print("  plt.show()")
    print()
    print("="*80)