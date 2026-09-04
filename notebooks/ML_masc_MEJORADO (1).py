"""
MACHINE LEARNING PARA SEGMENTACIÓN DE AGUA - VERSIÓN MEJORADA

Cambios principales:
1. Mejor manejo de errores y validaciones
2. Normalización de features
3. Downscale optimizado para evitar memoria
4. Mejor logging y debugging
5. Validación de ground truth antes de procesar
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# ================================================================
# CONFIGURACIÓN
# ================================================================

os.chdir(r'D:\proyecto_eutrofizacion')
PROJECT_ROOT = os.getcwd()

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
EXTERNAL_DATA = os.path.join(DATA_DIR, 'raw', 'external')
RIVER_DATA = os.path.join(EXTERNAL_DATA, 'river_water_dataset')
INDEX_CSV = os.path.join(METADATA_DIR, 'river_water_index.csv')
MODEL_OUTPUT = os.path.join(MODELS_DIR, 'water_segmenter_rf.pkl')
SCALER_OUTPUT = os.path.join(MODELS_DIR, 'feature_scaler.pkl')

os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("="*80)
print("MACHINE LEARNING - SEGMENTACIÓN DE AGUA (VERSIÓN MEJORADA)")
print("="*80)
print(f"\n✓ Directorio de trabajo: {PROJECT_ROOT}")
print(f"✓ Datos: {RIVER_DATA}")
print(f"✓ Modelos: {MODELS_DIR}\n")

# ================================================================
# FUNCIÓN 1: CARGAR MÁSCARA DESDE JSON
# ================================================================

def cargar_mascara_desde_labelimg(ruta_json, alto_img, ancho_img):
    """
    Carga máscara de segmentación desde JSON (LabelImg).
    Retorna None si hay error.
    """
    mascara = np.zeros((alto_img, ancho_img), dtype=np.uint8)
    
    if not os.path.isfile(ruta_json):
        return None
    
    try:
        with open(ruta_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
    
    for shape in data.get('shapes', []):
        if shape.get('label', '').lower() == 'water':
            points = np.array(shape['points'], dtype=np.int32)
            if len(points) > 0:
                cv2.fillPoly(mascara, [points], 255)
    
    return mascara if (mascara > 0).sum() > 0 else None

# ================================================================
# FUNCIÓN 2: EXTRAER FEATURES POR PÍXEL
# ================================================================

def extraer_features_pixel(imagen_bgr):
    """
    Extrae 6 features por píxel: HSV + Lab + textura local.
    
    IMPORTANTE: Los features deben ser idénticos a los usados en entrenamiento.
    """
    # Convertir espacios de color
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Calcular textura local: desviación estándar en ventana 5x5
    media_local = cv2.blur(gray, (5, 5))
    media_sq_local = cv2.blur(gray**2, (5, 5))
    textura = np.sqrt(np.abs(media_sq_local - media_local**2))
    
    # Apilar features: (H, S, V, a*, b*, textura)
    features = np.stack([
        hsv[:, :, 0],  # H
        hsv[:, :, 1],  # S
        hsv[:, :, 2],  # V
        lab[:, :, 1],  # a*
        lab[:, :, 2],  # b*
        textura        # Textura local
    ], axis=-1)
    
    # Remodelar a (n_pixeles, 6)
    return features.reshape(-1, 6)

# ================================================================
# FUNCIÓN 3: PREPARAR DATASET DE ENTRENAMIENTO
# ================================================================

def preparar_dataset_entrenamiento(
    indice_csv=None,
    n_imagenes=150,
    muestreo_pixeles=1000,
    downscale=4
):
    """
    Procesa imágenes de RivAIrSet, extrae features y submuestrea píxeles.
    
    MEJORADO: Validaciones más estrictas, mejor manejo de errores.
    """
    
    if indice_csv is None:
        indice_csv = INDEX_CSV
    
    if not os.path.isfile(indice_csv):
        print(f"❌ No existe índice: {indice_csv}")
        return None, None
    
    # Cargar índice
    df = pd.read_csv(indice_csv)
    df['filepath'] = df['filepath'].str.replace('\\', '/')
    df['segmentation_mask_path'] = df['segmentation_mask_path'].str.replace('\\', '/')
    
    df_validas = df[df['filepath'].notna() & df['segmentation_mask_path'].notna()].copy()
    
    if len(df_validas) == 0:
        print("❌ No hay imágenes válidas en el índice")
        return None, None
    
    # Submuestrear imágenes
    n_imagenes = min(n_imagenes, len(df_validas))
    df_sample = df_validas.sample(n=n_imagenes, random_state=42)
    
    print(f"Procesando {len(df_sample)} imágenes...")
    
    X_all, y_all = [], []
    images_procesadas = 0
    
    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        image_path = row['filepath']
        label_path = row['segmentation_mask_path']
        
        # Cargar imagen
        img = cv2.imread(image_path)
        if img is None:
            continue
        
        alto_orig, ancho_orig = img.shape[:2]
        
        # Cargar ground truth
        mask_gt = cargar_mascara_desde_labelimg(label_path, alto_orig, ancho_orig)
        if mask_gt is None:
            continue
        
        # Downscale
        img_small = cv2.resize(
            img,
            (ancho_orig // downscale, alto_orig // downscale),
            interpolation=cv2.INTER_AREA
        )
        mask_small = cv2.resize(
            mask_gt,
            (ancho_orig // downscale, alto_orig // downscale),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Extraer features
        features = extraer_features_pixel(img_small)
        labels = (mask_small.flatten() > 0).astype(int)
        
        # Submuestrear píxeles: balancea agua vs no-agua
        idx_agua = np.where(labels == 1)[0]
        idx_no_agua = np.where(labels == 0)[0]
        
        n_muestra = min(
            muestreo_pixeles // 2,
            len(idx_agua),
            len(idx_no_agua)
        )
        
        if n_muestra == 0:
            continue
        
        # Seleccionar índices equilibrados
        idx_seleccionados = np.concatenate([
            np.random.choice(idx_agua, n_muestra, replace=False),
            np.random.choice(idx_no_agua, n_muestra, replace=False)
        ])
        
        X_all.append(features[idx_seleccionados])
        y_all.append(labels[idx_seleccionados])
        
        images_procesadas += 1
    
    if len(X_all) == 0:
        print("❌ No se pudieron procesar imágenes")
        return None, None
    
    # Concatenar todos los features
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    
    print(f"\n✓ Dataset preparado:")
    print(f"  Imágenes: {images_procesadas}")
    print(f"  Píxeles totales: {len(X):,}")
    print(f"  Agua: {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.1f}%)")
    print(f"  No-agua: {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.1f}%)")
    
    return X, y

# ================================================================
# FUNCIÓN 4: ENTRENAR MODELO
# ================================================================

def entrenar_modelo(X, y, n_estimators=100, max_depth=15):
    """
    Entrena Random Forest con normalización de features.
    
    MEJORADO: Incluye StandardScaler para normalizar features.
    """
    
    print("\nDividiendo datos (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"  Train: {len(X_train):,} ({(y_train == 1).sum():,} agua)")
    print(f"  Test: {len(X_test):,} ({(y_test == 1).sum():,} agua)")
    
    # NOVEDAD: Normalizar features
    print("\nNormalizando features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Entrenando Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    clf.fit(X_train_scaled, y_train)
    print("✓ Entrenamiento completado")
    
    # Predicciones
    y_pred_train = clf.predict(X_train_scaled)
    y_pred_test = clf.predict(X_test_scaled)
    
    # Métricas
    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)
    
    print(f"\n✓ Modelo entrenado:")
    print(f"  Train F1: {f1_train:.4f}")
    print(f"  Test F1: {f1_test:.4f}")
    
    print("\nReporte de clasificación (Test set):")
    print(classification_report(y_test, y_pred_test, target_names=['no_agua', 'agua']))
    
    # Feature importance
    print("\nImportancia de features:")
    feature_names = ['H (Hue)', 'S (Sat)', 'V (Brillo)', 'a* (Lab)', 'b* (Lab)', 'Textura']
    for name, imp in zip(feature_names, clf.feature_importances_):
        print(f"  {name:15s}: {imp:.4f} ({imp*100:.1f}%)")
    
    return {
        'modelo': clf,
        'scaler': scaler,
        'X_train': X_train_scaled, 'y_train': y_train,
        'X_test': X_test_scaled, 'y_test': y_test,
        'y_pred_test': y_pred_test,
        'f1_train': f1_train,
        'f1_test': f1_test,
        'feature_importances': clf.feature_importances_,
    }

# ================================================================
# FUNCIÓN 5: GUARDAR MODELO Y SCALER
# ================================================================

def guardar_modelo(resultado, model_path=None, scaler_path=None):
    """
    Guarda modelo y scaler (IMPORTANTE PARA USAR DESPUÉS).
    """
    
    if model_path is None:
        model_path = MODEL_OUTPUT
    if scaler_path is None:
        scaler_path = SCALER_OUTPUT
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Guardar modelo
    with open(model_path, 'wb') as f:
        pickle.dump(resultado['modelo'], f)
    print(f"✓ Modelo guardado: {model_path}")
    
    # Guardar scaler (CRUCIAL)
    with open(scaler_path, 'wb') as f:
        pickle.dump(resultado['scaler'], f)
    print(f"✓ Scaler guardado: {scaler_path}")

# ================================================================
# FUNCIÓN 6: USAR MODELO ENTRENADO (CON NORMALIZACIÓN)
# ================================================================

def segmentar_imagen_con_modelo(ruta_imagen, modelo=None, scaler=None, downscale=4):
    """
    Aplica el modelo Random Forest entrenado a una imagen nueva.
    
    MEJORADO: Usa el scaler para normalizar features igual que en entrenamiento.
    """
    
    if modelo is None or scaler is None:
        # Cargar modelo y scaler
        if not os.path.isfile(MODEL_OUTPUT) or not os.path.isfile(SCALER_OUTPUT):
            print(f"❌ Modelo o scaler no encontrados")
            return None
        
        with open(MODEL_OUTPUT, 'rb') as f:
            modelo = pickle.load(f)
        with open(SCALER_OUTPUT, 'rb') as f:
            scaler = pickle.load(f)
    
    # Cargar imagen
    img = cv2.imread(ruta_imagen)
    if img is None:
        print(f"❌ No se pudo cargar: {ruta_imagen}")
        return None
    
    altura_orig, ancho_orig = img.shape[:2]
    print(f"Imagen cargada: {ancho_orig}×{altura_orig}")
    
    # Downscale
    img_small = cv2.resize(
        img,
        (ancho_orig // downscale, altura_orig // downscale),
        interpolation=cv2.INTER_AREA
    )
    
    # Extraer features
    print("Extrayendo features...")
    features = extraer_features_pixel(img_small)
    
    # CRUCIAL: Normalizar con el scaler
    print("Normalizando features...")
    features_scaled = scaler.transform(features)
    
    # Predicción
    print("Prediciendo...")
    predicciones = modelo.predict(features_scaled)
    
    # Remodelar a imagen
    mascara = predicciones.reshape(img_small.shape[:2]).astype(np.uint8) * 255
    
    # Restaurar resolución original
    mascara_final = cv2.resize(
        mascara,
        (ancho_orig, altura_orig),
        interpolation=cv2.INTER_NEAREST
    )
    
    print(f"✓ Predicción completada")
    print(f"  Agua detectada: {(mascara_final > 0).sum() / mascara_final.size * 100:.1f}%")
    
    return {
        'imagen_original': img,
        'mascara_predicha': mascara_final,
        'altura_orig': altura_orig,
        'ancho_orig': ancho_orig
    }

# ================================================================
# FUNCIÓN 7: VISUALIZAR RESULTADOS
# ================================================================

def visualizar_resultados(
    ruta_imagen,
    ruta_ground_truth_json,
    resultado_prediccion,
    figsize=(18, 6)
):
    """
    Visualiza lado a lado: original, ground truth, predicción.
    """
    
    # Cargar ground truth
    altura = resultado_prediccion['altura_orig']
    ancho = resultado_prediccion['ancho_orig']
    mascara_gt = cargar_mascara_desde_labelimg(ruta_ground_truth_json, altura, ancho)
    
    # Imagen original en RGB
    imagen_rgb = cv2.cvtColor(resultado_prediccion['imagen_original'], cv2.COLOR_BGR2RGB)
    
    # Crear figura
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Subplot 1: Original
    axes[0].imshow(imagen_rgb)
    axes[0].set_title('Imagen Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Subplot 2: Ground truth
    if mascara_gt is not None:
        axes[1].imshow(mascara_gt, cmap='gray')
        axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'No hay GT', ha='center', va='center')
        axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Subplot 3: Predicción
    axes[2].imshow(resultado_prediccion['mascara_predicha'], cmap='gray')
    axes[2].set_title('Predicción RF', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

# ================================================================
# MAIN: EJECUTAR PIPELINE COMPLETO
# ================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("PASO 1: Preparar dataset")
    print("="*80)
    X, y = preparar_dataset_entrenamiento(n_imagenes=150, downscale=4)
    
    if X is None:
        print("❌ Error en preparación de datos")
        exit(1)
    
    print("\n" + "="*80)
    print("PASO 2: Entrenar modelo")
    print("="*80)
    resultado = entrenar_modelo(X, y)
    
    print("\n" + "="*80)
    print("PASO 3: Guardar modelo y scaler")
    print("="*80)
    guardar_modelo(resultado)
    
    print("\n" + "="*80)
    print("✓ ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print("\nPara usar en nuevas imágenes:")
    print("  resultado = segmentar_imagen_con_modelo('ruta/imagen.jpg')")
    print("  fig = visualizar_resultados('ruta/imagen.jpg', 'ruta/gt.json', resultado)")
    print("  plt.show()")
