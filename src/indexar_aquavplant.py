"""
Código pre-trabajo: Organización de la data que descargue
Borrable
"""

def indexar_aquavplant(carpeta_base, salida_csv):
    """
    AqUavplant: sitios geográficos + máscaras de plantas acuáticas (multiclase).
    
    IMPORTANTE: A diferencia de RivAIrSet (que solo detecta "agua sí/no"),
    AqUavplant detecta especies de plantas acuáticas específicas.
    Las plantas acuáticas densas son indicador visual de eutrofización.
    """
    filas = []
    
    for sitio in os.listdir(carpeta_base):
        ruta_sitio = os.path.join(carpeta_base, sitio)
        if not os.path.isdir(ruta_sitio):
            continue
        
        ruta_images = os.path.join(ruta_sitio, 'images')
        ruta_masks_binary = os.path.join(ruta_sitio, 'masks_binary')
        ruta_masks_multiclass = os.path.join(ruta_sitio, 'masks_multiclass')
        
        # Si las carpetas no existen (por si aún no corriste el reorganizador), sáltalo
        if not all(os.path.isdir(r) for r in [ruta_images, ruta_masks_binary, ruta_masks_multiclass]):
            print(f"⚠️  {sitio} no tiene estructura reorganizada, saltando...")
            continue
        
        for archivo in os.listdir(ruta_images):
            if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                nombre_sin_ext = os.path.splitext(archivo)[0]
                
                # Busca las máscaras correspondientes
                ruta_mask_binary = os.path.join(ruta_masks_binary, f"{nombre_sin_ext}_binaryMask.png")
                ruta_mask_multiclass = os.path.join(ruta_masks_multiclass, f"{nombre_sin_ext}_multiclassMask.png")
                
                has_binary = os.path.exists(ruta_mask_binary)
                has_multiclass = os.path.exists(ruta_mask_multiclass)
                
                filas.append({
                    'filepath': os.path.join(ruta_images, archivo),
                    'source': 'aquavplant',
                    'group': sitio,
                    'has_gps': False,
                    'site': sitio,
                    'mask_binary_path': ruta_mask_binary if has_binary else None,
                    'mask_multiclass_path': ruta_mask_multiclass if has_multiclass else None,
                    'mask_type': 'aquatic_plants_multiclass',
                    'mask_format': 'png_indexed',
                    'num_classes': 31  # 31 especies de plantas acuáticas
                })
    
    with open(salida_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'filepath', 'source', 'group', 'has_gps', 'site',
            'mask_binary_path', 'mask_multiclass_path', 'mask_type', 'mask_format', 'num_classes'
        ])
        writer.writeheader()
        writer.writerows(filas)
    
    imágenes_con_masks = sum(1 for f in filas if f['mask_multiclass_path'])
    print(f"✓ Indexadas {len(filas)} imágenes de AqUavplant")
    print(f"  Con máscaras multiclase de plantas acuáticas: {imágenes_con_masks}")
    return filas