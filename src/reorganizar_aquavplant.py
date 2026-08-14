"""
Código pre-trabajo: Organización de la data que descargue
Borrable
"""

import os
import shutil
from pathlib import Path

def reorganizar_aquavplant(carpeta_base):
    """
    Transforma:
      Shapla_Bil_1/Image4/Image4.jpg
      Shapla_Bil_1/Image4/Image4_binaryMask.png
      Shapla_Bil_1/Image4/Image4_multiclassMask.png
    
    En:
      Shapla_Bil_1/images/Image4.jpg
      Shapla_Bil_1/masks_binary/Image4_binaryMask.png
      Shapla_Bil_1/masks_multiclass/Image4_multiclassMask.png
    """
    
    for sitio in os.listdir(carpeta_base):
        ruta_sitio = os.path.join(carpeta_base, sitio)
        if not os.path.isdir(ruta_sitio):
            continue
        
        print(f"Reorganizando {sitio}...")
        
        # Crea las subcarpetas de destino
        ruta_images = os.path.join(ruta_sitio, 'images')
        ruta_masks_binary = os.path.join(ruta_sitio, 'masks_binary')
        ruta_masks_multiclass = os.path.join(ruta_sitio, 'masks_multiclass')
        
        os.makedirs(ruta_images, exist_ok=True)
        os.makedirs(ruta_masks_binary, exist_ok=True)
        os.makedirs(ruta_masks_multiclass, exist_ok=True)
        
        # Itera sobre cada carpeta Image1, Image2, etc.
        for carpeta_imagen in os.listdir(ruta_sitio):
            ruta_carpeta_imagen = os.path.join(ruta_sitio, carpeta_imagen)
            
            # Solo procesa si es una carpeta (no los directorios que acabamos de crear)
            if not os.path.isdir(ruta_carpeta_imagen) or carpeta_imagen in ['images', 'masks_binary', 'masks_multiclass']:
                continue
            
            # Busca los archivos dentro
            for archivo in os.listdir(ruta_carpeta_imagen):
                ruta_origen = os.path.join(ruta_carpeta_imagen, archivo)
                
                if archivo.endswith('.jpg') or archivo.endswith('.jpeg'):
                    # Mueve imagen JPG a images/
                    ruta_destino = os.path.join(ruta_images, archivo)
                    shutil.move(ruta_origen, ruta_destino)
                    print(f"  {archivo} → images/")
                
                elif 'binaryMask' in archivo:
                    # Mueve máscara binaria
                    ruta_destino = os.path.join(ruta_masks_binary, archivo)
                    shutil.move(ruta_origen, ruta_destino)
                    print(f"  {archivo} → masks_binary/")
                
                elif 'multiclassMask' in archivo or 'multiclass' in archivo:
                    # Mueve máscara multiclase
                    ruta_destino = os.path.join(ruta_masks_multiclass, archivo)
                    shutil.move(ruta_origen, ruta_destino)
                    print(f"  {archivo} → masks_multiclass/")
            
            # Elimina la carpeta Image1/Image2/etc. ya vacía
            try:
                os.rmdir(ruta_carpeta_imagen)
                print(f"  Eliminada carpeta vacía: {carpeta_imagen}/")
            except OSError:
                print(f"  ⚠️  No se pudo eliminar {carpeta_imagen}/ (no está vacía)")
        
        print(f"✓ {sitio} reorganizado\n")

# Ejecuta
reorganizar_aquavplant('AqUavplant')