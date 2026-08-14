import pandas as pd
import re
from pathlib import Path

# Lee el CSV
df = pd.read_csv(r'D:\proyecto_eutrofizacion\data\metadata\river_water_index.csv')

print(f"Antes: {len(df)} filas")
print(df['filepath'].head(20))

# Extrae el número del filename (DJI_1234.JPG → 1234)
def extraer_numero(filepath):
    """Extrae el número de DJI_XXXX.JPG"""
    filename = Path(filepath).name  # obtiene "DJI_1234.JPG"
    match = re.search(r'DJI_(\d+)', filename)
    if match:
        return int(match.group(1))
    return float('inf')  # Si no encuentra número, va al final

# Crea columna temporal con el número
df['numero_imagen'] = df['filepath'].apply(extraer_numero)

# Ordena por grupo (año) y luego por número
df = df.sort_values(['group', 'numero_imagen']).reset_index(drop=True)

# Elimina la columna temporal
df = df.drop('numero_imagen', axis=1)

# Guarda el CSV ordenado correctamente
df.to_csv('river_water_index.csv', index=False)

print(f"Después: {len(df)} filas (ordenadas correctamente)")
print(df['filepath'].head(20))