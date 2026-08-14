import cv2
import numpy as np

def conectar_graffiti_y_cerrar(mascara, skeleton_kernel=3, dilation_kernel=30):
    """
    Conecta los píxeles negros (graffiti) fragmentados en líneas continuas.
    
    ESTRATEGIA:
    1. Aplica Morphological Skeleton en píxeles NEGROS
       (obtiene líneas finas de graffiti sin engrosar)
    2. Dilata ligeramente para conectar gaps pequeños
    3. Resultado: Líneas continuas que delimitan el agua
    
    Parámetros:
    -----------
    skeleton_kernel : int
        Tamaño del kernel para skeleton (3-5)
    dilation_kernel : int
        Tamaño del kernel para dilatar negro (3-5)
    
    Retorna:
    --------
    np.ndarray : Máscara con graffiti conectado
    """
    
    #Invierte la máscara (para trabajar con píxeles negros)
    mascara_inv = cv2.bitwise_not(mascara)
    
    #Aplica skeleton al negro para obtener líneas finas
    # Skeleton = líneas delgadas que representan la forma
    kernel_skeleton = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (skeleton_kernel, skeleton_kernel)
    )
    
    skeleton = cv2.morphologyEx(
        mascara_inv,
        cv2.MORPH_ERODE,
        kernel_skeleton,
        iterations=1
    )
    
    #Dilata el skeleton ligeramente para conectar huecos
    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilation_kernel, dilation_kernel)
    )
    
    negro_conectado = cv2.dilate(
        skeleton,
        kernel_dilate,
        iterations=2
    )
    
    #Vuelve a invertir
    mascara_final = cv2.bitwise_not(negro_conectado)
    
    return mascara_final

def suavizar_bordes_contorno(mascara, contour_approx=7):
    """
    Suaviza los bordes dentados de la máscara.
    
    1. Encuentra el contorno principal
    2. Aproxima el contorno a una curva suave (Ramer-Douglas-Peucker)
    3. Redibuuja el contorno suavizado
    """
    
    # Encuentra contorno principal
    contours, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return mascara
    
    contorno_mayor = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contorno_mayor)
    perimetro = cv2.arcLength(contorno_mayor, True)
    
    # Aproxima el contorno a una curva suave
    epsilon = contour_approx  # Adjust based on smoothness desired
    contorno_suave = cv2.approxPolyDP(
        contorno_mayor,
        epsilon,
        True  # closed contour
    )
    
    # Redibuја SOLO el contorno suavizado
    mascara_suave = np.zeros_like(mascara)
    cv2.drawContours(
        mascara_suave,
        [contorno_suave],
        0,
        255,
        -1  # Rellena
    )
    
    return mascara_suave