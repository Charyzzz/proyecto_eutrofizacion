import streamlit as st
import textwrap
import tempfile
import shutil
import pandas as pd
from pathlib import Path

import app_pipeline as pipe


# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Water Anomaly Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# INICIALIZAR SESSION STATE
# ============================================================

if 'resultados_imagenes' not in st.session_state:
    st.session_state.resultados_imagenes = None

if 'df_superpixeles' not in st.session_state:
    st.session_state.df_superpixeles = None

if 'modo_resultado' not in st.session_state:
    st.session_state.modo_resultado = None

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0


# ============================================================
# CARGA DE MODELOS (cacheado -- solo se carga una vez por sesión
# de servidor, no en cada rerun/click)
# ============================================================

@st.cache_resource(show_spinner="Cargando la U-Net...")
def cargar_unet_cacheado():
    return pipe.cargar_unet()


# ============================================================
# CONSTANTES
# ============================================================

MAX_TOTAL_BYTES = 20* 1024 ** 3       # 20 GB
MAX_IMAGES = 2500


# ============================================================
# FUNCIÓN PARA FORMATEAR TAMAÑOS
# ============================================================

def format_size(size_bytes):

    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"

    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"

    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    textwrap.dedent(
        """
        <style>

        /* =====================================================
           FONDO
           ===================================================== */

        .stApp {
            background-color: #f3f5f7;
        }

        .main .block-container {
            max-width: 1250px;
            padding-top: 25px;
            padding-bottom: 50px;
        }


        /* =====================================================
           OCULTAR ELEMENTOS DE STREAMLIT
           ===================================================== */

        #MainMenu {
            visibility: hidden;
        }

        header {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }


        /* =====================================================
           TÍTULO
           ===================================================== */

        .main-title {
            font-size: 42px;
            font-weight: 700;
            color: #273449;
            line-height: 1.12;
            letter-spacing: -0.8px;
            margin-bottom: 12px;
        }

        .subtitle {
            font-size: 16px;
            line-height: 1.6;
            color: #687589;
            max-width: 720px;
        }


        /* =====================================================
           CONTENEDORES
           ===================================================== */

        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: white;
            border: 1px solid #e2e7ec;
            border-radius: 18px;
            box-shadow: 0 4px 18px rgba(30, 45, 65, 0.05);
            padding: 20px;
        }


        /* =====================================================
           TÍTULOS DE SECCIÓN
           ===================================================== */

        .section-title {
            font-size: 21px;
            font-weight: 650;
            color: #29374a;
            margin-bottom: 5px;
        }

        .section-description {
            font-size: 15px;
            color: #788596;
            margin-bottom: 15px;
        }


        /* =====================================================
           ICONO
           ===================================================== */

        .section-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;

            width: 42px;
            height: 42px;

            background-color: #edf3fa;
            border-radius: 50%;

            margin-right: 8px;

            font-size: 20px;
        }


        /* =====================================================
           UPLOADER
           ===================================================== */

        [data-testid="stFileUploader"] {
            background-color: #fafbfd;
            border: 2px dashed #d7dfe8;
            border-radius: 14px;
            padding: 14px;
            margin-top: 18px;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #aebdce;
            background-color: #f8fafc;
        }

        [data-testid="stFileUploader"] button {
            border-radius: 8px;
            border: 1px solid #d6dde5;
            background-color: white;
        }


        /* =====================================================
           CONTADOR DE ARCHIVOS
           ===================================================== */

        .file-counter {
            background-color: #f7f9fb;
            border: 1px solid #e3e8ee;
            border-radius: 12px;
            padding: 15px 19px;
            margin-top: 18px;
            color: #566477;
            font-size: 15px;
        }

        .file-number {
            color: #3678c5;
            font-size: 19px;
            font-weight: 700;
        }


        /* =====================================================
           ALMACENAMIENTO
           ===================================================== */

        .storage-container {
            margin-top: 18px;
        }

        .storage-label {
            display: flex;
            justify-content: space-between;

            margin-bottom: 7px;

            font-size: 14px;
            color: #687589;
        }

        .storage-used {
            font-weight: 650;
            color: #35465d;
        }

        .storage-limit {
            color: #8a96a5;
        }


        /* =====================================================
           INFORMACIÓN
           ===================================================== */

        .info-box {
            text-align: center;
            color: #718096;
            font-size: 14px;
            margin-top: 18px;
        }


        /* =====================================================
           IMÁGENES
           ===================================================== */

        [data-testid="stImage"] img {
            border-radius: 10px;
        }


        /* =====================================================
           BOTÓN
           ===================================================== */

        .stButton > button {
            background-color: #35465d;
            color: white;

            border: none;
            border-radius: 11px;

            min-height: 52px;

            padding: 12px 28px;

            font-size: 16px;
            font-weight: 600;

            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            background-color: #29394f;
            color: white;

            transform: translateY(-1px);

            box-shadow:
                0 5px 12px rgba(30, 45, 65, 0.15);
        }


        /* =====================================================
           MENSAJE DE ANÁLISIS
           ===================================================== */

        .analysis-message {
            background-color: #eef5fb;

            border: 1px solid #d7e6f4;

            border-radius: 12px;

            padding: 16px 20px;

            color: #50657c;

            margin-top: 20px;

            text-align: center;
        }

        </style>
        """
    ),
    unsafe_allow_html=True
)


# ============================================================
# ENCABEZADO
# ============================================================

header_left, header_right = st.columns(
    [4.5, 1],
    gap="large"
)


# ============================================================
# TÍTULO
# ============================================================

with header_left:

    st.markdown(
        textwrap.dedent(
            """
            <div class="main-title">
                Detección de anomalías en<br>
                cuerpos de agua
            </div>
            """
        ),
        unsafe_allow_html=True
    )

    st.markdown(
        textwrap.dedent(
            """
            <div class="subtitle">
                Sube tus imágenes y analiza posibles anomalías en
                lagos, lagunas y ríos utilizando técnicas de
                visión por computadora y aprendizaje automático.
            </div>
            """
        ),
        unsafe_allow_html=True
    )


# ============================================================
# LOGO
# ============================================================

with header_right:

    st.image(
        "assets/logo_pucp.png",
        width=300
    )


# ============================================================
# ESPACIO
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# PANEL 1 — CARGAR IMÁGENES
# ============================================================

with st.container(border=True):

    st.markdown(
        textwrap.dedent(
            """
            <div class="section-title">
                <span class="section-icon">☁️</span>
                1. Cargar Imágenes
            </div>
            """
        ),
        unsafe_allow_html=True
    )

    st.markdown(
        textwrap.dedent(
            """
            <div class="section-description">
                Arrastra y suelta tus imágenes aquí o selecciona archivos.
            </div>
            """
        ),
        unsafe_allow_html=True
    )

    # ========================================================
    # UPLOADER
    # ========================================================

    uploaded_files = st.file_uploader(
        "Selecciona las imágenes del cuerpo de agua",

        type=[
            "jpg",
            "jpeg",
            "png"
        ],

        accept_multiple_files=True,

        label_visibility="collapsed",

        key=f"water_images_{st.session_state.uploader_key}"
    )

    if uploaded_files:
        if st.button("🗑️  Borrar imágenes cargadas"):
            st.session_state.uploader_key += 1
            st.session_state.resultados_imagenes = None
            st.session_state.df_superpixeles = None
            st.session_state.modo_resultado = None
            st.rerun()


# ============================================================
# SI SE CARGARON IMÁGENES
# ============================================================

if uploaded_files:

    number_images = len(uploaded_files)

    # ========================================================
    # COMPROBAR CANTIDAD DE IMÁGENES
    # ========================================================

    if number_images > MAX_IMAGES:

        st.error(
            f"Has seleccionado {number_images:,} imágenes. "
            f"El máximo permitido es {MAX_IMAGES:,}."
        )

        st.stop()


    # ========================================================
    # CALCULAR TAMAÑO TOTAL
    # ========================================================

    total_size = sum(
        image.size
        for image in uploaded_files
    )

    total_size_text = format_size(total_size)


    # Porcentaje utilizado del GB

    storage_percentage = (
        total_size / MAX_TOTAL_BYTES
    )

    storage_percentage = min(
        storage_percentage,
        1.0
    )


    # ========================================================
    # SI SUPERA 20 GB
    # ========================================================

    if total_size > MAX_TOTAL_BYTES:

        st.error(
            f"⚠️ El tamaño total de las imágenes es "
            f"{total_size_text}. "
            f"El límite permitido es 20 GB."
        )

        st.progress(1.0)

        st.markdown(
            textwrap.dedent(
                """
                <div class="info-box">
                    Elimina algunas imágenes y vuelve a cargarlas
                    para continuar.
                </div>
                """
            ),
            unsafe_allow_html=True
        )

        st.stop()


    # ========================================================
    # CONTADOR DE ARCHIVOS
    # ========================================================

    st.markdown(
        textwrap.dedent(
            f"""
            <div class="file-counter">
                📄 &nbsp;
                <b>Archivos cargados:</b>
                <span class="file-number">
                    {number_images:,} imágenes
                </span>
            </div>
            """
        ),
        unsafe_allow_html=True
    )


    # ========================================================
    # INDICADOR DE ALMACENAMIENTO
    # ========================================================

    storage_left, storage_right = st.columns([3, 1])

    with storage_left:
        st.markdown(
            f"**Almacenamiento utilizado: {total_size_text}**"
        )

    with storage_right:
        st.markdown(
            "**Límite: 20.00 GB**"
        )

    st.progress(
        storage_percentage
    )


    # ========================================================
    # PANEL 2 — VISTA PREVIA
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    with st.container(border=True):

        st.markdown(
            textwrap.dedent(
                """
                <div class="section-title">
                    🖼️ 2. Vista Previa de Imágenes
                </div>
                """
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            textwrap.dedent(
                f"""
                <div class="section-description">
                    Mostrando las primeras 10 imágenes de un total de
                    {number_images:,}.
                </div>
                """
            ),
            unsafe_allow_html=True
        )


        # ====================================================
        # SOLO LAS PRIMERAS 10
        # ====================================================

        preview_images = uploaded_files[:10]


        # ====================================================
        # GRID 5 × 2
        # ====================================================

        columns = st.columns(
            5,
            gap="small"
        )


        for i, image in enumerate(preview_images):

            with columns[i % 5]:

                st.image(
                    image,
                    caption=image.name,
                    use_container_width=True
                )


        # ====================================================
        # INFORMACIÓN
        # ====================================================

        if number_images > 10:
            st.markdown(
                """
                <div class="info-box">

                    Solo se muestran las primeras 10 imágenes.
                    El análisis utilizará todas las imágenes cargadas.

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # OPCIÓN DE SALTO ENTRE IMÁGENES
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">🔀 3. Opciones de muestreo</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            textwrap.dedent(
                """
                <div class="section-description">
                    Si tus imágenes se solapan o fueron tomadas casi en el
                    mismo lugar, salta algunas para no analizar tomas
                    redundantes.
                </div>
                """
            ),
            unsafe_allow_html=True
        )

        salto = st.selectbox(
            "Saltar N imágenes entre cada una analizada",
            options=[0, 1, 2, 3, 5, 10],
            index=0,
            key="salto_imagenes"
        )

    stride = salto + 1
    archivos_a_procesar = uploaded_files[::stride]
    n_a_procesar = len(archivos_a_procesar)

    st.markdown(
        textwrap.dedent(
            f"""
            <div class="info-box">
                Se analizarán <b>{n_a_procesar}</b> de {number_images:,} imágenes cargadas
                (saltando {salto} entre cada una).
            </div>
            """
        ),
        unsafe_allow_html=True
    )


    # ========================================================
    # BOTONES: ANÁLISIS o ENTRENAMIENTO
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    button_col1, button_col2 = st.columns(2)

    with button_col1:
        hacer_analisis = st.button(
            "🔎  Hacer el análisis",
            use_container_width=True,
            help="Detecta anomalías con el Isolation Forest ya entrenado (no vuelve a entrenar nada)."
        )

    with button_col2:
        entrenar_modelo = st.button(
            "🧠  Entrenar Isolation Forest",
            use_container_width=True,
            help="Ajusta (fit) un Isolation Forest nuevo usando estas imágenes como línea base, y reemplaza el modelo guardado."
        )


    # ========================================================
    # EJECUTAR EL PIPELINE (máscara -> superpíxeles -> Isolation Forest)
    # ========================================================

    if hacer_analisis or entrenar_modelo:

        modelo_guardado = None
        if hacer_analisis:
            modelo_guardado = pipe.cargar_isolation_forest()
            if modelo_guardado is None:
                st.error(
                    "Todavía no hay un Isolation Forest entrenado "
                    "(no se encontró models/isolation_forest_agua.pkl). "
                    "Usa primero **🧠 Entrenar Isolation Forest** con un "
                    "conjunto de imágenes de referencia."
                )
                st.stop()

        try:
            model, device, transform, config = cargar_unet_cacheado()
        except FileNotFoundError as e:
            st.error(f"No se pudo cargar la U-Net: {e}")
            st.stop()

        resultados_imgs = {}
        tmp_dir = Path(tempfile.mkdtemp(prefix="water_app_"))
        progreso = st.progress(0.0, text="Procesando imágenes...")

        try:
            for i, archivo in enumerate(archivos_a_procesar):
                ruta_tmp = tmp_dir / archivo.name
                with open(ruta_tmp, "wb") as f:
                    f.write(archivo.getbuffer())

                resultado = pipe.procesar_imagen(ruta_tmp, model, device, transform, config)
                if resultado is not None:
                    resultados_imgs[archivo.name] = resultado

                progreso.progress(
                    (i + 1) / n_a_procesar,
                    text=f"Procesando imágenes... ({i + 1}/{n_a_procesar})"
                )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        progreso.empty()

        df_superpixeles = pipe.combinar_filas(resultados_imgs)

        if df_superpixeles.empty:
            st.warning(
                "No se detectó agua suficiente en ninguna imagen -- no hay "
                "superpíxeles para analizar. Revisa que las imágenes "
                "correspondan al mismo cuerpo de agua con el que se entrenó "
                "la U-Net."
            )
            st.stop()

        if entrenar_modelo:
            scaler, modelo, df_superpixeles = pipe.entrenar_isolation_forest(df_superpixeles)
            pipe.guardar_isolation_forest(
                scaler, modelo,
                metadata={"n_imagenes_entrenamiento": len(resultados_imgs)}
            )
            st.session_state.modo_resultado = "entrenamiento"
        else:
            df_superpixeles = pipe.predecir_anomalias(
                df_superpixeles,
                modelo_guardado["scaler"],
                modelo_guardado["modelo"],
                modelo_guardado["features"],
            )
            st.session_state.modo_resultado = "analisis"

        st.session_state.resultados_imagenes = resultados_imgs
        st.session_state.df_superpixeles = df_superpixeles


    # ========================================================
    # RESULTADOS (persisten entre reruns -- ej. al usar los selectbox
    # de abajo -- porque viven en session_state, no se recalculan)
    # ========================================================

    if st.session_state.df_superpixeles is not None:

        df_sp = st.session_state.df_superpixeles
        resultados_imgs = st.session_state.resultados_imagenes
        modo = st.session_state.modo_resultado

        n_imagenes_proc = len(resultados_imgs)
        n_con_agua = sum(1 for r in resultados_imgs.values() if not r["sin_agua_suficiente"])
        n_superpixeles = len(df_sp)
        n_anomalos = int((df_sp["anomalia"] == -1).sum())
        pct_anomalos = f"{n_anomalos / n_superpixeles * 100:.1f}%" if n_superpixeles else "0%"
        imagenes_con_anomalia = sorted(df_sp.loc[df_sp["anomalia"] == -1, "foto_origen"].unique())

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Resumen general ----
        with st.container(border=True):

            titulo = "✓ Entrenamiento completado" if modo == "entrenamiento" else "📊 Resultados del análisis"
            st.markdown(
                f'<div class="section-title">{titulo}</div>',
                unsafe_allow_html=True
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Imágenes procesadas", f"{n_imagenes_proc:,}")
            m2.metric("Con agua detectada", f"{n_con_agua:,}")
            m3.metric("Superpíxeles analizados", f"{n_superpixeles:,}")
            m4.metric("Superpíxeles anómalos", f"{n_anomalos:,}", pct_anomalos)

            if modo == "entrenamiento":
                st.markdown(
                    textwrap.dedent(
                        """
                        <div class="analysis-message">
                            ✓ Isolation Forest reentrenado y guardado en
                            <code>models/isolation_forest_agua.pkl</code>.
                            La próxima vez que uses "Hacer el análisis" se
                            usará este modelo actualizado.
                        </div>
                        """
                    ),
                    unsafe_allow_html=True
                )

        # ---- Superpíxeles: verde = conservado, rojo = descartado ----
        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):

            st.markdown(
                '<div class="section-title">🟩 Superpíxeles por imagen</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                textwrap.dedent(
                    """
                    <div class="section-description">
                        Verde = superpíxel usado en el análisis (área suficiente).
                        Rojo = descartado por área insuficiente.
                    </div>
                    """
                ),
                unsafe_allow_html=True
            )

            nombres_con_agua = [n for n, r in resultados_imgs.items() if not r["sin_agua_suficiente"]]

            if nombres_con_agua:
                imagen_sel = st.selectbox(
                    "Elige una imagen",
                    nombres_con_agua,
                    key="sel_superpixeles"
                )
                r = resultados_imgs[imagen_sel]
                n_ok = len(r["filas_ok"])
                n_desc = len(r["filas_descartadas"])

                st.image(
                    r["overlay_superpixeles"],
                    caption=f"Verde: {n_ok} conservados  |  Rojo: {n_desc} descartados",
                    use_container_width=True
                )
            else:
                st.markdown(
                    '<div class="info-box">Ninguna imagen tuvo agua suficiente para generar superpíxeles.</div>',
                    unsafe_allow_html=True
                )

        # ---- Anomalías: círculo rojo sobre la imagen ----
        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):

            st.markdown(
                '<div class="section-title">🔴 Anomalías detectadas</div>',
                unsafe_allow_html=True
            )

            if imagenes_con_anomalia:
                st.markdown(
                    textwrap.dedent(
                        f"""
                        <div class="section-description">
                            {len(imagenes_con_anomalia)} de {n_con_agua} imágenes
                            con al menos una anomalía detectada.
                        </div>
                        """
                    ),
                    unsafe_allow_html=True
                )

                imagen_anom_sel = st.selectbox(
                    "Elige una imagen",
                    imagenes_con_anomalia,
                    key="sel_anomalias"
                )

                df_img = df_sp[df_sp["foto_origen"] == imagen_anom_sel]
                fig = pipe.dibujar_anomalias(
                    resultados_imgs[imagen_anom_sel]["imagen_rgb"],
                    df_img
                )
                st.pyplot(fig)

                gps_sel = resultados_imgs[imagen_anom_sel].get("gps")
                if gps_sel:
                    mapa_url = f"https://www.google.com/maps?q={gps_sel['lat']},{gps_sel['lon']}"
                    st.markdown(
                        f"📍 Ubicación: `{gps_sel['lat']:.6f}, {gps_sel['lon']:.6f}` "
                        f"&nbsp;[Ver en el mapa]({mapa_url})",
                        unsafe_allow_html=True
                    )
                else:
                    st.caption("Esta imagen no tiene datos de ubicación (GPS) en su EXIF.")
            else:
                st.markdown(
                    '<div class="info-box">No se detectaron anomalías en ninguna imagen.</div>',
                    unsafe_allow_html=True
                )

        # ---- Tabla resumen: ubicación de TODAS las imágenes con anomalías ----
        if imagenes_con_anomalia:
            st.markdown("<br>", unsafe_allow_html=True)

            with st.container(border=True):

                st.markdown(
                    '<div class="section-title">📍 Ubicación de imágenes con anomalías</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    textwrap.dedent(
                        """
                        <div class="section-description">
                            Coordenadas tomadas del EXIF (GPS) de cada foto -- las
                            imágenes sin ese dato aparecen con ubicación vacía.
                        </div>
                        """
                    ),
                    unsafe_allow_html=True
                )

                filas_ubicacion = []
                for nombre in imagenes_con_anomalia:
                    gps = resultados_imgs[nombre].get("gps")
                    n_anom_img = int((df_sp.loc[df_sp["foto_origen"] == nombre, "anomalia"] == -1).sum())

                    fila = {
                        "Imagen": nombre,
                        "Anomalías": n_anom_img,
                        "Latitud": round(gps["lat"], 6) if gps else None,
                        "Longitud": round(gps["lon"], 6) if gps else None,
                        "Ver en el mapa": (
                            f"https://www.google.com/maps?q={gps['lat']},{gps['lon']}" if gps else None
                        ),
                    }
                    filas_ubicacion.append(fila)

                df_ubicacion = pd.DataFrame(filas_ubicacion)

                st.dataframe(
                    df_ubicacion,
                    column_config={
                        "Ver en el mapa": st.column_config.LinkColumn(
                            "Ver en el mapa", display_text="Abrir mapa"
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )


# ============================================================
# SI NO HAY IMÁGENES
# ============================================================

else:

    st.markdown(
        textwrap.dedent(
            """
            <div class="info-box">
                Selecciona tus imágenes para comenzar el análisis.
            </div>
            """
        ),
        unsafe_allow_html=True
    )
