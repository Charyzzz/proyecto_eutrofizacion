import streamlit as st
import textwrap


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

if 'uploaded_names' not in st.session_state:
    st.session_state.uploaded_names = set()


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
# FUNCIÓN PARA VERIFICAR DUPLICADOS
# ============================================================

def check_duplicate_names(uploaded_files):
    """Verifica si hay archivos con nombres duplicados"""
    
    if not uploaded_files:
        return []
    
    current_names = {file.name for file in uploaded_files}
    duplicates = current_names & st.session_state.uploaded_names
    
    return list(duplicates)


# ============================================================
# FUNCIÓN PARA ACTUALIZAR NOMBRES GUARDADOS
# ============================================================

def update_uploaded_names(uploaded_files):
    """Actualiza la lista de nombres guardados"""
    
    if uploaded_files:
        st.session_state.uploaded_names.update(
            file.name for file in uploaded_files
        )


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

        /* =====================================================
           ALERTA DE DUPLICADOS
           ===================================================== */

        .duplicate-warning {
            background-color: #fef3cd;
            border: 1px solid #ffc107;
            border-radius: 12px;
            padding: 16px 20px;
            color: #856404;
            margin-top: 15px;
            font-size: 14px;
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

        key="water_images"
    )


# ============================================================
# SI SE CARGARON IMÁGENES
# ============================================================

if uploaded_files:

    number_images = len(uploaded_files)

    # ========================================================
    # VERIFICAR DUPLICADOS
    # ========================================================

    duplicates = check_duplicate_names(uploaded_files)

    if duplicates:
        st.error(
            f"❌ **Error:** Los siguientes archivos ya fueron cargados anteriormente y no pueden volver a subirse:\n\n"
            f"{', '.join([f'`{dup}`' for dup in sorted(duplicates)])}\n\n"
            f"Por favor, elimina estos archivos de tu selección e intenta de nuevo."
        )
        st.stop()


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
    # BOTÓN CONTINUAR
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    button_left, button_center, button_right = st.columns(
        [1, 1, 1]
    )


    with button_center:

        continue_analysis = st.button(
            "🔎  Continuar al Análisis  →",
            use_container_width=True
        )


    # ========================================================
    # PLACEHOLDER DEL PIPELINE
    # ========================================================

    if continue_analysis:
        
        # Actualizar nombres guardados
        update_uploaded_names(uploaded_files)

        st.markdown("<br>", unsafe_allow_html=True)

        analysis_html = f"""
        <div class="analysis-message">
            <b>✓ {number_images:,} imágenes listas para analizar.</b>
            <br><br>
            Tamaño total: <b>{total_size_text}</b>
            <br><br>
            El pipeline de segmentación y detección de anomalías se conectará aquí posteriormente.
        </div>
        """
        
        st.markdown(analysis_html, unsafe_allow_html=True)


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
