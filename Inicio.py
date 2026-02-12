import streamlit as st
# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Grupo Orlegi - Panel",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)
import base64
import os
import zipfile
import pandas as pd
import sqlite3
from pathlib import Path
import json
from utils.utils import (construir_pool_percentiles,inicializar_pool_por_defecto)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Esta funcion se encarga de leer el fichero columnas.json y devolver su contenido
# En este fichero se encuentran los nombres de todas las columnas del código
@st.cache_data
def cargar_columnas(json_path="data/columnas.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        columnas = json.load(f)
    return columnas
if "columnas" not in st.session_state:
    st.session_state.columnas = cargar_columnas()
# =========================
# CARGA AUTOMÁTICA DEL DATASET (desde 4 ZIP)
# Cambiada las rutas de los archivos para que funcione localmente
# =========================
@st.cache_data
def load_data():
    base_dir = "data"  # carpeta dentro de tu repo / proyecto
    #base_dir = "data" #ruta original
    zip_files = [
        "noviembre_2025_temporada_2022.zip",
        "noviembre_2025_temporada_2023.zip",
        "noviembre_2025_temporada_2024.zip",
        "noviembre_2025_temporada_2025.zip",
        "noviembre_2025_temporada_2026.zip",  # 👈 NUEVO
    ]

    dfs = []

    for zname in zip_files:
        zpath = os.path.join(base_dir, zname)

        # Leer el primer CSV que haya dentro del zip (sin extraer a disco)
        with zipfile.ZipFile(zpath, "r") as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError(f"El zip {zname} no contiene ningún CSV")
            csv_inside = csv_names[0]

            with z.open(csv_inside) as f:
                df_part = pd.read_csv(
                    f,
                    sep=",",#Separador cambiado para mejorar la eficiencia
                    engine="c",#Cambiado el engine a c para mejorar la eficiencia de 45s a 8s
                    encoding="utf-8-sig"
                )
                dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)

    # Normalizamos Fin de contrato a string SIEMPRE
    if "Fin de contrato" in df.columns:
        df["Fin de contrato"] = df["Fin de contrato"].astype(str)

    return df

# Creo el atrivuto bd_cargada en st para comprobar que se han cargado los datos
if "bd_cargada" not in st.session_state:
    st.session_state.bd_cargada = False
# Campo creado para controlar si el proceso de carga de la base de datos se esta ejecutando
if "bd_cargando" not in st.session_state:
    st.session_state.bd_cargando = False


# =========================
# BLOQUEO DE PANTALLA
# Esta funcion se encarga de bloquear la pantalla al usuario 
# Se usa cuando la base de datos se está cargando, de esta manera
# evitamos problemas de render con componentes que no se desdibujan
# y a su vez evitamos que el usuario pueda acceder a otras páginas
# sin haberse cargado la base de datos
# =========================
def bloquear_pantalla(texto="Cargando datos…"):
    st.markdown(
        f"""
        <style>
        .orlegi-overlay {{
            position: fixed;
            inset: 0;
            background: rgba(4, 18, 46, 0.92);
            backdrop-filter: blur(6px);
            z-index: 99999;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .orlegi-loader-card {{
            background-color: rgba(6, 27, 70, 0.85);
            border-radius: 1rem;
            box-shadow: 0 0 30px rgba(0,0,0,.5);
            padding: 2.5rem 3rem;
            text-align: center;
            min-width: 320px;
        }}

        .orlegi-spinner {{
            width: 48px;
            height: 48px;
            border: 4px solid rgba(255,255,255,0.25);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1.5rem;
        }}

        .orlegi-loader-text {{
            font-size: 1.1rem;
            font-weight: 500;
            color: white;
            opacity: 0.95;
        }}

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        </style>

        <div class="orlegi-overlay">
            <div class="orlegi-loader-card">
                <div class="orlegi-spinner"></div>
                <div class="orlegi-loader-text">{texto}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


if st.session_state.get("bd_cargando", False) and not st.session_state.get("forzar_cambio", False):
    bloquear_pantalla("Cargando base de datos…") 
# =========================
# UTILIDADES IMÁGENES (robusto si falta el archivo)
# Cambiadas las urls de las imagenes para el correcto funcionamiento local
# =========================
def get_image_base64(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None
    
FONDO_PATH = "data/Captura de pantalla 2025-11-24 a las 16.52.04.png"
LOGO_PATH  = "data/logo.png"
#FONDO_PATH = "data/Captura de pantalla 2025-11-24 a las 16.52.04.png" #url original
#LOGO_PATH  = "data/logo.png" #url original

fondo_base64 = get_image_base64(FONDO_PATH)
logo_base64  = get_image_base64(LOGO_PATH)





# =========================
# CSS (tu estética)
# =========================
bg_div = ""
bg_css = ""
if fondo_base64:
    bg_css = f"""
    .background-image {{
        position: fixed;
        inset: 0;
        background-image: url("data:image/png;base64,{fondo_base64}");
        background-size: cover;
        background-position: center;
        opacity: 0.4;
        z-index: 0;
    }}
    """
    bg_div = """<div class="background-image"></div>"""

st.markdown(
    f"""
    <style>
    :root {{
        --orlegi-primary: #0b3c7c;
        --orlegi-primary-hover: #0f4f9c;
        --orlegi-dark: rgba(4, 18, 46, 0.88);
        --orlegi-card: rgba(6, 27, 70, 0.95);
        --orlegi-text: #ffffff;
    }}

    html, body, .stApp {{
        height: 100%;
        color: var(--orlegi-text);
        font-family: ui-sans-serif, system-ui;
    }}

    header {{
    background: transparent !important;
    height: 0 !important;
    overflow: hidden;
    }}

    footer {{
        display: none !important;
    }}

    .stDeployButton {{
        display:none
    }}
    
    main.block-container {{
        padding-top: 0 !important;
        position: relative;
        z-index: 1; /* encima del fondo */
    }}

    {bg_css}

    .login-box {{
        background-color: var(--orlegi-dark);
        border-radius: 1rem;
        box-shadow: 0 0 25px rgba(0,0,0,.4);
        padding: 2rem;
        max-width: 420px;
        margin: auto;
    }}

    div.stButton > button:first-child {{
        background-color: var(--orlegi-primary);
        color: white;
        border-radius: 8px;
        border: 1px solid var(--orlegi-primary);
    }}

    div.stButton > button:first-child:hover {{
        background-color: var(--orlegi-primary-hover);
    }}

    label {{
        color: white !important;
    }}
    </style>

    {bg_div}
    """,
    unsafe_allow_html=True
)

# =========================
# BASE DE DATOS (SQLite)
# =========================
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = str(DATA_DIR / "usuarios.db")

def db_connect():
    # timeout ayuda en Cloud si hay locks breves
    return sqlite3.connect(DB_PATH, timeout=30)

def init_db():
    conn = db_connect()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS usuarios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        usuario TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        rol TEXT DEFAULT 'user',
        forzar_cambio INTEGER DEFAULT 1
    )
    """)
    conn.commit()
    conn.close()

def crear_usuario_si_no_existe(usuario: str, password: str, rol: str = "user"):
    """
    Inserta solo si no existe. NO ocultamos errores reales.
    """
    usuario = normalizar_usuario(usuario)
    conn = db_connect()
    c = conn.cursor()
    try:
        c.execute(
            """INSERT INTO usuarios (usuario, password, rol, forzar_cambio)
               VALUES (?, ?, ?, 1)""",
            (usuario, generate_password_hash(password), rol)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # usuario ya existe
        pass
    finally:
        conn.close()

def get_user(usuario: str):
    usuario = normalizar_usuario(usuario)
    conn = db_connect()
    c = conn.cursor()
    c.execute(
        "SELECT usuario, password, forzar_cambio, rol FROM usuarios WHERE usuario=?",
        (usuario,)
    )
    row = c.fetchone()
    conn.close()
    return row  # (usuario, password_hash, forzar_cambio, rol)

def verificar_login(usuario: str, password: str) -> bool:
    user = get_user(usuario)
    if user:
        return check_password_hash(user[1], password)
    return False

def debe_cambiar_password(usuario: str) -> bool:
    user = get_user(usuario)
    return bool(user) and int(user[2]) == 1

def cambiar_password(usuario: str, nueva: str):
    usuario = normalizar_usuario(usuario)
    conn = db_connect()
    c = conn.cursor()
    c.execute(
        """UPDATE usuarios
           SET password=?, forzar_cambio=0
           WHERE usuario=?""",
        (generate_password_hash(nueva), usuario)
    )
    conn.commit()
    conn.close()

def normalizar_usuario(usuario: str) -> str:
    # evita fallos por espacios/mayúsculas
    return (usuario or "").strip().lower()

def seed_usuarios_iniciales():
    """
    IMPORTANTE: solo sembrar si la tabla está vacía.
    Así NO dependes de reruns.
    """
    conn = db_connect()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM usuarios")
    n = c.fetchone()[0]
    conn.close()

    if n > 0:
        return

    usuarios = [
        "admin","rpuerta","pquiros","ivillaseñor",
        "jriestra","ggarcia","jmhernandez",
        "flobeiras","mromero"
    ]
    for u in usuarios:
        crear_usuario_si_no_existe(u, "Orlegi2025", "admin" if u == "admin" else "user")

init_db()
seed_usuarios_iniciales()

# =========================
# SESSION STATE
# =========================
if "logueado" not in st.session_state:
    st.session_state.logueado = False
if "usuario_actual" not in st.session_state:
    st.session_state.usuario_actual = None
if "forzar_cambio" not in st.session_state:
    st.session_state.forzar_cambio = False


# =========================
# LOGIN
# =========================
if not st.session_state.logueado:

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none !important; }
        .block-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 90vh;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if logo_base64:
        st.markdown(
            f"""
            <div style="text-align:center; margin-bottom:1.5rem;">
                <img src="data:image/png;base64,{logo_base64}" style="width:260px;">
            </div>
            """,
            unsafe_allow_html=True
        )

    usuario_raw = st.text_input("Usuario")
    contrasena = st.text_input("Contraseña", type="password")

    if st.button("Acceder"):
        usuario = normalizar_usuario(usuario_raw)
        if verificar_login(usuario, contrasena):
            st.session_state.logueado = True
            st.session_state.usuario_actual = usuario
            st.session_state.forzar_cambio = debe_cambiar_password(usuario)
            # Solo cargamos BD si NO hay cambio de contraseña pendiente.
            st.session_state.bd_cargando = not st.session_state.forzar_cambio
            st.rerun()
        else:
            st.error("❌ Usuario o contraseña incorrectos")

    st.stop()

# =========================
# FORZAR CAMBIO PRIMER LOGIN
# =========================
if st.session_state.forzar_cambio:
    # Evita overlays heredados de estados previos durante el cambio obligatorio.
    st.session_state.bd_cargando = False
    st.warning("⚠️ Debes cambiar tu contraseña antes de continuar")

    actual = st.text_input("Contraseña actual", type="password")
    nueva = st.text_input("Nueva contraseña", type="password")
    confirmar = st.text_input("Confirmar nueva contraseña", type="password")

    if st.button("Guardar nueva contraseña"):
        u = st.session_state.usuario_actual
        if not verificar_login(u, actual):
            st.error("Contraseña actual incorrecta")
        elif nueva != confirmar:
            st.error("No coinciden")
        elif len(nueva) < 8:
            st.error("Mínimo 8 caracteres")
        else:
            cambiar_password(u, nueva)
            st.session_state.forzar_cambio = False
            # Tras cambiar contraseña por primera vez, ahora sí arrancamos carga de BD.
            if not st.session_state.get("bd_cargada", False):
                st.session_state.bd_cargando = True
            st.success("Contraseña actualizada correctamente")
            st.rerun()

    st.stop()

# =========================
# CARGA DE BASE DE DATOS
# Se carga la  base de datos y a su vez se bloquea a sidebar, evitando que los usuarios puedan acceder a las paginas
# sin tener la base de datos cargada
# =========================
if st.session_state.bd_cargando and not st.session_state.bd_cargada:

    # Ocultamos todo
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none !important; }
        .block-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 90vh;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.spinner("Cargando base de datos..."):
        st.session_state.data = load_data()
        st.session_state.bd_cargada = True
        st.session_state.bd_cargando = False

    st.rerun()

#  Inicializar SOLO si no existe todavía
if "df_pool_percentiles" not in st.session_state:
    df_pool, pool_info = inicializar_pool_por_defecto(st.session_state.data)
    st.session_state["df_pool_percentiles"] = df_pool
    st.session_state["pool_info"] = pool_info
# =========================
# PANEL NORMAL
# =========================
if logo_base64:
    st.markdown(
        f"""
        <div style="position:fixed; top:30px; right:30px; z-index:1000;">
            <img src="data:image/png;base64,{logo_base64}" style="width:180px; margin-top:1rem; ">
        </div>
        """,
        unsafe_allow_html=True
    )

#st.success(f"Bienvenido {st.session_state.usuario_actual}")
st.title(f"Bienvenido {st.session_state.usuario_actual} ")
st.markdown("---")

# =========================
# SECCIÓN 1: MÉTRICAS DEL POOL ACTIVO
# =========================
if "pool_info" in st.session_state:
    pool_info = st.session_state["pool_info"]
    
    st.subheader(" Pool Activo")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Temporada", pool_info["temporada"])
    col2.metric("Ligas", len(pool_info["liga"]))
    col3.metric("Jugadores", pool_info["n_jugadores"])
    
    # Métrica adicional útil: promedio de edad o valor de mercado
    if "df_pool_percentiles" in st.session_state:
        df_pool = st.session_state["df_pool_percentiles"]
        if "Edad" in df_pool.columns:
            edad_promedio = df_pool["Edad"].mean()
            col4.metric("Edad promedio", f"{edad_promedio:.1f} años")
    
    # Expandible con detalles de ligas
    with st.expander(" Ver ligas seleccionadas"):
        for i, liga in enumerate(pool_info["liga"], 1):
            st.write(f"{i}. {liga}")

# =========================
# SECCIÓN 2: INSIGHTS RÁPIDOS
# =========================
if "df_pool_percentiles" in st.session_state:
    df_pool = st.session_state["df_pool_percentiles"]
    
    st.markdown("---")
    st.subheader(" Insights de Scouting")
    
    # Tres columnas con información útil
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    # INSIGHT 1: Jugadores interesantes (bajo valor, alto rendimiento)
    with insight_col1:
        st.markdown("####  Jugador interesante")
        st.caption("Alto rendimiento, bajo valor de mercado")
        
        if "Valor_Mercado" in df_pool.columns and "Percentil Score Total" in df_pool.columns:
            # Filtrar jugadores con alto score pero bajo valor
            joyas = df_pool[
                (df_pool["Percentil Score Total"] >= 80) & 
                (df_pool["Valor_Mercado"] < df_pool["Valor_Mercado"].quantile(0.3))
            ].sort_values("Percentil Score Total", ascending=False).head(3)
            
            if not joyas.empty:
                for _, jugador in joyas.iterrows():
                    st.write(f"**{jugador['Jugador']}**")
                    st.caption(f"{jugador['Equipo']} | Score: {jugador.get('Percentil Score Total', 'N/A')}")
            else:
                st.info("No se encontraron jugadores interesantes")
        else:
            st.info("Datos no disponibles")
    
    # INSIGHT 2: Jóvenes promesas
    with insight_col2:
        st.markdown("####  Jóvenes Promesas")
        st.caption("Sub-23 con alto rendimiento")
        
        if "Edad" in df_pool.columns and "Percentil Score Total" in df_pool.columns:
            jovenes = df_pool[
                (df_pool["Edad"] < 23) & 
                (df_pool["Percentil Score Total"] >= 75)
            ].sort_values("Percentil Score Total", ascending=False).head(3)
            
            if not jovenes.empty:
                for _, jugador in jovenes.iterrows():
                    st.write(f"**{jugador['Jugador']}**")
                    st.caption(f"{jugador['Equipo']} | {jugador['Edad']} años | Score: {jugador.get('Percentil Score Total', 'N/A')}")
            else:
                st.info("No se encontraron jóvenes promesas")
        else:
            st.info("Datos no disponibles")
    
    # INSIGHT 3: Contratos por vencer
    with insight_col3:
        st.markdown("#### Oportunidades")
        st.caption("Contratos cercanos a vencer")
        
        if "Fin de contrato" in df_pool.columns and "Percentil Score Total" in df_pool.columns:
            # Convertir a datetime si es string
            try:
                df_pool["Fin_contrato_dt"] = pd.to_datetime(df_pool["Fin de contrato"], errors='coerce')
                año_actual = datetime.now().year
                
                oportunidades = df_pool[
                    (df_pool["Fin_contrato_dt"].dt.year <= año_actual + 1) & 
                    (df_pool["Percentil Score Total"] >= 75)
                ].sort_values("Percentil Score Total", ascending=False).head(3)
                
                if not oportunidades.empty:
                    for _, jugador in oportunidades.iterrows():
                        st.write(f"**{jugador['Jugador']}**")
                        st.caption(f"{jugador['Equipo']} | Vence: {jugador['Fin de contrato']}")
                else:
                    st.info("No hay oportunidades destacadas")
            except:
                st.info("Datos no disponibles")
        else:
            st.info("Datos no disponibles")
    
    
    # =========================
    # SECCIÓN 3: DISTRIBUCIÓN DEL POOL (GRÁFICOS)
    # =========================
    st.markdown("---")
    st.subheader("Análisis del Pool")
    
    graph_col1, graph_col2 = st.columns(2)
    
    # GRÁFICO 1: Distribución por posición
    with graph_col1:
        if "Pos" in df_pool.columns:
            st.markdown("##### Jugadores por Posición")
            pos_counts = df_pool["Pos"].value_counts().head(10)
            
            fig_pos = px.bar(
                x=pos_counts.index,
                y=pos_counts.values,
                labels={"x": "Posición", "y": "Cantidad"},
                color=pos_counts.values,
                color_continuous_scale="Viridis"
            )
            fig_pos.update_layout(
                showlegend=False,
                height=300,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_pos, use_container_width=True)
    
    # GRÁFICO 2: Distribución de edad
    with graph_col2:
        if "Edad" in df_pool.columns:
            st.markdown("##### Distribución de Edad")
            
            fig_edad = px.histogram(
                df_pool,
                x="Edad",
                nbins=20,
                labels={"Edad": "Edad", "count": "Cantidad"},
                color_discrete_sequence=["#1f77b4"]
            )
            fig_edad.update_layout(
                showlegend=False,
                height=300,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_edad, use_container_width=True)
    
    # GRÁFICO 3: Scatter - Edad vs Rendimiento (ancho completo)
    if "Edad" in df_pool.columns and "Percentil Score Total" in df_pool.columns:
        st.markdown("##### Rendimiento vs Edad")
        
        # Preparar datos para el scatter
        scatter_data = df_pool.dropna(subset=["Edad", "Percentil Score Total"])
        
        # Colorear por valor de mercado si existe
        if "Valor_Mercado" in scatter_data.columns:
            fig_scatter = px.scatter(
                scatter_data,
                x="Edad",
                y="Percentil Score Total",
                color="Valor_Mercado",
                hover_data=["Jugador", "Equipo", "Pos"],
                labels={
                    "Edad": "Edad",
                    "Percentil Score Total": "Score Total",
                    "Valor_Mercado": "Valor de Mercado"
                },
                color_continuous_scale="Turbo"
            )
        else:
            fig_scatter = px.scatter(
                scatter_data,
                x="Edad",
                y="Percentil Score Total",
                hover_data=["Jugador", "Equipo", "Pos"],
                labels={
                    "Edad": "Edad",
                    "Percentil Score Total": "Score Total"
                }
            )
        
        fig_scatter.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.info(" No hay un pool activo. Ve a **Campograma** para configurar tus filtros y empezar a explorar jugadores.")

# from streamlit_image_coordinates import streamlit_image_coordinates
# from PIL import Image

# st.title("Mapeo de Posiciones en Imagen")

# # 1. Cargar la imagen (puede ser local o URL)
# # Asegúrate de tener una imagen disponible

# if "rol_seleccionado" not in st.session_state:
#     st.session_state.rol_seleccionado = "MC Ofensivo" # Valor por defecto
# ZONA_MC = {
#     "x_min": 460,
#     "x_max": 560,
#     "y_min": 685,
#     "y_max": 815
# }
# ROLES_INFO = {
#     "MC Ofensivo": {
#         "metrics": {
#             "Pases clave": 0.9,
#             "xA": 0.85,
#             "Progresiones": 0.8,
#             "Acciones último tercio": 0.88,
#             "Tiros": 0.6
#         },
#         "descripcion": "Mediocentro orientado a generación ofensiva y último pase.",
#         "metodologia": """
#         Este perfil prioriza métricas de creación y progresión.
#         Se ponderan acciones en último tercio y volumen de generación
#         por encima de métricas defensivas.
#         """
#     },
#     "MC Contención": {
#         "metrics": {
#             "Intercepciones": 0.9,
#             "Duelos defensivos": 0.85,
#             "Recuperaciones": 0.88,
#             "Pases seguros": 0.75,
#             "Faltas cometidas": 0.4
#         },
#         "descripcion": "Perfil enfocado en equilibrio y recuperación.",
#         "metodologia": """
#         Se priorizan acciones defensivas y control del ritmo.
#         La progresión tiene menor peso relativo.
#         """
#     }
# }
# def render_radar(role_name):
#     role_data = ROLES_INFO[role_name]
#     metrics = role_data["metrics"]

#     categories = list(metrics.keys())
#     values = list(metrics.values())

#     fig = go.Figure()

#     fig.add_trace(go.Scatterpolar(
#         r=values,
#         theta=categories,
#         fill='toself',
#         name=role_name
#     ))

#     fig.update_layout(
#         polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
#         showlegend=False
#     )

#     st.plotly_chart(fig, use_container_width=True)


# img = Image.open("data/mapaPosiciones.PNG")
# col_mapa, col_info = st.columns([2, 1])

# with col_mapa:
#     value = streamlit_image_coordinates(img, key="mapa")

# with col_info:
#     if "rol_seleccionado" in st.session_state:
#         rol = st.session_state.rol_seleccionado
#         info = ROLES_INFO[rol]

#         st.markdown(f"## 📍 {rol}")
#         st.write(info["descripcion"])

#         render_radar(rol)

#         with st.expander("Metodología de evaluación"):
#             st.write(info["metodologia"])


# # 3. Mostrar las coordenadas obtenidas
# if value is not None:
#     st.write(f"Coordenadas: {value['x']}, {value['y']}")
#     x = value["x"]
#     y = value["y"]

#     if (
#         ZONA_MC["x_min"] <= x <= ZONA_MC["x_max"]
#         and
#         ZONA_MC["y_min"] <= y <= ZONA_MC["y_max"]
#     ):
#         st.session_state.rol_seleccionado = "MC Contención"
#     else: 
#         st.session_state.posicion_seleccionada = ""
# =========================
# BOTÓN CERRAR SESIÓN
# =========================
st.markdown("---")
if st.sidebar.button("🚪 Cerrar sesión"):
    st.session_state.logueado = False
    st.session_state.usuario_actual = None
    st.session_state.forzar_cambio = False
    st.rerun()
