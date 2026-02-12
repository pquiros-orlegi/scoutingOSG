import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import construir_pool_percentiles, match_posicion
from mplsoccer import Pitch
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import base64
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
import json
import re
import io
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from io import BytesIO

FONDO_PATH = "data/Captura de pantalla 2025-11-24 a las 16.52.04.png"
#FONDO_PATH = "data/Captura de pantalla 2025-11-24 a las 16.52.04.png" #ruta original

def get_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

fondo_base64 = get_image_base64(FONDO_PATH)

st.markdown(
    f"""
    <style>
    html, body, .stApp {{
        margin: 0;
        padding: 0;
        height: 100%;
        overscroll-behavior: none;
        background: transparent !important;  /* 👈 que no tape la imagen */
    }}

    /* Contenedor principal del contenido */
    .main .block-container {{
        position: relative;
        z-index: 2;               /* por encima del fondo */
        background: transparent !important;
    }}

    .background-image-rating {{
        position: fixed;
        top: 0;
        left: 0;
        height: 100%;
        width: 100%;
        background-image: url("data:image/png;base64,{fondo_base64}");
        background-size: cover;
        background-position: center;
        opacity: 0.99;            /* igual que el login; sube a 0.2 si lo quieres más fuerte */
        z-index: 0;               /* detrás del contenido, pero visible */
    }}
    </style>

    <div class="background-image-rating"></div>
    """,
    unsafe_allow_html=True
)

# =========================
# ⚠️ (OPCIONAL) PROTEGER CON LOGIN DE home.py
# =========================
if "logueado" in st.session_state and not st.session_state.logueado:
    st.error("Debes iniciar sesión en la página principal (home.py) para acceder a este panel.")
    st.stop()

# Medida de seguridad implementada para evitar acceder a la pagina sin haberse cargado la base de datos

if not st.session_state.get("bd_cargada", False):
    st.warning("⚠️ Primero debes cargar la base de datos en Inicio")
    st.stop()

if "orden_campograma" not in st.session_state:
    st.session_state.orden_campograma = {
        "PORTERO": "Portero",
        "LATERAL": "Genérico",
        "DFC": "Genérico",
        "MC": "Genérico",
        "EXTREMO": "Genérico",
        "DELANTERO": "Delantero",
        "Segundo Delantero": "Segundo delantero",
        "MC ContenciÃ³n": "ContenciÃ³n",
        "MC Box to Box": "Box to Box",
        "MC Ofensivo": "Ofensivo",
    }

# Esta funcion se encarga de leer el fichero columnas.json y devolver su contenido
# En este fichero se encuentran los nombres de todas las columnas del código
@st.cache_data
def cargar_columnas(json_path="data/columnas.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        columnas = json.load(f)
    return columnas

# Esta función se encarga de leer el fichero cellstyle.json y devolver su contenido
@st.cache_data
def cargar_jscode_colors(json_path="data/cellstyle.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        code = json.load(f)
    return code

# Esta función se encarga de leer el fichero grupos_score.json y devolver su contenido
@st.cache_data
def cargar_grupos_score(json_path="data/grupos_score.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        scores = json.load(f)
    return scores
# Esta función se encarga de leer el fichero roles_cfg.json y devolver su contenido
@st.cache_data
def cargar_roles_cfg(json_path="data/roles_cfg.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        scores = json.load(f)
    return scores
# Esta función se encarga de leer el fichero map_pos_to_familia.json y devolver su contenido
@st.cache_data
def cargar_map_pos_to_familia(json_path="data/map_pos_to_familia.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        scores = json.load(f)
    return scores
# =========================
# LISTA GLOBAL DE TODAS LAS COLUMNAS DEL SCRIPT CARGADAS DESDE COLUMNAS.JSON
# =========================
if "columnas" not in st.session_state:
    st.session_state.columnas = cargar_columnas()

# =========================
# LISTA GLOBAL DE TODOS LOS ESTILOS DE LAS CELDAS, CARGADA DE JSCODE.JSON
# =========================
if "cellstyle" not in st.session_state:
    st.session_state.cellstyle = cargar_jscode_colors()








def make_cellstyle_js(score_type):
    color = st.session_state.cellstyle["CELLSTYLE_SCORE"]["highlightColors"].get(score_type, st.session_state.cellstyle["CELLSTYLE_SCORE"]["defaultBackground"])
    if(score_type=="SCORE_JS"):
        default_color="black"
    else:
        default_color = st.session_state.cellstyle["CELLSTYLE_SCORE"]["defaultColor"]
    
    return JsCode(f"""function(params) {{
        const baseStyle = {{
            backgroundColor: '{st.session_state.cellstyle["CELLSTYLE_SCORE"]["defaultBackground"]}',
            color: '{default_color}',
            textAlign: 'center',
            verticalAlign: 'middle',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: '1px solid #ffffff22',
            padding: '0px',
            fontSize: '10px'
        }};
        if (params.value >= 85) {{
            return {{
                ...baseStyle,
                backgroundColor: '{color}'
            }};
        }}
        return baseStyle;
    }}""")

# =========================
# JS PARA COLOREAR CELDAS DE SCORE (>=85 VERDE) - GENERALES
# =========================
CELLSTYLE_SCORE_JS = make_cellstyle_js("SCORE_JS")
CELLSTYLE_SCORE_GK_PORTERO_JS = make_cellstyle_js("GK_PORTERO")
CELLSTYLE_SCORE_GK_ATAJADOR_JS = make_cellstyle_js("GK_ATAJADOR")
CELLSTYLE_SCORE_GK_PIES_JS = make_cellstyle_js("GK_PIES")

# =========================
# ESTILOS ESPECIALES SCORE LATERALES (>= 85)
# =========================
CELLSTYLE_SCORE_LAT_GENERICO_JS = make_cellstyle_js("LAT_GENERICO")
CELLSTYLE_SCORE_LAT_DEFENSIVO_JS = make_cellstyle_js("LAT_DEFENSIVO")
CELLSTYLE_SCORE_LAT_OFENSIVO_JS = make_cellstyle_js("LAT_OFENSIVO")

# =========================
# ESTILOS ESPECIALES SCORE DFC (>= 85)
# =========================
CELLSTYLE_SCORE_DFC_GENERICO_JS = make_cellstyle_js("DFC_GENERICO")
CELLSTYLE_SCORE_DFC_DEFENSIVO_JS = make_cellstyle_js("DFC_DEFENSIVO")
CELLSTYLE_SCORE_DFC_COMBINATIVO_JS = make_cellstyle_js("DFC_COMBINATIVO")

# =========================
# ESTILOS ESPECIALES SCORE MC (>= 85)
# =========================
CELLSTYLE_SCORE_MC_GENERICO_JS = make_cellstyle_js("MC_GENERICO")
CELLSTYLE_SCORE_MC_CONTENCION_JS = make_cellstyle_js("MC_CONTENCION")
CELLSTYLE_SCORE_MC_OFENSIVO_JS = make_cellstyle_js("MC_OFENSIVO")
CELLSTYLE_SCORE_MC_B2B_JS = make_cellstyle_js("MC_B2B")

# =========================
# ESTILOS ESPECIALES SCORE EXTREMOS (>= 85)
# =========================
CELLSTYLE_SCORE_EXT_GENERICO_JS = make_cellstyle_js("EXT_GENERICO")
CELLSTYLE_SCORE_EXT_WIDEOUT_JS = make_cellstyle_js("EXT_WIDEOUT")
CELLSTYLE_SCORE_EXT_INCORPORACION_JS = make_cellstyle_js("EXT_INCORPORACION")
CELLSTYLE_SCORE_EXT_COMBINATIVO_JS = make_cellstyle_js("EXT_COMBINATIVO")

# =========================
# ESTILOS ESPECIALES SCORE DELANTEROS (>= 85)
# =========================
CELLSTYLE_SCORE_DEL_DELANTERO_JS = make_cellstyle_js("DEL_DELANTERO")
CELLSTYLE_SCORE_DEL_9_JS = make_cellstyle_js("DEL_9")
CELLSTYLE_SCORE_DEL_SEGUNDO_JS =make_cellstyle_js("DEL_SEGUNDO")

# =========================
# HELPER: CREAR JS DEGRADADO POR COLUMNA (TIPO CMAP)
# =========================
def crear_cmap_js(cmap: str, vmin: float, vmax: float, invert: bool = False) -> JsCode:
    inv = "true" if invert else "false"
    return JsCode(f"""
function(params) {{
    const baseStyle = {{
        backgroundColor: '#09202E',
        color: 'white',
        textAlign: 'center',
        verticalAlign: 'middle',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        border: '1px solid #ffffff22',
        padding: '0px',
        fontSize: '10px'
    }};
    if (params.value == null || isNaN(params.value)) return baseStyle;

    var v = Number(params.value);
    var min = {vmin};
    var max = {vmax};
    if (max === min) return baseStyle;

    var t = (v - min) / (max - min);
    if (t < 0) t = 0;
    if (t > 1) t = 1;

    if ({inv}) {{
        t = 1 - t;   // 👈 invierte el gradiente
    }}

    var r, g, b;
    if ("{cmap}" === "Blues") {{
        var light = 230 - Math.round(t * 150);
        r = light; g = light + 10; b = 255;
    }} else if ("{cmap}" === "Reds") {{
        r = 255; g = 230 - Math.round(t * 180); b = g;
    }} else if ("{cmap}" === "Oranges") {{
        r = 255; g = 200 - Math.round(t * 150); b = 0;
    }} else if ("{cmap}" === "Greens") {{
        g = 255; r = 230 - Math.round(t * 180); b = r;
    }} else if ("{cmap}" === "Purples") {{
        r = 230 - Math.round(t * 80);
        g = 220 - Math.round(t * 160);
        b = 255;
    }} else {{
        return baseStyle;
    }}

    var bg = 'rgb(' + r + ',' + g + ',' + b + ')';
    return {{
        backgroundColor: bg,
        color: 'black',
        textAlign: 'center',
        verticalAlign: 'middle',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        border: '1px solid #ffffff22',
        padding: '0px',
        fontSize: '10px'
    }};
}}
""")




def limpiar_header(colname: str) -> str:
    return re.sub(r"\s*\(.*?\)", "", str(colname)).strip()



# =========================
    # ✅ FIX: NEGATIVOS (menos = mejor) -> invert=True
    # =========================
NEGATIVE_TOKENS = [
    "PÉRDIDAS", "PERDIDAS",
    "ACCIONES FALLIDAS",
    "FALLIDAS",
]

def is_negative_metric(colname: str) -> bool:
    up = str(colname).upper()
    return any(tok in up for tok in NEGATIVE_TOKENS)

# Bloque de codigo refactorizado de la funcion mostrar_tabla_aggrid
# Aplica gradientes de color a columnas numéricas de AgGrid,
# invirtiendo automáticamente métricas negativas y usando rangos reales
def aplicar_cmap(gb, tabla, df_base, cols, cmap, invert_if_negative=True):
    for col in cols:
        if col not in tabla.columns:
            continue

        s = pd.to_numeric(df_base[col], errors="coerce")
        if s.dropna().empty:
            continue

        vmin, vmax = float(s.min()), float(s.max())
        inv = invert_if_negative and is_negative_metric(col)

        gb.configure_column(
            col,
            cellStyle=crear_cmap_js(cmap, vmin, vmax, invert=inv)
        )

# =========================
# HELPER: MOSTRAR TABLA CON AgGrid
# =========================
def mostrar_tabla_aggrid( rol_seleccionado,df_tabla, key, df_base):
    """
    Todas las columnas a 200px (estable local/web) manteniendo TODOS los gradientes y estilos.
    """
    tabla = df_tabla.copy()
    jugador_col = ["Jugador"] if "Jugador" in tabla.columns else []

    percentil_cols = [c for c in tabla.columns if c.startswith("Percentil Score ")]

    resto_cols = [
        c for c in tabla.columns
        if c not in jugador_col and c not in percentil_cols
    ]

    nuevo_orden = jugador_col + percentil_cols + resto_cols
    tabla = tabla[nuevo_orden]
    # 🔢 métricas con paréntesis → 2 decimales
    for col in tabla.columns:
        if "(" in col and ")" in col:
            tabla[col] = pd.to_numeric(tabla[col], errors="coerce").round(2)

    if df_base is None:
        df_base = tabla

    # --- Jugador primero ---
    if "Jugador" in tabla.columns:
        cols = ["Jugador"] + [c for c in tabla.columns if c != "Jugador"]
        tabla = tabla[cols]

    if "Minutos jugados" in tabla.columns:
        tabla["Minutos jugados"] = pd.to_numeric(
            tabla["Minutos jugados"], errors="coerce"
        ).astype("Int64")

    gb = GridOptionsBuilder.from_dataframe(
        tabla,
        enableRowGroup=True,
        enableValue=True,
        enablePivot=True
    )
    

    # ✅ Estilos base (los tuyos)
    base_cell_style = {
        'backgroundColor': '#09202E',
        'color': 'white',
        'textAlign': 'center',
        'verticalAlign': 'middle',
        'border': '1px solid #ffffff22',
        'padding': '0px',
        'fontSize': '10px'
    }
    base_header_style = {
        'backgroundColor': '#09202E',
        'color': 'white',
        'fontWeight': 'bold',
        'borderBottom': '1px solid #ffffff22',
        'fontSize': '11px',
        'padding': '0px',
        'textAlign': 'center',
        'verticalAlign': 'middle'
    }

    # ✅ TODAS las columnas a 200px (y que no cambien)
    gb.configure_default_column(
        wrapText=False,
        autoHeight=False,
        resizable=True,
        sortable=True,
        filter=True,
        width=200,
        minWidth=200,
        maxWidth=200,
        cellStyle=base_cell_style,
        headerStyle=base_header_style
    )

    # Fijar Jugador a la izquierda (también 200px)
    if "Jugador" in tabla.columns:
        gb.configure_column(
            "Jugador",
            pinned="left",
            lockPinned=True,
            width=200,
            minWidth=200,
            maxWidth=200
        )

    # Grid options estables (sin autosize)
    gb.configure_grid_options(
        suppressSizeToFit=True,
        suppressColumnVirtualisation=False,
        alwaysShowHorizontalScroll=True
    )

    # Configuración general por grupo
    GRUPOS_SCORE = cargar_grupos_score()

    # ===== Iteración dinámica =====
    # Colores default para cualquier Percentil Score
    cols_percentil_score = [c for c in tabla.columns if c.startswith("Percentil Score ")]
    for col in cols_percentil_score:
        gb.configure_column(col, cellStyle=make_cellstyle_js("default"))

    # Recorremos los grupos según el key
    for grupo, cfg in GRUPOS_SCORE.items():
        if key in cfg["keys"]:
            # Percentiles
            for col_name, score_type in cfg["percentil"].items():
                if col_name in tabla.columns:
                    gb.configure_column(col_name, cellStyle=make_cellstyle_js(score_type))
            
            # Colormap
            for sufijo, (cmap, invert) in cfg["cmap_cols"].items():
                cols = [c for c in tabla.columns if sufijo in c]
                aplicar_cmap(gb=gb, tabla=tabla, df_base=df_base, cols=cols, cmap=cmap, invert_if_negative=invert)
    
    grid_options = gb.build()
    
    # =========================
    # Limpieza de headers (quitar lo que hay entre paréntesis)
    # =========================
    for coldef in grid_options.get("columnDefs", []):
        field = coldef.get("field")
        if field:
            coldef["headerName"] = limpiar_header(field)


    # ❌ NO autosize (porque queremos 200 fijo)
    num_rows = len(tabla)
    grid_height = 60 + 30 * min(num_rows, 10)

    # ❌ NO autosize (porque queremos 200 fijo)
    num_rows = len(tabla)
    grid_height = 60 + 30 * min(num_rows, 10)
    grid_key = f"{key}_{rol_seleccionado}_{len(tabla)}"
    AgGrid(
        tabla,
        gridOptions=grid_options,
        theme="streamlit",
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False,
        reload_data=True,
        update_mode='NO_UPDATE',
        data_return_mode='AS_INPUT',
        domLayout='normal',
        height=grid_height,
        key=grid_key,
        custom_css={
            ".ag-root-wrapper": {
                "border": "none !important",
                "max-width": "100% !important",
                "margin": "0 auto",
                "overflow-x": "auto !important",
                "overflow-y": "auto !important"
            },
            ".ag-cell, .ag-header-cell-label": {
                "display": "flex !important",
                "align-items": "center !important",
                "justify-content": "center !important",
            },
            ".ag-theme-streamlit, .ag-root, .ag-cell, .ag-header-cell": {
                "font-family": "Arial, sans-serif !important"
            },
            ".ag-header-cell-label": {
                "white-space": "nowrap !important"
            },
            ".ag-cell": {
                "white-space": "nowrap !important",
                "line-height": "12px !important"
            },
            ".ag-theme-streamlit": {
                "width": "100% !important"
            }
        }
    )
    return tabla #Devuelve el df usado




def limpiar_header(colname: str) -> str:
    """
    Elimina todo lo que esté entre paréntesis y limpia espacios.
    Ej: "% Pases (GK_PORTERO)" -> "% Pases"
    """
    return re.sub(r"\s*\(.*?\)", "", colname).strip()




# =========================
# HELPERS DE RANKING
# =========================
def rankings_defensivos(df_filtrado: pd.DataFrame):
    # helper: ordena priorizando el SCORE bruto;
    # si no existe, usa el percentil como backup.
    def sort_by_score(df_pos, score_col_name):
        if score_col_name in df_pos.columns:
            return df_pos.sort_values(score_col_name, ascending=False).copy()
        else:
            pct_col = f"Percentil {score_col_name}"
            if pct_col in df_pos.columns:
                return df_pos.sort_values(pct_col, ascending=False).copy()
            return df_pos.copy()

    # PORTEROS
    df_por = df_filtrado[
        df_filtrado["Pos"].apply(
            lambda v: match_posicion(v, {"POR", "GK", "PORTERO", "GOALKEEPER"})
        )
    ].copy()
    df_por = sort_by_score(df_por, "Score GK Total")

    # LATERAL IZQUIERDO
    df_li = df_filtrado[
        df_filtrado["Pos"].apply(lambda v: match_posicion(v, {"LI", "CAI"}))
    ].copy()
    df_li = sort_by_score(df_li, "Score Lateral Total")

    # DFC (pool común)
    df_dfc_pool = df_filtrado[
        df_filtrado["Pos"].apply(lambda v: match_posicion(v, {"DFC"}))
    ].copy()
    df_dfc_pool = sort_by_score(df_dfc_pool, "Score Central Total")

    # mismo split que tenías antes: uno sí, uno no
    df_dfc_der = df_dfc_pool.iloc[0::2].copy()
    df_dfc_izq = df_dfc_pool.iloc[1::2].copy()

    # LATERAL DERECHO
    df_ld = df_filtrado[
        df_filtrado["Pos"].apply(lambda v: match_posicion(v, {"LD", "CAD"}))
    ].copy()
    df_ld = sort_by_score(df_ld, "Score Lateral Total")

    # MC (pool para los 3 roles)
    df_mc_pool = df_filtrado[
        df_filtrado["Pos"].apply(lambda v: match_posicion(v, {"MCD", "MC", "MCO"}))
    ].copy()

    df_mc_contencion = sort_by_score(df_mc_pool.copy(), "Score MC Contención")
    df_mc_b2b       = sort_by_score(df_mc_pool.copy(), "Score MC Box-to-Box")
    df_mc_ofensivo  = sort_by_score(df_mc_pool.copy(), "Score MC Ofensivo")

    # EXTREMOS
    df_ei = df_filtrado[
        df_filtrado["Pos"].apply(lambda v: match_posicion(v, {"EI", "MI"}))
    ].copy()
    df_ei = sort_by_score(df_ei, "Score Extremo Wide Out")

    df_ed = df_filtrado[
        df_filtrado["Pos"].apply(lambda v: match_posicion(v, {"ED", "MD"}))
    ].copy()
    df_ed = sort_by_score(df_ed, "Score Extremo Wide Out")

    # DELANTEROS (pool compartido)
    df_dc_pool = df_filtrado[
        df_filtrado["Pos"].apply(lambda v: match_posicion(v, {"DC", "SDI", "SDD"}))
    ].copy()
    df_dc = sort_by_score(df_dc_pool.copy(), "Score 9")
    df_sd = sort_by_score(df_dc_pool.copy(), "Score Segundo Delantero")

    rankings = {
        "Portero": df_por,
        "Lateral izquierdo": df_li,
        "DFC Izquierdo": df_dfc_izq,
        "DFC Derecho": df_dfc_der,
        "Lateral derecho": df_ld,
        "MC Contención": df_mc_contencion,
        "MC Box to Box": df_mc_b2b,
        "MC Ofensivo": df_mc_ofensivo,
        "Extremo Izquierdo": df_ei,
        "Extremo Derecho": df_ed,
        "Delantero": df_dc,
        "Segundo Delantero": df_sd,
    }

    score_cols = {
        "Portero": "Score GK Total",
        "Lateral izquierdo": "Score Lateral Total",
        "DFC Izquierdo": "Score Central Total",
        "DFC Derecho": "Score Central Total",
        "Lateral derecho": "Score Lateral Total",
        "MC Contención": "Score MC Contención",
        "MC Box to Box": "Score MC Box-to-Box",
        "MC Ofensivo": "Score MC Ofensivo",
        "Extremo Izquierdo": "Score Extremos Total",
        "Extremo Derecho": "Score Extremos Total",
        "Delantero": "Score 9",
        "Segundo Delantero": "Score Segundo Delantero",
    }

    return rankings, score_cols





def _pct_border_color(pct):
    if pct is None:
        return "#999999"
    try:
        p = float(pct)
    except:
        return "#999999"

    if p >= 80:
        return "#2aa84a"   # verde
    if p >= 50:
        return "#f2c200"   # amarillo
    return "#d7263d"       # rojo

def truncate_text(txt, max_chars: int):
    """
    Corta el texto para que NO se desborde.
    Ej: "Aaron Yaakobishvili" -> "Aaron Yaakob..."
    """
    if txt is None:
        return ""
    s = str(txt).strip()
    if len(s) <= max_chars:
        return s
    # deja hueco para "..."
    return s[: max(0, max_chars - 3)].rstrip() + "..."



def draw_position_table(
    ax, x, y, title, rows,
    width=30,
    row_h=3.6,
    pct_w=5.0,
    pad=0.55,
    title_gap=2.0,
):
    """
    rows = [(jugador, equipo, minutos, percentil), ...]
    """
    n = len(rows)
    if n == 0:
        return

    total_h = n * row_h

    # --- TÍTULO ARRIBA ---
    card_x = x - width / 2
    title_box = FancyBboxPatch(
        (card_x, y + total_h/2 + title_gap - 1.3),
        width,
        2.4,
        boxstyle="round,pad=0.25,rounding_size=0.9",
        linewidth=1.2,
        edgecolor="#0e2841",
        facecolor="#0e2841",
        zorder=1000,
        clip_on=True
    )
    ax.add_patch(title_box)

    ax.text(
        x,
        y + total_h/2 + title_gap,
        str(title).upper(),
        ha="center",
        va="center",
        fontsize=8.1,   # 👈 MÁS PEQUEÑO
        fontweight="bold",
        color="white",
        zorder=1001,
        clip_on=True
    )

    # --- TARJETA BASE ---
    card = FancyBboxPatch(
        (x - width/2, y - total_h/2),
        width, total_h,
        boxstyle="round,pad=0.28,rounding_size=1.0",
        linewidth=0.9,
        edgecolor="#00000022",
        facecolor="white",
        alpha=0.99,
        zorder=999,
        clip_on=False
    )
    ax.add_patch(card)

    # ===== Columnas: Jugador | Equipo | Minutos | Percentil =====
    x_left = x - width/2
    x_right = x + width/2

    mins_w = 7.0
    x_pct_left = x_right - pct_w
    x_mins_left = x_pct_left - mins_w
    x_team_right = x_mins_left

    # Separadores verticales
    for x_sep in [x_team_right, x_mins_left, x_pct_left]:
        ax.plot(
            [x_sep, x_sep],
            [y - total_h/2, y + total_h/2],
            color="#00000022",
            linewidth=1,
            zorder=1001,
            clip_on=False
        )

    # Filas
    top_y = y + total_h/2
    for i, (jug, eq, mins, pct) in enumerate(rows):
        y_row_bottom = top_y - (i + 1) * row_h
        y_row_center = y_row_bottom + row_h / 2

        # Separador horizontal
        ax.plot(
            [x_left, x_right],
            [y_row_bottom, y_row_bottom],
            color="#00000022",
            linewidth=1,
            zorder=1001,
            clip_on=False
        )

        # Jugador
        ax.text(
            x_left + pad,
            y_row_center,
            truncate_text(jug, 14),
            ha="left",
            va="center",
            fontsize=6.2,   # 👈 MÁS PEQUEÑO
            fontweight="bold",
            color="#111",
            zorder=1002,
            clip_on=False
        )

        # Equipo
        ax.text(
            x_team_right - pad,
            y_row_center,
            truncate_text(eq, 14),
            ha="right",
            va="center",
            fontsize=5.8,   # 👈 MÁS PEQUEÑO
            color="#111",
            zorder=1002,
            clip_on=False
        )

        # Minutos
        mins_txt = ""
        if mins is not None and mins != "" and pd.notna(mins):
            try:
                mins_txt = f"{int(mins)}"
            except:
                mins_txt = str(mins)

        ax.text(
            (x_mins_left + x_pct_left) / 2,
            y_row_center,
            mins_txt,
            ha="center",
            va="center",
            fontsize=5.8,   # 👈 MÁS PEQUEÑO
            color="#111",
            zorder=1002,
            clip_on=False
        )

        # Percentil
        border = _pct_border_color(pct)
        badge = FancyBboxPatch(
            (x_pct_left + 0.55, y_row_bottom + 0.55),
            pct_w - 1.10,
            row_h - 1.10,
            boxstyle="round,pad=0.12,rounding_size=0.55",
            linewidth=2.0,
            edgecolor=border,
            facecolor="white",
            zorder=1003,
            clip_on=False
        )
        ax.add_patch(badge)

        ax.text(
            x_pct_left + pct_w / 2,
            y_row_center,
            "" if pct is None else str(int(pct)),
            ha="center",
            va="center",
            fontsize=7.0,   # 👈 MÁS PEQUEÑO
            fontweight="bold",
            color="#111",
            zorder=1004,
            clip_on=False
        )



#====================================
# Funcion que crea la imagen del campo
#====================================
def dibujar_campograma_defensivo(
    rankings,
    score_cols,
    temporada,
    liga_str,
    roles_cfg=None,
    map_pos_to_familia=None
):

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#d0f0c0",
        line_color="black"
    )
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_alpha(0)

    # ✅ MÁRGENES EXTRA para que NO SE CORTE nada
    # statsbomb suele ser x: 0-120, y:0-80
    ax.set_xlim(-6, 126)
    ax.set_ylim(-6, 86)

    posiciones_campo = {
        "Portero": (5, 40),
        "Lateral izquierdo": (25, 75),
        "DFC Izquierdo": (25, 55),
        "DFC Derecho": (25, 25),
        "Lateral derecho": (25, 2),
        "MC Contención": (60, 20),
        "MC Box to Box": (60, 60),
        "MC Ofensivo": (75, 40),
        "Extremo Izquierdo": (100, 75),
        "Delantero": (110, 50),
        "Segundo Delantero": (110, 27),
        "Extremo Derecho": (100, 2),
    }

    if all(df_pos.empty for df_pos in rankings.values()):
        ax.set_title(
            f"Sin datos (alineación completa) — {temporada} | {liga_str}",
            fontsize=14,
            fontweight="bold",
            color="white",
        )
        return fig

    for pos_nombre, df_pos in rankings.items():
        if df_pos.empty or pos_nombre not in posiciones_campo:
            continue

        x, y = posiciones_campo[pos_nombre]

        pct_col = None
        rol_activo = None
        if roles_cfg and map_pos_to_familia:
            familia = map_pos_to_familia.get(pos_nombre)
            orden_cfg = st.session_state.get("orden_campograma", {})
            rol_activo = orden_cfg.get(pos_nombre)
            if not rol_activo and familia:
                rol_activo = orden_cfg.get(familia)
            pct_col = roles_cfg.get(familia, {}).get(rol_activo) if familia and rol_activo else None

        if not pct_col:
            score_col = score_cols.get(pos_nombre)
            pct_col = f"Percentil {score_col}" if score_col else None

        top_df = df_pos.head(3)

        rows = []
        for _, r in top_df.iterrows():
            jugador = r.get("Jugador", "")
            equipo = r.get("Equipo", "")
            mins = r.get("Minutos jugados", None)

            pct = None
            if pct_col and pct_col in top_df.columns:
                val = r.get(pct_col, None)
                if pd.notna(val):
                    pct = int(val)

            rows.append((jugador, equipo, mins, pct))


        # ✅ Caja pequeña + título arriba
        draw_position_table(
            ax=ax,
            x=x, y=y,
            title=f"{('Seg. delantero' if pos_nombre == 'Segundo Delantero' else pos_nombre)} - {('Seg delantero' if rol_activo == 'Segundo delantero' else rol_activo)}" if rol_activo else ("Seg. delantero" if pos_nombre == "Segundo Delantero" else pos_nombre),
            rows=rows,
            width=30,     # 👈 más pequeña
            row_h=3.6,    # 👈 más compacta
            pct_w=5.0
        )

    ax.set_title(
        f"Alineación completa — {temporada} | {liga_str}",
        fontsize=14,
        fontweight="bold",
        color="white"
    )

    return fig
#====================================
# Funcion que crea la imagen del campo
#====================================
def dibujar_campograma_defensivo_prueba(
    rankings,
    score_cols,
    temporada,
    liga_str,
    pos_name,
    rol_seleccionado,
    roles_cfg=None,
    map_pos_to_familia=None
):

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#d0f0c0",
        line_color="black"
    )
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_alpha(0)

    # ✅ MÁRGENES EXTRA para que NO SE CORTE nada
    # statsbomb suele ser x: 0-120, y:0-80
    ax.set_xlim(-6, 126)
    ax.set_ylim(-6, 86)

    posiciones_campo = {
        "Portero": (5, 40),
        "Lateral izquierdo": (25, 75),
        "DFC Izquierdo": (25, 55),
        "DFC Derecho": (25, 25),
        "Lateral derecho": (25, 2),
        "MC Contención": (60, 20),
        "MC Box to Box": (60, 60),
        "MC Ofensivo": (75, 40),
        "Extremo Izquierdo": (100, 75),
        "Delantero": (110, 40),
        "Segundo Delantero": (108, 24),
        "Extremo Derecho": (100, 2),
    }

    if all(df_pos.empty for df_pos in rankings.values()):
        ax.set_title(
            f"Sin datos (alineación completa) — {temporada} | {liga_str}",
            fontsize=14,
            fontweight="bold",
            color="white",
        )
        return fig

    for pos_nombre, df_pos in rankings.items():
        if df_pos.empty or pos_nombre not in posiciones_campo:
            continue

        x, y = posiciones_campo[pos_nombre]

        pct_col = None
        rol_activo = None
        if roles_cfg and map_pos_to_familia:
            familia = map_pos_to_familia.get(pos_nombre)
            orden_cfg = st.session_state.get("orden_campograma", {})
            rol_activo = orden_cfg.get(pos_nombre)
            if not rol_activo and familia:
                rol_activo = orden_cfg.get(familia)
            pct_col = roles_cfg.get(familia, {}).get(rol_activo) if familia and rol_activo else None

        if not pct_col:
            score_col = score_cols.get(pos_nombre)
            pct_col = f"Percentil {score_col}" if score_col else None

        top_df = df_pos.head(3)

        rows = []
        for _, r in top_df.iterrows():
            jugador = r.get("Jugador", "")
            equipo = r.get("Equipo", "")
            mins = r.get("Minutos jugados", None)

            pct = None
            if pct_col and pct_col in top_df.columns:
                val = r.get(pct_col, None)
                if pd.notna(val):
                    pct = int(val)

            rows.append((jugador, equipo, mins, pct))

        titulo = f"{('Seg. delantero' if pos_nombre == 'Segundo Delantero' else pos_nombre)} - {('Seg delantero' if rol_activo == 'Segundo delantero' else rol_activo)}" if rol_activo else ("Seg. delantero" if pos_nombre == "Segundo Delantero" else pos_nombre)
        
        # ✅ Caja pequeña + título arriba
        draw_position_table(
            ax=ax,
            x=x, y=y,
            title=titulo,
            rows=rows,
            width=30,     # 👈 más pequeña
            row_h=3.6,    # 👈 más compacta
            pct_w=5.0
        )

    ax.set_title(
        f"Alineación completa — {temporada} | {liga_str}",
        fontsize=14,
        fontweight="bold",
        color="white"
    )

    return fig

def export_campograma_pdf(fig, df_tabla, cols_por_pagina=6, jugador_col="Jugador"):
    buffer = BytesIO()

    # Guardar la figura del campograma en un buffer
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    img_buffer.seek(0)

    # Crear PDF en memoria
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20)
    elements = []

    # ==============================
    # Página 1: Campograma
    # ==============================
    elements.append(Table([["Campograma"]], colWidths=landscape(A4)[0]-40))
    elements.append(Image(img_buffer, width=landscape(A4)[0]-40, height=(landscape(A4)[1]-100)))
    elements.append(PageBreak())

    # ==============================
    # Páginas siguientes: Tabla
    # ==============================
    other_cols = [c for c in df_tabla.columns if c != jugador_col]
    total_cols = len(other_cols)

    for start in range(0, total_cols, cols_por_pagina-1):
        end = start + (cols_por_pagina-1)
        cols_actuales = [jugador_col] + other_cols[start:end]
        df_sub = df_tabla[cols_actuales]

        data = [df_sub.columns.tolist()] + df_sub.values.tolist()
        table = Table(data)

        # Estilo de tabla
        table_style = TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,0), 4),
            ('GRID', (0,0), (-1,-1), 0.25, colors.black),
        ])

        # Rotar encabezados (excepto jugador)
        for i, col_name in enumerate(df_sub.columns):
            if col_name != jugador_col:
                table_style.add('ROTATE', (i,0), (i,0), 90)

        # ==============================
        # Aplicar colores a valores >= 85
        # ==============================
        color_map = {}
        for col_name in df_sub.columns:
            if "Percentil Score GK Portero" in col_name or "Genérico" in col_name:
                color_map[col_name] = colors.HexColor("#5B9BD5")  # azul
            elif "Percentil Score GK Atajador" in col_name or "Defensivo" in col_name or "Contención" in col_name:
                color_map[col_name] = colors.HexColor("#FF0000")  # rojo
            elif "Percentil Score GK Juego de Pies" in col_name or "Ofensivo" in col_name or "Combinativo" in col_name:
                color_map[col_name] = colors.HexColor("#00B050")  # verde
            elif "Box-to-Box" in col_name:
                color_map[col_name] = colors.HexColor("#7030A0")  # morado

        for col_idx, col_name in enumerate(df_sub.columns):
            if col_name == jugador_col:
                continue
            for row_idx, val in enumerate(df_sub[col_name], start=1):  # start=1 porque fila 0 = headers
                try:
                    if float(val) >= 85:
                        table_style.add('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), color_map.get(col_name))
                except:
                    continue  # ignora valores que no sean numéricos

        table.setStyle(table_style)

        # Ajustar anchos de columna
        page_width = landscape(A4)[0] - 40
        col_width = page_width / len(df_sub.columns)
        table._argW = [col_width]*len(df_sub.columns)

        elements.append(table)
        elements.append(PageBreak())

    # ==============================
    # Construir PDF en memoria
    # ==============================
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# =========================
# FUNCIÓN PRINCIPAL
# =========================
def app():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.header("Campogramas y Rankings por Posición")
    # Datos cargados de la base de datos
    df = st.session_state.data

    st.sidebar.subheader("Filtros")

    # === Temporada (por defecto 2025) ===
    temporadas = sorted(df["Temporada"].dropna().unique())
    default_index = 0
    for i, t in enumerate(temporadas):
        if str(t) == "2025":
            default_index = i
            break

    temporada_sel = st.sidebar.selectbox("Temporada", temporadas, index=default_index)

    # ========= Selección Categoría_Liga =========
    if "Categoría_Liga" in df.columns and df["Categoría_Liga"].notna().any():
        opciones_categoria = sorted(df[df["Temporada"] == temporada_sel]["Categoría_Liga"].dropna().unique())
        categoria_sel = st.sidebar.multiselect(
            "Categoría de Liga",
            options=opciones_categoria,
            default=[]
        )
    else:
        categoria_sel = []

    # ========= Selección Liga / Competición =========
    if "Nombre_Liga" in df.columns and df["Nombre_Liga"].notna().any():
        opciones_liga = sorted(df[df["Temporada"] == temporada_sel]["Nombre_Liga"].dropna().unique())

        default_ligas = []
        for liga_nombre in opciones_liga:
            nombre_upper = str(liga_nombre).upper()
            if "LA LIGA" in nombre_upper or "LALIGA" in nombre_upper:
                default_ligas = [liga_nombre]
                break
        if not default_ligas and len(opciones_liga) > 0:
            default_ligas = [opciones_liga[0]]

        liga_sel = st.sidebar.multiselect(
            "Liga / Competición",
            options=opciones_liga,
            default=default_ligas
        )
    else:
        liga_sel = []

    # ======= CONSTRUCCIÓN AUTOMÁTICA DEL POOL DE PERCENTILES =======
    # Siempre que cambie algún filtro, recalculamos
    df_pool = construir_pool_percentiles(df, temporada_sel, categoria_sel, liga_sel)
    st.session_state["df_pool_percentiles"] = df_pool
    st.session_state["pool_info"] = {
        "temporada": temporada_sel,
        "categoria": categoria_sel,
        "liga": liga_sel,
        "n_jugadores": len(df_pool),
    }
    #Datos del pool actual (ya percentilizado)
    df_pool = st.session_state.get("df_pool_percentiles", pd.DataFrame())
    pool_info = st.session_state.get("pool_info", {})

    if df_pool.empty:
        st.warning("No hay datos para esa combinación de Temporada / Categoría / Liga.")
        return

    # ========= Sliders de segmentación (NO afectan al cálculo de percentiles) =========
    # Rango de minutos sobre df_pool (que YA está percentilizado)
    min_minutos = int(df_pool["Minutos jugados"].min())
    max_minutos = int(df_pool["Minutos jugados"].max())
    
    # Inicializamos en session_state si no existe
    if "filtro_minutos" not in st.session_state:
        st.session_state["filtro_minutos"] = (min_minutos, max_minutos)
    st.sidebar.subheader("Segmentación (no afecta percentiles)")
    minutos_min_sel, minutos_max_sel = st.sidebar.slider(
        "Minutos",
        min_value=min_minutos,
        max_value=max_minutos,
        step=90,
        key="filtro_minutos"
    )
   
    # Edad
    if "Edad" in df_pool.columns and df_pool["Edad"].notna().any():
        min_edad = int(df_pool["Edad"].min())
        max_edad = int(df_pool["Edad"].max())

        if "filtro_edad" not in st.session_state:
            st.session_state["filtro_edad"] = (min_edad, max_edad)

        edad_min_sel, edad_max_sel = st.sidebar.slider(
            "Edad",
            min_value=min_edad,
            max_value=max_edad,
            step=1,
            format="%d",
            key="filtro_edad"
        )
    else:
        min_edad, max_edad = None, None
        edad_min_sel, edad_max_sel = None, None

    # Valor mercado
    if "Valor_Mercado" in df_pool.columns and df_pool["Valor_Mercado"].notna().any():
        min_valor = int(df_pool["Valor_Mercado"].min())
        max_valor = int(df_pool["Valor_Mercado"].max())

        if "filtro_valor" not in st.session_state:
            st.session_state["filtro_valor"] = (min_valor, max_valor)

        valor_min_sel, valor_max_sel = st.sidebar.slider(
            "Valor de Mercado ",
            min_value=min_valor,
            max_value=max_valor,
            step=100,
            format="%d",
            key="filtro_valor"
        )
    else:
        min_valor, max_valor = None, None
        valor_min_sel, valor_max_sel = None, None
    #Pie bueno
    if "Pie bueno" in df_pool.columns and df_pool["Pie bueno"].notna().any():
        opciones_pie_bueno = sorted(df_pool["Pie bueno"].dropna().unique())

        if "filtro_pie_bueno" not in st.session_state:
            st.session_state["filtro_pie_bueno"] = []

        pie_bueno_sel = st.sidebar.multiselect(
            "Pie bueno",
            options=opciones_pie_bueno,
            key="filtro_pie_bueno"
        )
    else:
        opciones_pie_bueno = []
        pie_bueno_sel = []

    # Nacionalidad
    if "Nacionalidad" in df_pool.columns and df_pool["Nacionalidad"].notna().any():
        opciones_nacionalidad = sorted(df_pool["Nacionalidad"].dropna().unique())

        if "filtro_nacionalidad" not in st.session_state:
            st.session_state["filtro_nacionalidad"] = []

        nacionalidad_sel = st.sidebar.multiselect(
            "Nacionalidad",
            options=opciones_nacionalidad,
            key="filtro_nacionalidad"
        )
    else:
        opciones_nacionalidad = []
        nacionalidad_sel = []

    # =========================
    # Equipo
    # =========================
    if "Equipo" in df_pool.columns and df_pool["Equipo"].notna().any():
        opciones_equipo = sorted(df_pool["Equipo"].dropna().unique())

        if "filtro_equipo" not in st.session_state:
            st.session_state["filtro_equipo"] = []

        equipo_sel = st.sidebar.multiselect(
            "Equipo",
            options=opciones_equipo,
            key="filtro_equipo"
        )
    else:
        opciones_equipo = []
        equipo_sel = []


    # ===== Función de callback para resetear filtros de segmentación =====
    def reset_segmentacion():
        # Minutos
        if "Minutos jugados" in df_pool.columns:
            min_m = int(df_pool["Minutos jugados"].min())
            max_m = int(df_pool["Minutos jugados"].max())
            st.session_state["filtro_minutos"] = (min_m, max_m)

        # Edad
        if "Edad" in df_pool.columns and df_pool["Edad"].notna().any():
            min_e = int(df_pool["Edad"].min())
            max_e = int(df_pool["Edad"].max())
            st.session_state["filtro_edad"] = (min_e, max_e)

        # Valor de mercado
        if "Valor_Mercado" in df_pool.columns and df_pool["Valor_Mercado"].notna().any():
            min_v = int(df_pool["Valor_Mercado"].min())
            max_v = int(df_pool["Valor_Mercado"].max())
            st.session_state["filtro_valor"] = (min_v, max_v)
        # Pie bueno
        if "filtro_pie_bueno" in st.session_state:
            st.session_state["filtro_pie_bueno"]=[]
        # Nacionalidad
        if "filtro_nacionalidad" in st.session_state:
            st.session_state["filtro_nacionalidad"] = []

        # Equipo
        if "filtro_equipo" in st.session_state:
            st.session_state["filtro_equipo"] = []


        # Fin de contrato
        if "Fin de contrato" in df_pool.columns and df_pool["Fin de contrato"].notna().any():
            fc_key = f"fin_contrato_{temporada_sel}_{hash(tuple(categoria_sel))}_{hash(tuple(liga_sel))}"
            if fc_key in st.session_state:
                st.session_state[fc_key] = []

    # Fin de contrato (sobre pool)
    if "Fin de contrato" in df_pool.columns and df_pool["Fin de contrato"].notna().any():
        opciones_fin_contrato = sorted(df_pool["Fin de contrato"].dropna().unique())

        fin_contrato_key = f"fin_contrato_{temporada_sel}_{hash(tuple(categoria_sel))}_{hash(tuple(liga_sel))}"

        if fin_contrato_key not in st.session_state:
            st.session_state[fin_contrato_key] = []

        fin_contrato_sel = st.sidebar.multiselect(
            "Fin de Contrato",
            options=opciones_fin_contrato,
            key=fin_contrato_key
        )

        fin_contrato_sel = [str(x) for x in fin_contrato_sel]
    else:
        opciones_fin_contrato = []
        fin_contrato_key = None
        fin_contrato_sel = []

    # 🔁 Botón en el sidebar, debajo de "Fin de contrato"
    st.sidebar.button(
        "🔁 Filtros",
        on_click=reset_segmentacion
    )

    # ====== SEGMENTACIÓN FINAL (NO recalcula percentiles) ======
    df_filtrado = df_pool.copy()

    df_filtrado = df_filtrado[
        (df_filtrado["Minutos jugados"] >= minutos_min_sel) &
        (df_filtrado["Minutos jugados"] <= minutos_max_sel)
    ]

    if edad_min_sel is not None:
        df_filtrado = df_filtrado[
            df_filtrado["Edad"].isna() |
            df_filtrado["Edad"].between(edad_min_sel, edad_max_sel)
        ]

    if valor_min_sel is not None:
        df_filtrado = df_filtrado[
            df_filtrado["Valor_Mercado"].isna() |
            df_filtrado["Valor_Mercado"].between(valor_min_sel, valor_max_sel)
        ]

    if fin_contrato_sel:
        df_filtrado = df_filtrado[
            df_filtrado["Fin de contrato"].isin(fin_contrato_sel)
        ]
    if pie_bueno_sel:
        df_filtrado= df_filtrado[
            df_filtrado["Pie bueno"].isin(pie_bueno_sel)
        ]
    if nacionalidad_sel:
        df_filtrado = df_filtrado[
            df_filtrado["Nacionalidad"].isin(nacionalidad_sel)
        ]

        # Filtro por Equipo
    if equipo_sel:
        df_filtrado = df_filtrado[
            df_filtrado["Equipo"].isin(equipo_sel)
        ]



    if pool_info.get("liga"):
        liga_str = ", ".join(pool_info["liga"])
    else:
        liga_str = "Todas las ligas (pool temporada)"


    if df_filtrado.empty:
        st.warning("No hay datos tras segmentar por Minutos / Edad / Valor / Contrato.")
        return

    # ===== Rankings y 11 ideal =====
    rankings, score_cols = rankings_defensivos(df_filtrado)

    with st.expander("Recuento de jugadores por posición"):
        for k, v in rankings.items():
            st.write(f"{k}: {len(v)} jugadores")

    
    

    # Reservo el sitio para dibujar el campograma
    campograma_slot = st.empty()
    
    campograma_slot.spinner = campograma_slot.info("Cargando campograma...")
    # =========================
    # DETALLE POR POSICIÓN
    # =========================
    def mostrar_posicion(rankings, pos_name, columnas, subheader, key,rol,pos,cantidad,roles_cfg):

        st.subheader(subheader)
        df_pos = rankings.get(pos_name, pd.DataFrame())
        cols_exist = [c for c in columnas if c in df_pos.columns]

        if not cols_exist:
            st.write(f"No hay columnas de {subheader.lower()} disponibles en el dataset.")
            return

        if df_pos.empty:
            tabla = pd.DataFrame(columns=cols_exist)
            mostrar_tabla_aggrid(rol_seleccionado=rol, df_tabla=tabla, key=key, df_base=df_pos)
            return

        df_work = df_pos.copy()

        # -------- ORDENACIÓN SEGÚN ROL PASADO --------
        if rol and roles_cfg:
            col_orden = roles_cfg[pos].get(rol) 
            if col_orden and col_orden in df_work.columns:
                df_work = df_work.sort_values(col_orden, ascending=False)

        # -------- TOP N --------
        tabla = df_work[cols_exist].head(cantidad)

        return mostrar_tabla_aggrid(rol_seleccionado=rol, df_tabla=tabla, key=key, df_base=df_pos)   
    
    #Map para relacionar posición con familia (para luego sacar el rol seleccionado en la configuración)
    MAP_POS_TO_FAMILIA = cargar_map_pos_to_familia()
    #Roles para ordenar el campograma (si se ha seleccionado alguno en la tabla)
    roles_cfg = cargar_roles_cfg()
    
    # Función para convertir un DataFrame a bytes de Excel (para el botón de descarga)
    def df_to_excel_bytes(df):
        buffer = BytesIO()
        
        # Crear el writer con xlsxwriter
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Datos")
            workbook = writer.book
            worksheet = writer.sheets["Datos"]
            
            formato_azul = workbook.add_format({'bg_color': '#5B9BD5', 'font_color': '#FFFFFF'})
            formato_rojo = workbook.add_format({'bg_color': '#FF0000', 'font_color': '#FFFFFF'})
            formato_verde = workbook.add_format({'bg_color': '#00B050', 'font_color': '#FFFFFF'})
            formato_morado = workbook.add_format({'bg_color': '#7030A0', 'font_color': '#FFFFFF'})
            
            for i, col in enumerate(df.columns):
                col_letter = chr(65 + i)  # Convierte a letra A, B, C...
                
                # GK específico
                if col == "Percentil Score GK Portero":
                    worksheet.conditional_format(f"{col_letter}2:{col_letter}{len(df)+1}", 
                                                {'type': 'cell', 'criteria': '>=', 'value': 85, 'format': formato_azul})
                elif col == "Percentil Score GK Atajador":
                    worksheet.conditional_format(f"{col_letter}2:{col_letter}{len(df)+1}", 
                                                {'type': 'cell', 'criteria': '>=', 'value': 85, 'format': formato_rojo})
                elif col == "Percentil Score GK Juego de Pies":
                    worksheet.conditional_format(f"{col_letter}2:{col_letter}{len(df)+1}", 
                                                {'type': 'cell', 'criteria': '>=', 'value': 85, 'format': formato_verde})
                
                # Percentiles genéricos y por rol
                elif "Genérico" in col:
                    worksheet.conditional_format(f"{col_letter}2:{col_letter}{len(df)+1}", 
                                                {'type': 'cell', 'criteria': '>=', 'value': 85, 'format': formato_azul})
                elif "Defensivo" in col or "Contención" in col:
                    worksheet.conditional_format(f"{col_letter}2:{col_letter}{len(df)+1}", 
                                                {'type': 'cell', 'criteria': '>=', 'value': 85, 'format': formato_rojo})
                elif "Ofensivo" in col or "Combinativo" in col:
                    worksheet.conditional_format(f"{col_letter}2:{col_letter}{len(df)+1}", 
                                                {'type': 'cell', 'criteria': '>=', 'value': 85, 'format': formato_verde})
                elif "Box-to-Box" in col:
                    worksheet.conditional_format(f"{col_letter}2:{col_letter}{len(df)+1}", 
                                                {'type': 'cell', 'criteria': '>=', 'value': 85, 'format': formato_morado})
            
        buffer.seek(0)
        return buffer.getvalue()
    

    def crear_botones_export(df_tabla, fig, nombre_archivo, temporada_sel):
        """
        Crea botones de exportación a Excel y PDF para una tabla y un campograma.
        
        Parámetros:
        - df_tabla: pd.DataFrame de la tabla que se quiere exportar.
        - fig: matplotlib.figure.Figure del campograma.
        - nombre_archivo: str, usado para nombres de archivo y keys de Streamlit.
        - temporada_sel: str o int, temporada para el nombre del archivo.
        """
        col_excel, col_pdf, _ = st.columns([2, 2, 7])
        if df_tabla is not None and not df_tabla.empty:
            # Botón para Excel
            with col_excel:
                st.download_button(
                    label=f"📊 Exportar a Excel",
                    data=df_to_excel_bytes(df_tabla),
                    file_name=f"{nombre_archivo}_{temporada_sel}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"export_{nombre_archivo}_excel"
                )
            with col_pdf:
                # Botón para PDF
                st.download_button(
                    label=f"📄 Exportar a PDF",
                    data=export_campograma_pdf(fig, df_tabla),
                    file_name=f"campograma_tabla_{nombre_archivo}.pdf",
                    mime="application/pdf",
                    key=f"export_{nombre_archivo}_pdf"
                )


    # Función para ordenar los rankings según el rol seleccionado en la configuración del campograma
    # Esto afecta al orden de los jugadores en el campograma, pero NO a las tablas (que se ordenan según el rol seleccionado en cada tabla, que puede ser distinto o ninguno)
    def ordenar_rankings_para_campograma(rankings, roles_cfg):

        rankings_ordenados = {}
        for pos, df_pos in rankings.items():
            if df_pos.empty:
                rankings_ordenados[pos] = df_pos
                continue

            familia = MAP_POS_TO_FAMILIA.get(pos)

            # Si por lo que sea no está mapeado, no ordenamos
            if not familia:
                rankings_ordenados[pos] = df_pos
                continue

            orden_cfg = st.session_state.get("orden_campograma", {})
            rol = orden_cfg.get(pos)
            if not rol:
                rol = orden_cfg.get(familia)
            col = roles_cfg.get(familia, {}).get(rol)

            if col and col in df_pos.columns:
                rankings_ordenados[pos] = df_pos.sort_values(col, ascending=False)
            else:
                rankings_ordenados[pos] = df_pos
        return rankings_ordenados
    
    # =========================
    # Función para procesar cada posición (tabla + campograma + export)
    # En esta función se muestra la tabla de la posición, se ordena el ranking según el rol seleccionado
    #  en la configuración del campograma y se dibuja el campograma con ese orden. 
    # Además, se crean los botones de export para esa posición.
    # =========================
    def procesar_posicion(rankings, pos_name, pos_label, columnas, roles_cfg, score_cols, pool_info, temporada_sel, liga_str, rol_options, key_prefix):
        """
        Muestra el selectbox de rol, slider de cantidad, tabla, campograma y botones de export para una posición.
        """
        col_rol, col_slider, _ = st.columns([3, 4, 3])

        # Selectbox rol
        with col_rol:
            rol_default_por_pos = {
                "MC Contención": "Contención",
                "MC Box to Box": "Box to Box",
                "MC Ofensivo": "Ofensivo",
                "Delantero": "9",
            }
            key_rol = f"rol_tabla_{key_prefix}"
            rol_default = rol_default_por_pos.get(pos_label, rol_options[0])
            if "MC Contenci" in pos_label:
                rol_default = next((r for r in rol_options if "Contenci" in r), rol_default)
            elif "MC Box to Box" in pos_label:
                rol_default = "Box to Box"
            elif "MC Ofensivo" in pos_label:
                rol_default = "Ofensivo"
            if key_rol not in st.session_state and rol_default in rol_options:
                st.session_state[key_rol] = rol_default

            rol_seleccionado = st.selectbox(
                "Ordenar por rol",
                options=rol_options,
                key=key_rol
            )
            st.session_state.orden_campograma[pos_label] = rol_seleccionado

        # Slider cantidad
        with col_slider:
            cantidad = st.slider(
                "Número de jugadores",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                key=f"slider_{key_prefix}"
            )

        # Mostrar tabla
        df_tabla = mostrar_posicion(
            rankings,
            pos_name=pos_label,
            columnas=columnas,
            subheader=pos_label,
            key=f"tabla_{key_prefix}",
            rol=rol_seleccionado,
            pos=pos_name.upper(),
            cantidad=cantidad,
            roles_cfg=roles_cfg
        )
        
        # Ordenar rankings
        ranking = ordenar_rankings_para_campograma(rankings, roles_cfg)

        # Dibujar campograma
        fig = dibujar_campograma_defensivo_prueba(
            ranking,
            score_cols,
            pool_info.get("temporada", temporada_sel),
            liga_str,
            pos_name=pos_name.upper(),
            rol_seleccionado=rol_seleccionado,
            roles_cfg=roles_cfg,
            map_pos_to_familia=MAP_POS_TO_FAMILIA
        )

        # Crear botones export
        crear_botones_export(df_tabla, fig, key_prefix, temporada_sel)

    # ===== Columnas =====
    tabs = st.tabs(["Porteros","Defensas","Mediocentros","Extremos","Delanteros"])
    columnas_cargadas=st.session_state.columnas
    with tabs[0]:
        st.subheader("Porteros")
        procesar_posicion(rankings=rankings, pos_name="PORTERO",pos_label="Portero",columnas=columnas_cargadas.get("Portero", []),
        roles_cfg=roles_cfg, score_cols=score_cols,pool_info=pool_info,temporada_sel=temporada_sel,
        liga_str=liga_str, rol_options=["Portero", "Atajador", "Juego de pies"], key_prefix="porteros")
    
    with tabs[1]:
        st.subheader("Defensas")
        sub_pos_list = st.multiselect(
            "Tipo de defensa",
            ["Lateral Izquierdo", "DFC Izquierdo", "DFC Derecho", "Lateral Derecho"],
            default=["Lateral Izquierdo"],
            key="defensas_subpos"
        )
        defensas_config = {
            "Lateral Izquierdo": {
                "key": "laterales_izq",
                "pos_name": "Lateral izquierdo",
                "familia": "LATERAL",
                "columnas": columnas_cargadas["Lateral"],
                "roles": ["Genérico", "Defensivo", "Ofensivo"]
            },
            "Lateral Derecho": {
                "key": "laterales_der",
                "pos_name": "Lateral derecho",
                "familia": "LATERAL",
                "columnas": columnas_cargadas["Lateral"],
                "roles": ["Genérico", "Defensivo", "Ofensivo"]
            },
            "DFC Izquierdo": {
                "key": "dfc_izq",
                "pos_name": "DFC Izquierdo",
                "familia": "DFC",
                "columnas": columnas_cargadas["DFC"],
                "roles": ["Genérico", "Defensivo", "Combinativo"]
            },
            "DFC Derecho": {
                "key": "dfc_der",
                "pos_name": "DFC Derecho",
                "familia": "DFC",
                "columnas": columnas_cargadas["DFC"],
                "roles": ["Genérico", "Defensivo", "Combinativo"]
            }
        }
        for sub_pos in sub_pos_list:
            config = defensas_config[sub_pos]
            with st.container():
                procesar_posicion(rankings=rankings,pos_name=config["familia"],pos_label=config["pos_name"],
                    columnas=config["columnas"], roles_cfg=roles_cfg,score_cols=score_cols,pool_info=pool_info,
                    temporada_sel=temporada_sel,liga_str=liga_str,rol_options=config["roles"],key_prefix=config["key"])
                
    with tabs[2]:
        st.subheader("Mediocentros")
        sub_pos_list = st.multiselect(
            "Tipo de mediocentro",
            ["MC Contención", "MC Box to Box", "MC Ofensivo"],
            default=["MC Contención"],
            key="mediocentros_subpos"
        )
        mc_config = {
            "MC Contención": {
                "key": "mc_contencion",
                "columnas": columnas_cargadas.get("MC_Contencion", [])
            },
            "MC Box to Box": {
                "key": "mc_box",
                "columnas": columnas_cargadas.get("MC_BoxtoBox", [])
            },
            "MC Ofensivo": {
                "key": "mc_ofensivo",
                "columnas": columnas_cargadas.get("MC_Ofensivo", [])
            }
        }
        for sub_pos in sub_pos_list:
            config = mc_config[sub_pos]
            with st.container():
                procesar_posicion(rankings=rankings,pos_name="MC", pos_label=sub_pos,columnas=config["columnas"],roles_cfg=roles_cfg,
                    score_cols=score_cols,pool_info=pool_info,temporada_sel=temporada_sel,liga_str=liga_str,
                    rol_options=["Genérico", "Contención", "Ofensivo", "Box to Box"],key_prefix=config["key"])

                
    with tabs[3]:
        st.subheader("Extremos")
        sub_pos_list = st.multiselect(
            "Tipo de extremo",
            ["Extremo Izquierdo", "Extremo Derecho"],
            default=["Extremo Izquierdo"],
            key="extremos_subpos"
        )
        extremo_keys = {
            "Extremo Izquierdo": "extremos_izq",
            "Extremo Derecho": "extremos_der"
        }
        for sub_pos in sub_pos_list:
            # Cada sub_pos en un contenedor para separar layout
            with st.container():
                procesar_posicion(rankings=rankings,pos_name="EXTREMO",pos_label=sub_pos,columnas=columnas_cargadas["Extremos"],
                    roles_cfg=roles_cfg,score_cols=score_cols,pool_info=pool_info,temporada_sel=temporada_sel,
                    liga_str=liga_str,rol_options=["Genérico", "Wide Out", "Incorporación", "Combinativo"],
                    key_prefix=extremo_keys[sub_pos])
                    
    with tabs[4]:
        procesar_posicion(rankings=rankings,pos_name="DELANTERO",pos_label="Delantero",columnas=columnas_cargadas["Delantero"],
        roles_cfg=roles_cfg,score_cols=score_cols,pool_info=pool_info, temporada_sel=temporada_sel,
        liga_str=liga_str,rol_options=["Delantero", "9", "Segundo delantero"],key_prefix="delanteros")
   
    # =========================
    # CAMPOGRAMA COMPLETO
    # Ordeno el campogrma en funcion de las checkboxes marcadas en cada posición 
    # Lo hago al final del script para evitar problemas de que no se actualice 
    # El campograma
    # Luego de tener los rankings ordenados dibujo el campograma en el espacio 
    # que le he reservado con st.empty()
    # =========================
    rankings_ordenados = ordenar_rankings_para_campograma(rankings, roles_cfg)
    fig = dibujar_campograma_defensivo(
        rankings_ordenados,
        score_cols,
        pool_info.get("temporada", temporada_sel),
        liga_str,
        roles_cfg=roles_cfg,
        map_pos_to_familia=MAP_POS_TO_FAMILIA
    )
        
    campograma_slot.pyplot(fig, use_container_width=True)
    
    
# Llamada a la app de Streamlit
app()
