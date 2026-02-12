# -*- coding: utf-8 -*-
import os

# ⚠️ Anti-crash BLAS (ANTES de numpy/sklearn)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import zipfile
import base64
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import cm, colors as mcolors
import json


# Medida de seguridad implementada para evitar acceder a la pagina sin haberse cargado la base de datos

if not st.session_state.get("bd_cargada", False):
    st.warning("⚠️ Primero debes cargar la base de datos en Inicio")
    st.stop()

# ======================================================
# STREAMLIT LAYOUT (maximo ancho + menos margenes)
# ======================================================
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 0.35rem !important;
        padding-bottom: 0.35rem !important;
        padding-left: 0.35rem !important;
        padding-right: 0.35rem !important;
        max-width: 100% !important;
    }
    section[data-testid="stSidebar"] { width: 340px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# (1) GRID: centrado + estilos base
# ======================================================
st.markdown(
    """
    <style>
    div[data-testid="stDataFrame"] div[role="gridcell"],
    div[data-testid="stDataFrame"] div[role="columnheader"],
    div[data-testid="stDataEditor"] div[role="gridcell"],
    div[data-testid="stDataEditor"] div[role="columnheader"]{
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
    }

    div[data-testid="stDataFrame"] div[role="gridcell"] *,
    div[data-testid="stDataFrame"] div[role="columnheader"] *,
    div[data-testid="stDataEditor"] div[role="gridcell"] *,
    div[data-testid="stDataEditor"] div[role="columnheader"] *{
        text-align: center !important;
        margin: 0 auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# (2) Rowheader (índice) - oculto (ya no queremos index)
# ======================================================
st.markdown(
    """
    <style>
    div[data-testid="stDataFrame"] [role="rowheader"],
    div[data-testid="stDataEditor"] [role="rowheader"]{
        display: none !important;
        width: 0 !important;
        min-width: 0 !important;
        max-width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        border: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# (3) FONDO + TEXTO BLANCO + ANCHOS (Métrica ancha)
# ======================================================
st.markdown(
    """
    <style>
    div[data-testid="stDataFrame"] [role="gridcell"],
    div[data-testid="stDataFrame"] [role="columnheader"],
    div[data-testid="stDataFrame"] [role="rowheader"],
    div[data-testid="stDataEditor"] [role="gridcell"],
    div[data-testid="stDataEditor"] [role="columnheader"],
    div[data-testid="stDataEditor"] [role="rowheader"]{
        color: white !important;
    }

    div[data-testid="stDataFrame"] [role="gridcell"],
    div[data-testid="stDataEditor"] [role="gridcell"]{
        background: rgba(0,0,0,0.35) !important;
    }

    div[data-testid="stDataFrame"] [role="columnheader"],
    div[data-testid="stDataEditor"] [role="columnheader"]{
        background: rgba(0,0,0,0.55) !important;
        font-weight: 700 !important;
    }

    /* Columna 1 (Métrica) */
    div[data-testid="stDataFrame"] [role="columnheader"][aria-colindex="1"],
    div[data-testid="stDataFrame"] [role="gridcell"][aria-colindex="1"],
    div[data-testid="stDataEditor"] [role="columnheader"][aria-colindex="1"],
    div[data-testid="stDataEditor"] [role="gridcell"][aria-colindex="1"]{
        flex: 0 0 680px !important;
        width: 680px !important;
        min-width: 680px !important;
        max-width: 680px !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding-left: 10px !important;
    }

    /* resto columnas */
    div[data-testid="stDataFrame"] [role="columnheader"]:not([aria-colindex="1"]),
    div[data-testid="stDataFrame"] [role="gridcell"]:not([aria-colindex="1"]),
    div[data-testid="stDataEditor"] [role="columnheader"]:not([aria-colindex="1"]),
    div[data-testid="stDataEditor"] [role="gridcell"]:not([aria-colindex="1"]){
        flex: 0 0 140px !important;
        width: 140px !important;
        min-width: 140px !important;
        max-width: 140px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# (4) HEADER WRAP
# ======================================================
st.markdown(
    """
    <style>
    div[data-testid="stDataFrame"] [role="columnheader"],
    div[data-testid="stDataEditor"] [role="columnheader"]{
        white-space: normal !important;
        overflow: hidden !important;
        text-overflow: clip !important;
        line-height: 1.05 !important;
        padding-top: 8px !important;
        padding-bottom: 8px !important;
        height: 76px !important;
        min-height: 76px !important;
        max-height: 76px !important;
        align-items: center !important;
    }

    div[data-testid="stDataFrame"] [role="columnheader"] *,
    div[data-testid="stDataEditor"] [role="columnheader"] *{
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
        line-height: 1.05 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# FONDO APP
# Cambiada la ruta para que funcione localmente
# ======================================================
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
        margin: 0; padding: 0; height: 100%;
        overscroll-behavior: none;
        background: transparent !important;
    }}
    .main .block-container {{
        position: relative; z-index: 2;
        background: transparent !important;
    }}
    .background-image-rating {{
        position: fixed; top: 0; left: 0;
        height: 100%; width: 100%;
        background-image: url("data:image/png;base64,{fondo_base64}");
        background-size: cover;
        background-position: center;
        opacity: 0.99;
        z-index: 0;
    }}
    </style>
    <div class="background-image-rating"></div>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# (OPCIONAL) LOGIN
# ======================================================
if "logueado" in st.session_state and not st.session_state.logueado:
    st.error("Debes iniciar sesión en la página principal (home.py) para acceder a este panel.")
    st.stop()

@st.cache_data
def cargar_metricas_posicion(json_path="data/metrica_posicion.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        metricas = json.load(f)
    return metricas

# Se obtienen los datos de todos los jugadores
df = st.session_state.data

# ======================================================
# MAPA: POSICION -> METRICAS  (TU MAPEO COMPLETO)
# ======================================================
METRICS_BY_POS = cargar_metricas_posicion()

def resolve_metrics_for_pos(pos_code: str) -> list[str]:
    pos_code = (pos_code or "").strip().upper()
    val = METRICS_BY_POS.get(pos_code)
    if val == "SAME_AS_LD":
        return METRICS_BY_POS["LD"]
    if val == "SAME_AS_MC":
        return METRICS_BY_POS["MC"]
    if val == "SAME_AS_ED":
        return METRICS_BY_POS["ED"]
    if val == "SAME_AS_DC":
        return METRICS_BY_POS["DC"]
    if isinstance(val, list):
        return val
    return []


# ======================================================
# SimilarPlayers (SIMILITUD REAL: coseno real)
# ======================================================
class SimilarPlayers:
    def __init__(
        self,
        data: pd.DataFrame,
        player: str,
        N: int,
        metrics: list[str],
        id_player: str,
        extra_info: list[str],
        similarity_extra: list[str] | None = None,
        minutes_to_pct: bool = False,
        minutes_col: str = "Minutos jugados",
        minutes_pct_den: float = 4264.0,
        clip_negative_to_zero: bool = True,   # ✅ para que no existan % negativos
    ):
        self.data = data
        self.player = player
        self.N = N
        self.metrics = metrics
        self.id_player = id_player
        self.extra_info = extra_info or []
        self.similarity_extra = similarity_extra or []
        self.minutes_to_pct = minutes_to_pct
        self.minutes_col = minutes_col
        self.minutes_pct_den = minutes_pct_den
        self.clip_negative_to_zero = clip_negative_to_zero
        self._player_unique = None

    @staticmethod
    def _make_unique(series: pd.Series) -> pd.Series:
        s = series.astype(str).fillna("")
        counts = s.groupby(s).cumcount()
        return pd.Series(
            np.where(counts == 0, s, s + " (" + (counts + 1).astype(str) + ")"),
            index=series.index,
        )

    @staticmethod
    def _move_row_below(df: pd.DataFrame, row_to_move: str, anchor_row: str) -> pd.DataFrame:
        if row_to_move not in df.index or anchor_row not in df.index:
            return df
        idx = list(df.index)
        idx.remove(row_to_move)
        anchor_pos = idx.index(anchor_row)
        idx.insert(anchor_pos + 1, row_to_move)
        return df.loc[idx, :]

    @staticmethod
    def _rgba_to_hex(rgba):
        return mcolors.to_hex(rgba, keep_alpha=False)

    def algorithm(self):
         # --- Columnas usadas para la similitud (sin duplicados, manteniendo orden)
        X_cols = list(dict.fromkeys(self.similarity_extra + self.metrics))
        # --- Todas las columnas necesarias para el cálculo     
        cols_needed = list(dict.fromkeys([self.id_player] + self.extra_info + X_cols))
        # --- Validación: comprobar que el DataFrame contiene todas las columnas requeridas
        missing = [c for c in cols_needed if c not in self.data.columns]
        if missing:
            raise KeyError(f"Faltan columnas en el df: {missing}")
        # --- Filtrado del DataFrame original
        data_filter = self.data[cols_needed].copy()
        # Guardamos los nombres originales para búsquedas y mensajes de error
        original_names = data_filter[self.id_player].astype(str).tolist()
        # Aseguramos identificadores únicos (evita conflictos al indexar)
        data_filter[self.id_player] = self._make_unique(data_filter[self.id_player])
         # --- Conversión de métricas a numérico
         # Valores no convertibles -> NaN -> 0.0
        for c in X_cols:
            data_filter[c] = pd.to_numeric(data_filter[c], errors="coerce").fillna(0.0)
        # --- Normalización opcional de minutos a porcentaje
        if self.minutes_to_pct and self.minutes_col in data_filter.columns:
            mins = pd.to_numeric(data_filter[self.minutes_col], errors="coerce").fillna(0.0)
            data_filter[self.minutes_col] = (mins / float(self.minutes_pct_den)) * 100.0
         # --- Matriz de características para el cálculo de similitud
        X = data_filter[X_cols].to_numpy(dtype=float)
        # Estandarización: media 0, desviación estándar 1
        X_std = StandardScaler().fit_transform(X)
        # --- Verificación de existencia del jugador objetivo
        if self.player not in original_names:
            candidates = [n for n in original_names if self.player.lower() in str(n).lower()][:20]
            raise ValueError(
                f"El jugador '{self.player}' no está en '{self.id_player}'. Sugerencias: {candidates}"
            )
        # --- Índice del jugador objetivo
        idx = original_names.index(self.player)
        # Identificador único del jugador (post _make_unique)
        self._player_unique = data_filter.iloc[idx][self.id_player]
        # --- Vector del jugador objetivo
        v = X_std[idx : idx + 1, :]
        # --- Similitud coseno frente a todos los jugadores
        sims = cosine_similarity(v, X_std).ravel()   # ✅ COSENO REAL
        if self.clip_negative_to_zero:
            sims = np.maximum(sims, 0.0)
        # --- Resultado final como DataFrame
        df_cosine = pd.DataFrame({self._player_unique: sims}, index=data_filter[self.id_player])
        return df_cosine, data_filter, idx

    def similar(self, df_cosine, df_filter, idx_ref):
        n_similar = df_cosine[self._player_unique].sort_values(ascending=False)

        n_similar_df = (
            n_similar.reset_index()
            .rename(columns={"index": self.id_player})
            .merge(df_filter, on=self.id_player, how="left")
        )

        # ✅ % Similarity REAL (sin min-max): coseno*100
        n_similar_df[self._player_unique] = n_similar_df[self._player_unique].apply(
            lambda x: round(100 * float(x), 2)
        )
        n_similar_df.rename(columns={self._player_unique: "% Similarity"}, inplace=True)

        # quitar referencia de la lista top
        n_similar_df = n_similar_df[n_similar_df[self.id_player] != self._player_unique].reset_index(drop=True)

        out = n_similar_df.reset_index(drop=True)
        top = out.iloc[: self.N, :].copy()

        preferred_cols = [
            "Jugador", "Equipo", "Edad", "Valor_Mercado", "Nombre_Liga",
            "Pos", "Altura", "Temporada", "Minutos jugados", "% Similarity"
        ]
        first = [c for c in preferred_cols if c in top.columns]
        rest = [c for c in top.columns if c not in first]
        top = top[first + rest]

        topT = top.T
        topT.columns = list(topT.iloc[0, :])
        df_final = topT.iloc[1:, :]

        # fila del jugador referencia
        player_row = pd.Series(index=df_final.index, dtype=float)
        for r in df_final.index:
            if r == "% Similarity":
                player_row[r] = 100.0
            elif r in df_filter.columns:
                val = df_filter.loc[df_filter[self.id_player] == self._player_unique, r]
                player_row[r] = val.iloc[0] if len(val) else np.nan
            else:
                player_row[r] = np.nan

        player_row.name = self._player_unique
        df_final = pd.concat([player_row.to_frame(), df_final], axis=1)

        df_final = self._move_row_below(df_final, "Altura", "% Similarity")

        # ✅ Índice -> columna REAL "Métrica"
        df_final_display = df_final.copy()
        df_final_display.insert(0, "Métrica", df_final_display.index)
        df_final_display = df_final_display.reset_index(drop=True)

        return df_final_display, out

    def _styles_similarity_blues(self, df: pd.DataFrame, metric_rows: list[str]) -> pd.DataFrame:
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        if df.shape[1] == 0:
            return styles

        ref_col = df.columns[0]
        green_fixed = self._rgba_to_hex(cm.get_cmap("Greens")(0.65))
        blues = cm.get_cmap("Blues")

        for r in metric_rows:
            if r in styles.index:
                styles.loc[r, ref_col] = f"background-color: {green_fixed}; color: white;"

        if df.shape[1] == 1:
            return styles

        for r in metric_rows:
            if r not in df.index:
                continue
            row = df.loc[r, :]
            ref = pd.to_numeric(row[ref_col], errors="coerce")
            if pd.isna(ref):
                continue

            others = pd.to_numeric(row[df.columns[1:]], errors="coerce")
            diffs = (others - ref).abs()

            max_diff = diffs.max(skipna=True)
            if pd.isna(max_diff) or float(max_diff) == 0.0:
                for c in df.columns[1:]:
                    styles.loc[r, c] = f"background-color: {self._rgba_to_hex(blues(0.85))}; color: white;"
                continue

            closeness = 1.0 - (diffs / float(max_diff))
            min_intensity = 0.20
            intensity = min_intensity + (1.0 - min_intensity) * closeness

            for c in df.columns[1:]:
                val = intensity.get(c, np.nan)
                if pd.isna(val):
                    continue
                styles.loc[r, c] = f"background-color: {self._rgba_to_hex(blues(float(val)))}; color: white;"

        return styles

    def output_dataframe(self, df_final: pd.DataFrame):
        # df_final tiene columna "Métrica"
        basic_rows = set(self.extra_info + [self.id_player, "% Similarity"])
        metric_mask = ~df_final["Métrica"].isin(basic_rows)

        sty = df_final.style
        num_cols = [c for c in df_final.columns if c != "Métrica"]

        # formato para métricas
        sty = sty.format("{:.2f}", subset=pd.IndexSlice[metric_mask, num_cols])

        # formato para similarity
        sim_mask = df_final["Métrica"] == "% Similarity"
        if sim_mask.any():
            sty = sty.format("{:.1f}", subset=pd.IndexSlice[sim_mask, num_cols])

        metric_labels = df_final.loc[metric_mask, "Métrica"].tolist()

        def _apply_styles(_):
            temp = df_final.set_index("Métrica")
            s = self._styles_similarity_blues(temp, metric_labels)

            out_styles = pd.DataFrame("", index=df_final.index, columns=df_final.columns)
            metric_to_rowpos = {m: i for i, m in enumerate(df_final["Métrica"].tolist())}

            for m in s.index:
                i = metric_to_rowpos.get(m)
                if i is None:
                    continue
                for c in s.columns:
                    if c in out_styles.columns:
                        out_styles.loc[i, c] = s.loc[m, c]
            return out_styles

        sty = sty.apply(_apply_styles, axis=None)

        sty = sty.set_properties(
            **{
                "font-family": "Calibri",
                "text-align": "center",
                "color": "white",
                "border-width": "thin",
                "border-color": "rgba(255,255,255,0.35)",
                "border-style": "solid",
                "border-collapse": "collapse",
            }
        ).set_table_styles(
            [{"selector": "th", "props": [("text-align", "center"), ("color", "white")]}]
        )
        return sty

    def run(self):
        df_cosine, df_filter, idx_ref = self.algorithm()
        df_final, df_all = self.similar(df_cosine, df_filter, idx_ref)
        return self.output_dataframe(df_final), df_all, df_final.columns.tolist()


# ======================================================
# UI
# ======================================================
st.title("Similar Players (Jugador+Temporada ➜ filtras Competición/Minutos/Valor)")

TEMP_COL = "Temporada"
MIN_COL = "Minutos jugados"
VAL_COL = "Valor_Mercado"
LIGA_COL = "Nombre_Liga"
HEIGHT_COL = "Altura"

required_cols = ["Jugador", "Pos", TEMP_COL, MIN_COL, VAL_COL, HEIGHT_COL, LIGA_COL]
missing_req = [c for c in required_cols if c not in df.columns]
if missing_req:
    st.error(f"Faltan columnas básicas en el dataset: {missing_req}")
    st.stop()

col1, col2 = st.columns([1, 2], vertical_alignment="bottom")
temporadas = sorted(df[TEMP_COL].astype(str).dropna().unique().tolist())

with col1:
    temporada_ref = st.selectbox("Temporada del jugador referencia", temporadas)

df_ref = df[df[TEMP_COL].astype(str) == str(temporada_ref)].copy()
jugadores_ref = sorted(df_ref["Jugador"].astype(str).dropna().unique().tolist())

with col2:
    player_sel = st.selectbox("Jugador de referencia", jugadores_ref)

row_ref = df_ref[df_ref["Jugador"].astype(str) == str(player_sel)]
if row_ref.empty:
    st.error("No encuentro ese jugador en esa temporada.")
    st.stop()
row_ref = row_ref.iloc[0]

pos_player = str(row_ref["Pos"]).strip().upper()
competicion_ref = str(row_ref[LIGA_COL])

minutos_ref = int(pd.to_numeric(pd.Series([row_ref[MIN_COL]]), errors="coerce").fillna(0).iloc[0])
valor_ref = int(pd.to_numeric(pd.Series([row_ref[VAL_COL]]), errors="coerce").fillna(0).iloc[0])

with st.sidebar:
    st.header("Filtros para buscar similares")

    comps = sorted(df_ref[LIGA_COL].astype(str).dropna().unique().tolist())
    comp_sel = st.multiselect(
        "Competición",
        comps,
        default=[competicion_ref] if competicion_ref in comps else (comps[:1] if comps else []),
    )

    mins_all = pd.to_numeric(df_ref[MIN_COL], errors="coerce").fillna(0)
    vals_all = pd.to_numeric(df_ref[VAL_COL], errors="coerce").fillna(0)
    mins_max = int(mins_all.max()) if len(mins_all) else 0
    vals_max = int(vals_all.max()) if len(vals_all) else 0

    min_minutes, max_minutes = st.slider(
        "Minutos (mín - máx)",
        min_value=0,
        max_value=max(1, mins_max),
        value=(max(0, int(minutos_ref * 0.5)), min(max(1, mins_max), int(minutos_ref * 1.5))),
        step=50,
    )

    step_val = max(1, int(vals_max / 100)) if vals_max > 0 else 1
    min_val, max_val = st.slider(
        "Valor_Mercado (mín - máx)",
        min_value=0,
        max_value=max(1, vals_max),
        value=(max(0, int(valor_ref * 0.5)), min(max(1, vals_max), int(valor_ref * 1.5))),
        step=step_val,
    )

    N = st.slider("Número de similares", 3, 20, 5, 1)

    extra_info_opts = ["Equipo", LIGA_COL, "Edad", VAL_COL, "Pos", HEIGHT_COL, MIN_COL, TEMP_COL]
    default_extra = [c for c in ["Equipo", LIGA_COL, "Edad", VAL_COL, "Pos", TEMP_COL, MIN_COL] if c in df.columns]
    extra_info = st.multiselect(
        "Info extra en tabla",
        [c for c in extra_info_opts if c in df.columns],
        default=default_extra,
    )

    minutes_as_pct = st.checkbox("Mostrar Minutos como % (sobre 4264)", value=False)
    clip_neg = st.checkbox("Recortar similitud negativa a 0", value=True)

df_compare = df_ref.copy()

if comp_sel:
    df_compare = df_compare[df_compare[LIGA_COL].astype(str).isin([str(x) for x in comp_sel])].copy()

mins = pd.to_numeric(df_compare[MIN_COL], errors="coerce").fillna(0)
vals = pd.to_numeric(df_compare[VAL_COL], errors="coerce").fillna(0)

df_compare = df_compare[
    (mins >= min_minutes) & (mins <= max_minutes) &
    (vals >= min_val) & (vals <= max_val)
].copy()

# meter referencia si se quedó fuera
df_player_ref = df_ref[df_ref["Jugador"].astype(str) == str(player_sel)].copy()
if not (df_compare["Jugador"].astype(str) == str(player_sel)).any():
    df_compare = pd.concat([df_player_ref, df_compare], ignore_index=True)

metrics_for_pos = resolve_metrics_for_pos(pos_player)
if not metrics_for_pos:
    st.warning(f"No tengo métricas mapeadas para la posición '{pos_player}'.")
    st.stop()

metrics_ok = [m for m in metrics_for_pos if m in df_compare.columns]
if len(metrics_ok) < 3:
    st.error("Hay menos de 3 métricas disponibles para esta posición con esos filtros.")
    st.stop()

st.info(
    f"Referencia: **{player_sel}** · Temporada: **{temporada_ref}** · Posición: **{pos_player}**\n\n"
    f"Competición: **{', '.join(comp_sel) if comp_sel else '—'}** · "
    f"Minutos: **{min_minutes}-{max_minutes}** · Valor: **{min_val}-{max_val}** · "
    f"Métricas: **{len(metrics_ok)}** · Altura incluida ✅\n\n"
    f"➡️ **% Similarity = COSENO REAL * 100 (sin min-max).**"
)

try:
    sp = SimilarPlayers(
        data=df_compare,
        player=player_sel,
        N=N,
        metrics=metrics_ok,
        id_player="Jugador",
        extra_info=extra_info,
        similarity_extra=[HEIGHT_COL],
        minutes_to_pct=minutes_as_pct,
        minutes_col=MIN_COL,
        minutes_pct_den=4264.0,
        clip_negative_to_zero=clip_neg,
    )

    tabla_style, df_all, cols = sp.run()

    st.subheader("Tabla de similitud (interactiva)")
    st.dataframe(tabla_style, use_container_width=True, height=720, hide_index=True)

except Exception as e:
    st.error(f"Error: {e}")