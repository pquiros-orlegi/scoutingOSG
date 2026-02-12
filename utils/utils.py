import streamlit as st
import pandas as pd
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4, landscape

# MATCH ROBUSTO DE POSICIONES
# =========================
def match_posicion(valor, codigos_validos):
    """
    Devuelve True si algún token de la cadena coincide o empieza por
    alguno de los códigos (POR, GK, etc.).
    Ejemplos válidos: "POR", "POR1", "GK", "GK2", "POR / DFC".
    """
    if pd.isna(valor):
        return False
    text = str(valor).upper()
    for sep in ["/", "-", ",", "|", ";"]:
        text = text.replace(sep, " ")
    tokens = [t.strip() for t in text.split() if t.strip()]
    for tok in tokens:
        for cod in codigos_validos:
            if tok == cod or tok.startswith(cod):
                return True
    return False


# =========================
# HELPER: CONVERTIR SCORES A PERCENTILES POR RANK (0-100, PASOS DE 5)
#  👉 CREA COLUMNAS NUEVAS: "Percentil {Score ...}"
#  👉 NO USA MINUTOS, SOLO RANK DENTRO DEL SUBSET QUE LE PASES
# =========================
def aplicar_percentiles(df: pd.DataFrame, columnas, step: int = 5) -> pd.DataFrame:
    """
    Crea columnas nuevas de percentil para cada columna en `columnas`.

    - Percentil basado en rank (el mejor valor del subset siempre tiene 100).
    - Se discretiza en saltos de `step` (por defecto 5): 0, 5, 10, ..., 100.
    - No usa minutos ni filtros adicionales; eso se hace fuera.
    """
    df = df.copy()

    for col in columnas:
        if col not in df.columns:
            continue

        serie = pd.to_numeric(df[col], errors="coerce")
        if serie.dropna().empty:
            continue

        new_col = f"Percentil {col}"

        # rank percentil 0-100 dentro del subset.
        # Usamos floor (sin round) para evitar falsos 100 en valores < 100
        # por efecto de redondeo (ej. 99.6 -> 100).
        pct = serie.rank(pct=True) * 100

        # discretización en saltos de `step`
        bucket = (pct // step) * step
        df[new_col] = bucket.clip(0, 100).astype("Int64")

    return df
# =========================
# HELPER: CONSTRUIR POOL DE PERCENTILES (POR ROL)
# =========================
def construir_pool_percentiles(df, temporada_sel, categoria_sel, liga_sel):
    """
    Devuelve df_pool: jugadores de esa temporada / liga / categoría,
    con COLUMNAS_SCORE convertidos a percentiles (0-100, saltos de 5),
    pero calculados **por rol/posición**.

    Ejemplo:
      - Score GK Portero → percentil solo entre porteros.
      - Score 9 / Score Segundo Delantero → solo entre delanteros (DC/SDI/SDD).
    """
    df_scope = df.copy()

    # Se aplican los filtros de busqueda
    if temporada_sel:
        df_scope = df_scope[df_scope["Temporada"] == temporada_sel]

    if categoria_sel:
        df_scope = df_scope[df_scope["Categoría_Liga"].isin(categoria_sel)]

    if liga_sel:
        df_scope = df_scope[df_scope["Nombre_Liga"].isin(liga_sel)]

    if df_scope.empty:
        return pd.DataFrame()

    # Base sin percentiles todavía
    df_pool = df_scope.copy()

    # Helper: aplicar percentiles a un subconjunto (máscara) y
    # copiar solo las columnas "Percentil ..." a df_pool
    def aplicar_en_subset(mask, score_cols):
        if not mask.any():
            return
        sub = df_scope[mask].copy()
        sub_pct = aplicar_percentiles(sub, score_cols, step=5)
        pct_cols = [c for c in sub_pct.columns if c.startswith("Percentil ")]
        if not pct_cols:
            return
        df_pool.loc[mask, pct_cols] = sub_pct[pct_cols]


    # ---- PORTEROS ----
    mask_gk = df_scope["Pos"].apply(
        lambda v: match_posicion(v, {"POR", "GK", "PORTERO", "GOALKEEPER"})
    )
    aplicar_en_subset(mask_gk, st.session_state.columnas["SCORES_GK"])

    # ---- LATERALES (LI / LD / CAI / CAD) ----
    mask_lat = df_scope["Pos"].apply(
        lambda v: match_posicion(v, {"LI", "LD", "CAI", "CAD"})
    )
    aplicar_en_subset(mask_lat, st.session_state.columnas["SCORES_LATERAL"])

    # ---- CENTRALES (DFC) ----
    mask_dfc = df_scope["Pos"].apply(
        lambda v: match_posicion(v, {"DFC"})
    )
    aplicar_en_subset(mask_dfc, st.session_state.columnas["SCORES_CENTRAL"])

    # ---- MC (MCD / MC / MCO) ----
    mask_mc = df_scope["Pos"].apply(
        lambda v: match_posicion(v, {"MCD", "MC", "MCO"})
    )
    aplicar_en_subset(mask_mc, st.session_state.columnas["SCORES_MC"])

    # ---- EXTREMOS (EI / MI / ED / MD) ----
    mask_ext = df_scope["Pos"].apply(
        lambda v: match_posicion(v, {"EI", "MI", "ED", "MD"})
    )
    aplicar_en_subset(mask_ext, st.session_state.columnas["SCORES_EXTREMO"])

    # ---- DELANTEROS (DC / SDI / SDD) ----
    mask_del = df_scope["Pos"].apply(
        lambda v: match_posicion(v, {"DC", "SDI", "SDD"})
    )
    aplicar_en_subset(mask_del, st.session_state.columnas["SCORES_DELANTERO"])

    return df_pool


def inicializar_pool_por_defecto(df):

    # --- Temporada por defecto ---
    temporadas = sorted(df["Temporada"].dropna().unique())

    if "2025" in [str(t) for t in temporadas]:
        temporada_default = 2025
    else:
        temporada_default = temporadas[0]

    df_temp = df[df["Temporada"] == temporada_default]

    # --- Liga por defecto ---
    if "Nombre_Liga" in df_temp.columns and df_temp["Nombre_Liga"].notna().any():
        ligas = sorted(df_temp["Nombre_Liga"].dropna().unique())
        if "LA LIGA" in ligas:
            liga_default = ["LA LIGA"]
        else:
            liga_default = [ligas[0]] if len(ligas) > 0 else []
    else:
        liga_default = []

    # --- Categoría por defecto ---
    categoria_default = []
    # --- Construcción del pool ---
    df_pool = construir_pool_percentiles(
        df,
        temporada_default,
        categoria_default,
        liga_default
    )

    pool_info = {
        "temporada": temporada_default,
        "categoria": categoria_default,
        "liga": liga_default,
        "n_jugadores": len(df_pool),
    }

    return df_pool, pool_info
