import os
import base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Impacto de Redes Sociales | UNAM FI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. CSS MODERNO, ADAPTATIVO Y RESPONSIVE (PC + CELULAR)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Banner Hero Institucional (Azul UNAM + acento Dorado) */
    .hero-banner {
        background: linear-gradient(135deg, #0B1F3A 0%, #1B365D 60%, #244775 100%);
        border-radius: 18px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.4rem;
        border: 1px solid rgba(212, 175, 55, 0.35);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        flex-wrap: wrap;
    }
    .hero-left {
        display: flex;
        align-items: center;
        gap: 1.3rem;
        flex: 1;
        min-width: 280px;
    }
    .hero-logo {
        width: 72px;
        height: 72px;
        object-fit: contain;
        background: rgba(255, 255, 255, 0.95);
        padding: 6px;
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        flex-shrink: 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(212, 175, 55, 0.2);
        color: #F4D06F;
        border: 1px solid rgba(212, 175, 55, 0.45);
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }
    .hero-title {
        color: #FFFFFF !important;
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
        word-break: normal;
        overflow-wrap: break-word;
    }
    .hero-subtitle {
        color: rgba(255, 255, 255, 0.82) !important;
        font-size: 0.95rem;
        margin: 0.35rem 0 0 0;
    }
    .hero-meta {
        text-align: right;
        color: rgba(255, 255, 255, 0.85);
        font-size: 0.82rem;
        line-height: 1.5;
        background: rgba(255, 255, 255, 0.07);
        padding: 0.75rem 1.1rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }

    /* Tarjetas KPI adaptativas a Dark y Light */
    div[data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-top: 3px solid #D4AF37;
        padding: 1rem 1.2rem;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.14);
    }

    /* Pestañas estilo píldora moderna */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--secondary-background-color);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1.2rem;
        overflow-x: auto;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 0.9rem;
        border: none !important;
        white-space: nowrap;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1B365D !important;
        color: #FFFFFF !important;
        box-shadow: 0 3px 10px rgba(27, 54, 93, 0.35);
    }

    /* Botones principales */
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.55rem 1.4rem !important;
        transition: all 0.2s ease !important;
    }
    .stFormSubmitButton > button[kind="primary"] {
        background: linear-gradient(90deg, #1B365D, #2F5597) !important;
        border: 1px solid rgba(212, 175, 55, 0.5) !important;
        color: #FFFFFF !important;
    }
    .stFormSubmitButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(47, 85, 151, 0.4) !important;
    }

    /* Contenedores con borde redondeado */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        border-color: rgba(128, 128, 128, 0.22) !important;
    }

    /* ==========================================
       AJUSTES EXCLUSIVOS PARA CELULAR (<= 768px)
       No modifica nada en computadora
       ========================================== */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.8rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
        .hero-banner {
            padding: 1.25rem 1rem !important;
            gap: 1rem !important;
            text-align: center;
        }
        .hero-left {
            flex-direction: column !important;
            align-items: center !important;
            min-width: 100% !important;
            gap: 0.75rem !important;
        }
        .hero-logo {
            width: 60px !important;
            height: 60px !important;
        }
        .hero-title {
            font-size: 1.32rem !important;
            line-height: 1.25 !important;
        }
        .hero-subtitle {
            font-size: 0.84rem !important;
        }
        .hero-meta {
            width: 100% !important;
            text-align: center !important;
            font-size: 0.78rem !important;
            padding: 0.6rem 0.8rem !important;
        }
        div[data-testid="metric-container"] {
            padding: 0.75rem 0.9rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px !important;
            font-size: 0.82rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. UTILIDADES: RUTAS AUTOMÁTICAS Y DATOS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_logo_html():
    for fname in ["escudo_unam.png", "escudo_unam.jpg", "escudo_fi.png", "logo_unam.png"]:
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = "png" if fname.endswith(".png") else "jpeg"
            return f'<img src="data:image/{ext};base64,{b64}" class="hero-logo" alt="Escudo UNAM">'
    return (
        '<div class="hero-logo" style="display:flex;align-items:center;justify-content:center;'
        'background:linear-gradient(135deg,#D4AF37,#F4D06F);color:#0B1F3A;font-weight:800;font-size:1.35rem;">'
        'UNAM</div>'
    )

@st.cache_data
def cargar_datos():
    csv_path = os.path.join(BASE_DIR, "Social_media_impact_on_life.csv")
    df = pd.read_csv(csv_path)
    df_clean = df.copy()
    df_clean["Perceived_Stress_Score"] = df_clean["Perceived_Stress_Score"].fillna(df_clean["Perceived_Stress_Score"].median())
    df_clean["Academic_Performance_GPA"] = df_clean["Academic_Performance_GPA"].fillna(df_clean["Academic_Performance_GPA"].median())
    df_clean["Uso_Nocturno"] = df_clean["Late_Night_Usage"].map({True: "Sí", False: "No"})
    return df, df_clean

@st.cache_resource
def entrenar_modelo(df):
    X = df.drop(columns=["Student_ID", "Overall_Impact"])
    y = df["Overall_Impact"]
    num_features = [
        "Age", "Daily_Usage_Hours", "Weekend_Extra_Hours", "Sleep_Duration_Hours",
        "Sleep_Quality_Score", "Perceived_Stress_Score", "Mental_Health_Index", "Academic_Performance_GPA"
    ]
    cat_features = [
        "Gender", "Academic_Level", "Primary_Platform", "Device_Type",
        "Late_Night_Usage", "Social_Comparison_Frequency"
    ]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ])
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    ohe_cols = list(pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_features))
    importances = pd.Series(pipe.named_steps["clf"].feature_importances_, index=num_features + ohe_cols).sort_values(ascending=False)
    return pipe, acc, importances

def estilo_plotly(fig):
    """Gráficas limpias y adaptativas (Dark/Light y sin solapamientos en celular)."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif"),
        margin=dict(l=10, r=10, t=20, b=50),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5, title_text="")
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig

PLOTLY_CFG = {"displayModeBar": False}

df_raw, df = cargar_datos()
rf_model, rf_acc, feat_imp = entrenar_modelo(df_raw)

# ==========================================
# 4. ENCABEZADO HERO INSTITUCIONAL
# ==========================================
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-left">
        {get_logo_html()}
        <div>
            <span class="hero-badge">Facultad de Ingeniería • Minería de Datos</span>
            <h1 class="hero-title">Impacto de las Redes Sociales en la Vida Estudiantil</h1>
            <p class="hero-subtitle">Análisis exploratorio, reglas de asociación, higiene del sueño y modelado predictivo (n = 4,500)</p>
        </div>
    </div>
    <div class="hero-meta">
        <strong>Equipo 07</strong> • Semestre 2027-1<br>
        Cano Nieto Carlos Arturo<br>
        Cortés Bolaños Angel David
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 5. BARRA DE FILTROS HORIZONTAL
# ==========================================
with st.container(border=True):
    f_col1, f_col2, f_col3, f_col4 = st.columns([2.2, 1.5, 1.1, 1.6])
    with f_col1:
        plataformas = st.multiselect(
            "📱 Plataforma Principal",
            options=sorted(df["Primary_Platform"].unique()),
            default=sorted(df["Primary_Platform"].unique()),
            placeholder="Selecciona plataformas..."
        )
    with f_col2:
        niveles = st.multiselect(
            "🎓 Nivel Académico",
            options=sorted(df["Academic_Level"].unique()),
            default=sorted(df["Academic_Level"].unique()),
            placeholder="Selecciona niveles..."
        )
    with f_col3:
        uso_nocturno = st.segmented_control(
            "🌙 Uso Nocturno",
            options=["Todos", "Sí", "No"],
            default="Todos"
        )
        if uso_nocturno is None:
            uso_nocturno = "Todos"
    with f_col4:
        rango_uso = st.slider(
            "⏱️ Horas de Uso Diario",
            min_value=float(df["Daily_Usage_Hours"].min()),
            max_value=float(df["Daily_Usage_Hours"].max()),
            value=(float(df["Daily_Usage_Hours"].min()), float(df["Daily_Usage_Hours"].max())),
            step=0.5
        )

df_filt = df[
    (df["Primary_Platform"].isin(plataformas)) &
    (df["Academic_Level"].isin(niveles)) &
    (df["Daily_Usage_Hours"].between(rango_uso[0], rango_uso[1]))
]
if uso_nocturno != "Todos":
    df_filt = df_filt[df_filt["Uso_Nocturno"] == uso_nocturno]

if df_filt.empty:
    st.warning("No hay estudiantes que coincidan con los filtros seleccionados. Ajusta los controles superiores.")
    st.stop()

# ==========================================
# 6. TARJETAS KPI PRINCIPALES
# ==========================================
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Estudiantes", f"{len(df_filt):,}", f"{len(df_filt)/len(df)*100:.1f}% de la muestra")
k2.metric("Uso Diario", f"{df_filt['Daily_Usage_Hours'].mean():.2f} h", f"Mediana: {df_filt['Daily_Usage_Hours'].median():.1f} h")
k3.metric("Sueño Promedio", f"{df_filt['Sleep_Duration_Hours'].mean():.2f} h", "Umbral crítico: 6.25 h")
k4.metric("Estrés (PSS)", f"{df_filt['Perceived_Stress_Score'].mean():.1f} / 40", f"Salud Mental: {df_filt['Mental_Health_Index'].mean():.0f}/100")
k5.metric("GPA Promedio", f"{df_filt['Academic_Performance_GPA'].mean():.2f} / 4.0", f"Negativo: {(df_filt['Overall_Impact']=='Negative').mean()*100:.1f}%")

st.write("")

# ==========================================
# 7. PESTAÑAS INTERACTIVAS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Panorama y EDA",
    "🔗 Correlaciones, Apriori y Clusters",
    "🤖 Modelos y Caja Blanca",
    "🎯 Simulador Predictivo en Vivo"
])

COLOR_MAP = {"Beneficial": "#2ECC71", "Neutral": "#F39C12", "Negative": "#E74C3C"}

# --- TAB 1: EDA ---
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### 🌙 Horas de Sueño vs. Rendimiento Académico (GPA)")
            fig_scatter = px.scatter(
                df_filt.sample(min(1200, len(df_filt)), random_state=42),
                x="Sleep_Duration_Hours",
                y="Academic_Performance_GPA",
                color="Overall_Impact",
                size="Daily_Usage_Hours",
                hover_data=["Primary_Platform", "Perceived_Stress_Score", "Mental_Health_Index", "Uso_Nocturno"],
                color_discrete_map=COLOR_MAP,
                labels={"Sleep_Duration_Hours": "Horas de Sueño", "Academic_Performance_GPA": "GPA", "Overall_Impact": "Impacto"},
                opacity=0.72
            )
            fig_scatter.add_vline(x=6.25, line_dash="dash", line_color="#E74C3C", annotation_text="Umbral: 6.25h")
            st.plotly_chart(estilo_plotly(fig_scatter), use_container_width=True, config=PLOTLY_CFG)

    with c2:
        with st.container(border=True):
            st.markdown("#### 📱 Paradoja de Percepción por Plataforma")
            plat_counts = df_filt.groupby(["Primary_Platform", "Overall_Impact"]).size().reset_index(name="Estudiantes")
            fig_bar = px.bar(
                plat_counts,
                x="Primary_Platform",
                y="Estudiantes",
                color="Overall_Impact",
                barmode="group",
                color_discrete_map=COLOR_MAP,
                labels={"Primary_Platform": "Plataforma", "Overall_Impact": "Impacto"}
            )
            st.plotly_chart(estilo_plotly(fig_bar), use_container_width=True, config=PLOTLY_CFG)

    c3, c4 = st.columns(2)
    with c3:
        with st.container(border=True):
            st.markdown("#### ⏱️ Distribución de Uso Diario por Impacto Global")
            fig_box = px.box(
                df_filt,
                x="Overall_Impact",
                y="Daily_Usage_Hours",
                color="Overall_Impact",
                color_discrete_map=COLOR_MAP,
                labels={"Overall_Impact": "Impacto Global", "Daily_Usage_Hours": "Horas Diarias"}
            )
            fig_box.add_hline(y=5.85, line_dash="dash", line_color="#E74C3C", annotation_text="Umbral Apriori: 5.85h")
            st.plotly_chart(estilo_plotly(fig_box), use_container_width=True, config=PLOTLY_CFG)

    with c4:
        with st.container(border=True):
            st.markdown("#### 🛌 Efecto del Uso Nocturno en Sueño, Estrés y GPA")
            noct_df = df_filt.groupby("Uso_Nocturno")[["Sleep_Duration_Hours", "Perceived_Stress_Score", "Academic_Performance_GPA"]].mean().reset_index()
            noct_melt = noct_df.melt(id_vars="Uso_Nocturno", var_name="Métrica", value_name="Promedio")
            noct_melt["Métrica"] = noct_melt["Métrica"].replace({
                "Sleep_Duration_Hours": "Horas de Sueño",
                "Perceived_Stress_Score": "Estrés (PSS)",
                "Academic_Performance_GPA": "GPA"
            })
            fig_noct = px.bar(
                noct_melt,
                x="Métrica",
                y="Promedio",
                color="Uso_Nocturno",
                barmode="group",
                text_auto=".2f",
                color_discrete_map={"Sí": "#E74C3C", "No": "#3498DB"}
            )
            st.plotly_chart(estilo_plotly(fig_noct), use_container_width=True, config=PLOTLY_CFG)

# --- TAB 2: CORRELACIONES, APRIORI Y CLUSTERING ---
with tab2:
    col_a, col_b = st.columns([1.15, 1])
    with col_a:
        with st.container(border=True):
            st.markdown("#### 🔥 Matriz de Correlación de Pearson")
            num_cols = [
                "Daily_Usage_Hours", "Sleep_Duration_Hours", "Sleep_Quality_Score",
                "Perceived_Stress_Score", "Mental_Health_Index", "Academic_Performance_GPA"
            ]
            labels_es = ["Uso Diario", "Sueño (h)", "Calidad Sueño", "Estrés (PSS)", "Salud Mental", "GPA"]
            corr = df_filt[num_cols].corr().round(2)
            corr.index = labels_es
            corr.columns = labels_es
            fig_corr = px.imshow(
                corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"
            )
            st.plotly_chart(estilo_plotly(fig_corr), use_container_width=True, config=PLOTLY_CFG)

    with col_b:
        with st.container(border=True):
            st.markdown("#### 🧩 Reglas de Asociación Críticas (Apriori)")
            st.caption("Soporte > 20% • Confianza > 45% • Consecuente: GPA Bajo")
            reglas = pd.DataFrame({
                "Factor": ["Riesgo por Pantalla", "Salud Mental", "Privación de Sueño"],
                "Umbral (Antecedente)": ["Uso > 5.85 h/día", "Índice < 76.5 pts", "Sueño < 6.25 h/noche"],
                "Confianza": ["67.4%", "63.3%", "61.1%"],
                "Lift": [2.02, 1.89, 1.83]
            })
            st.dataframe(reglas, use_container_width=True, hide_index=True)

        with st.container(border=True):
            st.markdown("#### 🎯 Segmentación K-Means en Vivo (k = 3)")
            km_data = df_filt[["Daily_Usage_Hours", "Sleep_Duration_Hours", "Perceived_Stress_Score", "Academic_Performance_GPA"]].copy()
            if len(km_data) >= 10:
                km = KMeans(n_clusters=3, random_state=42, n_init=10)
                km_data["Cluster"] = "Cluster " + (km.fit_predict(StandardScaler().fit_transform(km_data)) + 1).astype(str)
                fig_km = px.scatter(
                    km_data.sample(min(800, len(km_data)), random_state=42),
                    x="Daily_Usage_Hours", y="Sleep_Duration_Hours", color="Cluster",
                    labels={"Daily_Usage_Hours": "Uso Diario (h)", "Sleep_Duration_Hours": "Horas de Sueño"},
                    color_discrete_sequence=["#3498DB", "#E74C3C", "#F39C12"]
                )
                st.plotly_chart(estilo_plotly(fig_km), use_container_width=True, config=PLOTLY_CFG)

# --- TAB 3: MODELOS Y CAJA BLANCA ---
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### 🏆 Comparación de Esquemas de Clasificación")
            modelos_df = pd.DataFrame({
                "Modelo": ["Red Neuronal (MLP)", "Random Forest", "Árbol de Decisión", "KNN (k=5)", "Naive Bayes"],
                "Exactitud Test (%)": [98.2, 97.1, 96.7, 94.2, 92.3],
                "CV 5-Fold (%)": [97.6, 96.9, 96.3, 94.7, 91.2]
            })
            fig_mod = px.bar(
                modelos_df.melt(id_vars="Modelo", var_name="Métrica", value_name="Porcentaje"),
                x="Modelo", y="Porcentaje", color="Métrica", barmode="group",
                text_auto=".1f",
                color_discrete_sequence=["#3498DB", "#D4AF37"]
            )
            fig_mod.update_yaxes(range=[85, 101])
            st.plotly_chart(estilo_plotly(fig_mod), use_container_width=True, config=PLOTLY_CFG)

    with c2:
        with st.container(border=True):
            st.markdown("#### 🔍 Interpretabilidad: Top 10 Variables (Random Forest)")
            top10 = (feat_imp.head(10) * 100).reset_index()
            top10.columns = ["Variable", "Importancia (%)"]
            fig_imp = px.bar(
                top10[::-1], x="Importancia (%)", y="Variable", orientation="h",
                text_auto=".1f", color="Importancia (%)", color_continuous_scale="Blues"
            )
            fig_imp.update_layout(coloraxis_showscale=False)
            st.plotly_chart(estilo_plotly(fig_imp), use_container_width=True, config=PLOTLY_CFG)
            st.info("💡 **Hallazgo disruptivo:** Conocer qué red social usa el estudiante aporta **menos del 1%** al poder predictivo del modelo.")

# --- TAB 4: SIMULADOR PREDICTIVO ---
with tab4:
    with st.container(border=True):
        st.markdown("#### 🎯 Simulador de Riesgo Estudiantil en Tiempo Real (Random Forest)")
        st.caption(f"Modelo entrenado en vivo sobre los 4,500 registros (Exactitud verificada en prueba: {rf_acc*100:.2f}%). Ajusta los parámetros para simular un perfil:")

        with st.form("simulador_form"):
            f1, f2, f3 = st.columns(3)
            with f1:
                sim_age = st.slider("Edad", 15, 26, 20)
                sim_gender = st.selectbox("Género", sorted(df["Gender"].unique()))
                sim_level = st.selectbox("Nivel Académico", sorted(df["Academic_Level"].unique()))
                sim_plat = st.selectbox("Plataforma Principal", sorted(df["Primary_Platform"].unique()))
                sim_dev = st.selectbox("Dispositivo", sorted(df["Device_Type"].unique()))
            with f2:
                sim_usage = st.slider("Uso Diario (horas)", 0.5, 14.0, 6.5, 0.5)
                sim_wknd = st.slider("Horas Extra Fin de Semana", 0.0, 4.5, 2.0, 0.5)
                sim_sleep = st.slider("Horas de Sueño", 3.0, 10.5, 5.5, 0.5)
                sim_squal = st.slider("Calidad de Sueño (1-5)", 1, 5, 2)
            with f3:
                sim_night = st.toggle("¿Usa redes en la madrugada?", value=True)
                sim_comp = st.selectbox("Comparación Social", ["Never", "Rarely", "Sometimes", "Frequently", "Always"], index=3)
                sim_pss = st.slider("Estrés Percibido (PSS 0-40)", 0, 40, 22)
                sim_mh = st.slider("Índice Salud Mental (32-98)", 32, 98, 65)
                sim_gpa = st.slider("GPA Actual (1.9 - 4.0)", 1.9, 4.0, 3.0, 0.05)

            submitted = st.form_submit_button("⚡ Predecir Impacto Global", type="primary", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame([{
                "Age": sim_age,
                "Gender": sim_gender,
                "Academic_Level": sim_level,
                "Primary_Platform": sim_plat,
                "Daily_Usage_Hours": sim_usage,
                "Weekend_Extra_Hours": sim_wknd,
                "Device_Type": sim_dev,
                "Sleep_Duration_Hours": sim_sleep,
                "Sleep_Quality_Score": sim_squal,
                "Late_Night_Usage": sim_night,
                "Social_Comparison_Frequency": sim_comp,
                "Perceived_Stress_Score": float(sim_pss),
                "Mental_Health_Index": sim_mh,
                "Academic_Performance_GPA": float(sim_gpa)
            }])
            pred = rf_model.predict(input_df)[0]
            probs = rf_model.predict_proba(input_df)[0]
            clases = rf_model.classes_

            st.divider()
            r1, r2 = st.columns([1, 1.5])
            with r1:
                if pred == "Beneficial":
                    st.success(f"### Predicción: **{pred}**\nPerfil de **bajo riesgo** con hábitos digitales equilibrados.")
                elif pred == "Neutral":
                    st.warning(f"### Predicción: **{pred}**\nPerfil en **zona de alerta moderada**. Vigilar horas de sueño y estrés.")
                else:
                    st.error(f"### Predicción: **{pred}**\nPerfil de **alto riesgo clínico y académico**. Se recomienda intervención de higiene del sueño.")
            with r2:
                prob_df = pd.DataFrame({"Clase": clases, "Probabilidad (%)": probs * 100})
                fig_p = px.bar(prob_df, x="Clase", y="Probabilidad (%)", color="Clase", color_discrete_map=COLOR_MAP, text_auto=".1f")
                fig_p.update_yaxes(range=[0, 108])
                st.plotly_chart(estilo_plotly(fig_p), use_container_width=True, config=PLOTLY_CFG)
