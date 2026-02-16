import streamlit as st
import pandas as pd
from GameDataset import GameDataset
from GameAnalyzer import GameAnalyzer
from GameRecommender import GameRecommender

# ══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE LA PÁGINA
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BGG Game Recommender",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
# ESTILOS CSS PERSONALIZADOS
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# ESTILOS CSS PERSONALIZADOS
# ══════════════════════════════════════════════════════════════════

st.markdown("""<style>

    /* Fondo principal */
    .main {
        background-color: #0e1117;
    }

    /* Botones */
    .stButton>button {
        background-color: #00ffcc;
        color: #000000;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #ff0055;
        color: #ffffff;
        transform: scale(1.05);
    }

    /* Títulos */
    h1 {
        color: #00ffcc;
        font-weight: bold;
    }

    h2 {
        color: #ff0055;
    }

    h3 {
        color: #00ffcc;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }

    /* Métricas */
    [data-testid="stMetricValue"] {
        color: #00ffcc;
    }</style>

""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# FUNCIÓN DE CARGA DE DATOS (CON CACHE)
# ══════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    """Carga el dataset y crea las instancias necesarias."""
    # Cargar CSV
    df = pd.read_csv('games_enriched.csv')

    # Limpiar NaN en boardgame
    df = df.dropna(subset=['boardgame'])
    # Eliminar duplicados
    df = df.drop_duplicates(subset='boardgame').reset_index(drop=True)
    # Asegurar que boardgame sea string
    df['boardgame'] = df['boardgame'].astype(str)

    # Crear instancias
    dataset = GameDataset(df)
    analyzer = GameAnalyzer(dataset)
    recommender = GameRecommender(dataset)

    return dataset, analyzer, recommender, df


# ══════════════════════════════════════════════════════════════════
# CARGA INICIAL
# ══════════════════════════════════════════════════════════════════

try:
    dataset, analyzer, recommender, df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"❌ Error al cargar datos: {e}")
    st.stop()

# ══════════════════════════════════════════════════════════════════
# SIDEBAR - NAVEGACIÓN
# ══════════════════════════════════════════════════════════════════

st.sidebar.title("🎲 BGG Recommender")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navegación",
    ["🏠 Inicio",
     "🔍 Buscar Similares",
     "🎯 Búsqueda Conceptual",
     "⚖️ Comparar Juegos",
     "📊 Análisis Estadístico"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"📚 **{len(df):,}** juegos en el dataset")
st.sidebar.info(f"🎲 **190** mecánicas únicas")

# ══════════════════════════════════════════════════════════════════
# PÁGINA: INICIO
# ══════════════════════════════════════════════════════════════════

if page == "🏠 Inicio":
    st.title("🎲 Sistema de Recomendación de Juegos de Mesa")
    st.markdown("### Basado en datos de BoardGameGeek")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Juegos Analizados", f"{len(df):,}")

    with col2:
        st.metric("Mecánicas Únicas", "190")

    with col3:
        st.metric("Rating Promedio", f"{df['avg_rating'].mean():.2f}")

    st.markdown("---")

    st.markdown("""
    ## 🚀 Características del Sistema

    ### 🔍 Búsqueda de Juegos Similares
    Encuentra juegos parecidos a tus favoritos basándose en mecánicas compartidas

    ### 🎯 Búsqueda Conceptual
    Describe las mecánicas que buscas y encuentra juegos que las tengan

    ### ⚖️ Comparación de Juegos
    Compara dos juegos lado a lado para ver qué tienen en común

    ### 📊 Análisis Estadístico
    Explora correlaciones y tendencias en el diseño de juegos
    """)

    st.markdown("---")

    st.markdown("""
    ## 🧮 Tecnología

    - **MultiLabelBinarizer**: Vectorización de mecánicas
    - **Similitud de Coseno**: Cálculo de similitud entre juegos
    - **Fórmula Híbrida**: Combina similitud mecánica + rating + complejidad
    - **Filtros Avanzados**: Por jugadores, tiempo, complejidad
    """)

# ══════════════════════════════════════════════════════════════════
# PÁGINA: BUSCAR SIMILARES
# ══════════════════════════════════════════════════════════════════

elif page == "🔍 Buscar Similares":
    st.title("🔍 Buscar Juegos Similares")

    st.markdown("""
    Encuentra juegos parecidos a uno que ya conoces. El sistema analiza las mecánicas 
    compartidas y recomienda alternativas similares.
    """)

    st.markdown("---")

    # Formulario de búsqueda
    col1, col2 = st.columns([2, 1])

    with col1:
        # Autocomplete de juegos
        juego_seleccionado = st.selectbox(
            "Selecciona un juego:",
            options=sorted(df['boardgame'].tolist()),
            index=0
        )

    with col2:
        n_recomendaciones = st.slider(
            "Número de recomendaciones:",
            min_value=1,
            max_value=10,
            value=5
        )

    col3, col4, col5 = st.columns(3)

    with col3:
        tolerancia = st.slider(
            "Tolerancia de complejidad:",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.5,
            help="Qué tan diferentes pueden ser en complejidad"
        )

    with col4:
        excluir_familia = st.checkbox(
            "Excluir expansiones/variantes",
            value=True,
            help="Evita recomendar expansiones del mismo juego"
        )

    with col5:
        st.write("")  # Espaciado
        buscar = st.button("🔍 Buscar", use_container_width=True)

    if buscar or juego_seleccionado:
        st.markdown("---")

        with st.spinner("Buscando juegos similares..."):
            try:
                resultados = recommender.recommend_similar_games(
                    juego_seleccionado,
                    n=n_recomendaciones,
                    complexity_tolerance=tolerancia,
                    exclude_family=excluir_familia
                )

                if isinstance(resultados, str):
                    st.error(resultados)
                else:
                    st.success(f"✅ Encontrados {len(resultados)} juegos similares")

                    # Mostrar resultados
                    for idx, row in resultados.iterrows():
                        with st.expander(f"⭐ {row['boardgame']} (Rating: {row['avg_rating']:.2f})"):
                            col_a, col_b, col_c = st.columns(3)

                            with col_a:
                                st.metric("Similitud", f"{row['similarity']:.2%}")

                            with col_b:
                                st.metric("Rating", f"{row['avg_rating']:.2f}")

                            with col_c:
                                st.metric("Complejidad", f"{row['complexity']:.2f}")

                            st.markdown(f"**Mecánicas:** {row['mechanics'][:200]}...")

            except Exception as e:
                st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════
# PÁGINA: BÚSQUEDA CONCEPTUAL
# ══════════════════════════════════════════════════════════════════

elif page == "🎯 Búsqueda Conceptual":
    st.title("🎯 Búsqueda Conceptual")

    st.markdown("""
    Describe el tipo de juego que buscas seleccionando mecánicas, número de jugadores 
    y tiempo de juego. El sistema encontrará los mejores juegos que coincidan.
    """)

    st.markdown("---")

    # Obtener mecánicas únicas
    todas_mecanicas = sorted(recommender.mlb.classes_)

    # Mecánicas más comunes (sugerencias)
    mecanicas_comunes = [
        "Hand Management", "Dice Rolling", "Set Collection",
        "Cooperative Game", "Worker Placement", "Variable Player Powers",
        "Deck, Bag, and Pool Building", "Solo / Solitaire Game"
    ]

    col1, col2 = st.columns([2, 1])

    with col1:
        mecanicas_seleccionadas = st.multiselect(
            "Selecciona mecánicas deseadas:",
            options=todas_mecanicas,
            default=[],
            help="Elige al menos una mecánica"
        )

    with col2:
        st.markdown("**Mecánicas populares:**")
        for mec in mecanicas_comunes[:5]:
            st.text(f"• {mec}")

    col3, col4, col5 = st.columns(3)

    with col3:
        min_jugadores = st.number_input(
            "Mínimo de jugadores:",
            min_value=1,
            max_value=10,
            value=1
        )

    with col4:
        max_tiempo = st.number_input(
            "Tiempo máximo (min):",
            min_value=10,
            max_value=300,
            value=120,
            step=10
        )

    with col5:
        n_resultados = st.slider(
            "Resultados:",
            min_value=1,
            max_value=10,
            value=5
        )

    buscar_conceptual = st.button("🎯 Buscar Juegos", use_container_width=True)

    if buscar_conceptual:
        if not mecanicas_seleccionadas:
            st.warning("⚠️ Por favor selecciona al menos una mecánica")
        else:
            st.markdown("---")

            with st.spinner("Buscando juegos que coincidan..."):
                try:
                    resultados = recommender.recommend_by_features(
                        mechanics_list=mecanicas_seleccionadas,
                        min_players=min_jugadores,
                        max_time=max_tiempo,
                        n=n_resultados
                    )

                    if resultados.empty:
                        st.warning("❌ No se encontraron juegos con esas características exactas")
                        st.info("💡 Intenta reducir el número de mecánicas o aumentar el tiempo máximo")
                    else:
                        st.success(f"✅ Encontrados {len(resultados)} juegos")

                        for idx, row in resultados.iterrows():
                            with st.expander(f"🎲 {row['boardgame']} (Match: {row['match']:.2%})"):
                                col_a, col_b, col_c = st.columns(3)

                                with col_a:
                                    st.metric("Match", f"{row['match']:.2%}")

                                with col_b:
                                    st.metric("Rating", f"{row['avg_rating']:.2f}")

                                with col_c:
                                    st.metric("Jugadores", f"{row['min_players']}-{row['max_players']}")

                                st.markdown(f"**Mecánicas completas:** {row['mechanics']}")

                except Exception as e:
                    st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════
# PÁGINA: COMPARAR JUEGOS
# ══════════════════════════════════════════════════════════════════

elif page == "⚖️ Comparar Juegos":
    st.title("⚖️ Comparar Juegos")

    st.markdown("""
    Compara dos juegos lado a lado para ver qué mecánicas comparten y cuáles son únicas de cada uno.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        juego_a = st.selectbox(
            "Primer juego:",
            options=sorted(df['boardgame'].tolist()),
            index=0
        )

    with col2:
        juego_b = st.selectbox(
            "Segundo juego:",
            options=sorted(df['boardgame'].tolist()),
            index=1
        )

    comparar = st.button("⚖️ Comparar", use_container_width=True)

    if comparar:
        st.markdown("---")

        with st.spinner("Comparando juegos..."):
            try:
                comparacion = recommender.compare_games(juego_a, juego_b)

                if isinstance(comparacion, str):
                    st.error(comparacion)
                else:
                    # Métricas principales
                    col_m1, col_m2, col_m3 = st.columns(3)

                    with col_m1:
                        st.metric(
                            "Similitud Mecánica",
                            f"{comparacion['similitud']:.2%}",
                            help="0% = completamente diferentes, 100% = idénticos"
                        )

                    with col_m2:
                        st.metric(
                            "Mecánicas Compartidas",
                            len(comparacion['compartidas'])
                        )

                    with col_m3:
                        total_unicas = len(comparacion['unicas_a']) + len(comparacion['unicas_b'])
                        st.metric(
                            "Mecánicas Únicas Total",
                            total_unicas
                        )

                    st.markdown("---")

                    # Detalles
                    col_d1, col_d2, col_d3 = st.columns(3)

                    with col_d1:
                        st.markdown("### 🔗 Compartidas")
                        if comparacion['compartidas']:
                            for mec in comparacion['compartidas']:
                                st.success(f"✓ {mec}")
                        else:
                            st.info("(Ninguna)")

                    with col_d2:
                        st.markdown(f"### 💎 Únicas de {comparacion['nombres'][0]}")
                        if comparacion['unicas_a']:
                            for mec in list(comparacion['unicas_a'])[:10]:
                                st.warning(f"• {mec}")
                            if len(comparacion['unicas_a']) > 10:
                                st.text(f"... y {len(comparacion['unicas_a']) - 10} más")
                        else:
                            st.info("(Ninguna)")

                    with col_d3:
                        st.markdown(f"### 🚀 Únicas de {comparacion['nombres'][1]}")
                        if comparacion['unicas_b']:
                            for mec in list(comparacion['unicas_b'])[:10]:
                                st.warning(f"• {mec}")
                            if len(comparacion['unicas_b']) > 10:
                                st.text(f"... y {len(comparacion['unicas_b']) - 10} más")
                        else:
                            st.info("(Ninguna)")

            except Exception as e:
                st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════
# PÁGINA: ANÁLISIS ESTADÍSTICO
# ══════════════════════════════════════════════════════════════════

elif page == "📊 Análisis Estadístico":
    st.title("📊 Análisis Estadístico")

    st.markdown("---")

    # Correlación Weight vs Rating
    st.markdown("### 📈 Correlación Complejidad vs Rating")

    correlacion = analyzer.correlation_weight_rating()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Correlación de Pearson",
            f"{correlacion['pearson']['val']:.4f}",
            help="Mide relación lineal"
        )

    with col2:
        st.metric(
            "Correlación de Spearman",
            f"{correlacion['spearman']['val']:.4f}",
            help="Mide relación monotónica"
        )

    with col3:
        st.metric(
            "Significancia",
            "p < 0.001",
            help="Altamente significativo"
        )

    st.info("""
    💡 **Interpretación:** Existe una correlación moderada-fuerte (0.54) entre la complejidad 
    y el rating de los juegos. Los juegos más complejos tienden a tener mejores calificaciones 
    en BoardGameGeek.
    """)

    st.markdown("---")

    # Estadísticas del dataset
    st.markdown("### 📊 Estadísticas Generales")

    stats = dataset.get_stats()

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        st.metric("Total Juegos", f"{stats['total_juegos']:,}")

    with col_s2:
        st.metric("Rango de Años", f"{stats['rango_años'][0]}-{stats['rango_años'][1]}")

    with col_s3:
        st.metric("Rating Promedio", f"{stats['rating_promedio']:.2f}")

    with col_s4:
        st.metric("Complejidad Media", f"{stats['complejidad_media']:.2f}")

# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""

    🎲 Sistema de Recomendación BGG | Desarrollado con Streamlit y Python
    Datos: BoardGameGeek | Tecnología: MultiLabelBinarizer + Similitud de Coseno

""", unsafe_allow_html=True)

