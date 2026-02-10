import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class GameRecommender:
    """
    Sistema de recomendación basado en Etiquetas Exactas (MultiLabelBinarizer).
    Preserva las mecánicas compuestas (ej. 'Worker Placement') como entidades únicas.
    """

    def __init__(self, internal_dataset):
        print("Inicializando Motor (Modo: MultiLabelBinarizer)...")
        self.dataset = internal_dataset
        self.df = internal_dataset.df.copy()

        self.family_stop_words = {
            'the', 'a', 'an', 'of', 'and', 'or', 'in', 'on', 'at', 'to', 'for',
            'edition', 'volume', 'vol', 'deluxe', 'legacy', 'second', 'third',
            'expansion', 'game', 'boardgame', 'pack', 'set', 'box'
        }

        # No stop_words aquí porque MLB toma la etiqueta completa exacta.
        self.mlb = MultiLabelBinarizer(sparse_output=True) # sparse=True para eficiencia de memoria
        self._build_model()
        print("Motor listo (Mecánicas compuestas preservadas).")

    def _build_model(self):
        """
        Construye la matriz preservando frases completas.
        MEJORA: Fusionamos 'mechanics' y 'categories' para un perfil completo.
        """
        # 1. Fusion texto de Mecánicas + Categorías
        # Esto permite buscar "Horror" (Categoría) y "Deck Building" (Mecánica) a la vez
        combined_tags = (
            self.df['mechanics'].fillna('') + ";" +
            self.df['categories'].fillna('')
        )

        # 2. Limpiamos y convertimos a lista
        tags_list = combined_tags.apply(
            lambda x: [m.strip() for m in x.split(';') if m.strip()]
        )

        # 3. Entrenamos el MultiLabelBinarizer con el conjunto extendido
        self.mechanics_matrix = self.mlb.fit_transform(tags_list)

        # 4. Calculamos similitud
        self.similarity_matrix = cosine_similarity(self.mechanics_matrix)

        print(f"Modelo re-entrenado con {len(self.mlb.classes_)} etiquetas únicas (Mecánicas + Categorías).")

    def _get_clean_keywords(self, text):
        text_clean = re.sub(r'[^\w\s]', '', text.lower())
        return set(text_clean.split()) - self.family_stop_words

    def _resolve_game_id(self, game_name):
        mask = self.df['boardgame'].str.contains(re.escape(game_name), case=False, na=False)
        matches = self.df[mask].copy()
        if matches.empty: return None, None
        exact_match = matches[matches['boardgame'].str.lower() == game_name.lower()]
        if not exact_match.empty: return exact_match.index[0], exact_match.iloc[0]['boardgame']
        matches['title_len'] = matches['boardgame'].str.len()
        best_match = matches.sort_values('title_len').iloc[0]
        return best_match.name, best_match['boardgame']

    def recommend_similar_games(self, game_name, n=5, complexity_tolerance=0.5, exclude_family=True):
        idx, resolved_name = self._resolve_game_id(game_name)
        if idx is None: return f"Error: '{game_name}' no encontrado."

        base_game = self.df.iloc[idx]
        base_keywords = self._get_clean_keywords(base_game['boardgame'])
        base_weight = base_game['complexity']

        print(f"Analizando (MLB): {resolved_name} (Weight: {base_weight:.2f})")

        sim_scores = self.similarity_matrix[idx]
        ratings = self.df['avg_rating'].fillna(0)
        norm_ratings = ratings / 10.0

        weight_diff_global = abs(self.df['complexity'] - base_weight)
        weight_score = 1 / (1 + weight_diff_global)

        # Fórmula final
        final_score = (sim_scores * 0.50) + (norm_ratings * 0.30) + (weight_score * 0.20)

        res = self.df.copy()
        res['similarity'] = sim_scores
        res['weight_score'] = weight_score
        res['final_score'] = final_score

        res = res[res.index != idx]
        res = res[abs(res['complexity'] - base_weight) <= complexity_tolerance]

        if exclude_family:
            res = res[~res['boardgame'].apply(
                lambda x: len(base_keywords.intersection(self._get_clean_keywords(x))) > 0
            )]

        return res.sort_values('final_score', ascending=False).head(n)

    def recommend_by_features(self, mechanics_list=None, min_players=1, max_time=120, n=5):
        mask = (self.df['min_players'] <= min_players) & \
               (self.df['max_players'] >= min_players) & \
               (self.df['max_playtime'] <= max_time)
        candidates = self.df[mask].copy()

        if mechanics_list:
            # MLB espera una lista de listas [[tag1, tag2]], así que envolvemos la query
            # NOTA: Las mecánicas deben escribirse EXACTAMENTE igual (Case Sensitive)
            try:
                query_vec = self.mlb.transform([mechanics_list])
                candidate_indices = candidates.index
                sims = cosine_similarity(self.mechanics_matrix[candidate_indices], query_vec).flatten()
                candidates['match'] = sims
                return candidates.sort_values('match', ascending=False).head(n)
            except ValueError:
                # Si una mecánica que no existe en el dataset
                print("Alguna de las mecánicas no existe en el dataset. Revisa la ortografía.")
                return pd.DataFrame() # Retorna vacío

        return candidates.sort_values('avg_rating', ascending=False).head(n)

    def compare_games(self, game_a, game_b):
        """Compara usando lógica de conjuntos."""
        idx_a, name_a = self._resolve_game_id(game_a)
        idx_b, name_b = self._resolve_game_id(game_b)
        if idx_a is None or idx_b is None: return "Juego no encontrado"

        sim = self.similarity_matrix[idx_a, idx_b]

        # Extraemos sets reales
        mechs_a = set([m.strip() for m in self.df.iloc[idx_a]['mechanics'].split(';') if m.strip()])
        mechs_b = set([m.strip() for m in self.df.iloc[idx_b]['mechanics'].split(';') if m.strip()])

        return {
            'nombres': (name_a, name_b),
            'similitud': sim,
            'compartidas': mechs_a.intersection(mechs_b),
            'unicas_a': mechs_a - mechs_b,
            'unicas_b': mechs_b - mechs_a
        }

    # Gráficos (Radar y Vector)
    def plot_radar(self, game_a, game_b):
        """Gráfico de radar (Sin cambios, solo requiere matplotlib importado)."""
        idx_a, name_a = self._resolve_game_id(game_a)
        idx_b, name_b = self._resolve_game_id(game_b)
        if idx_a is None or idx_b is None: return

        g1 = self.df.iloc[idx_a]
        g2 = self.df.iloc[idx_b]
        categories = ['Complexity', 'Rating', 'Min Players', 'Max Players']
        val1 = [g1['complexity'], g1['avg_rating']/2, g1['min_players'], min(g1['max_players'], 6)]
        val2 = [g2['complexity'], g2['avg_rating']/2, g2['min_players'], min(g2['max_players'], 6)]
        N = len(categories)
        angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
        angles += angles[:1]
        val1 += val1[:1]
        val2 += val2[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        plt.style.use('dark_background')
        ax.plot(angles, val1, linewidth=2, linestyle='solid', label=name_a, color='#00ffcc')
        ax.fill(angles, val1, '#00ffcc', alpha=0.2)
        ax.plot(angles, val2, linewidth=2, linestyle='solid', label=name_b, color='#ff0055')
        ax.fill(angles, val2, '#ff0055', alpha=0.2)
        plt.xticks(angles[:-1], categories, color='white', size=10)
        ax.set_rlabel_position(0)
        plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="gray", size=8)
        plt.ylim(0, 6)
        plt.title(f"COMPARATIVA DE PERFIL\n{name_a} vs {name_b}", color='white', y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='#222', edgecolor='white', labelcolor='white')
        plt.show()

    def plot_vector_comparison(self, game_a, game_b):
        """Visualiza coincidencias de etiquetas (MLB)."""
        import numpy as np
        idx_a, name_a = self._resolve_game_id(game_a)
        idx_b, name_b = self._resolve_game_id(game_b)
        if idx_a is None or idx_b is None: return

        # Vectores binarios (MLB devuelve matriz densa o sparse, convertimos a array plano)
        vec_a = self.mechanics_matrix[idx_a].toarray().flatten()
        vec_b = self.mechanics_matrix[idx_b].toarray().flatten()

        # Nombres de las etiquetas (frases completas)
        feature_names = self.mlb.classes_

        mask = (vec_a > 0) | (vec_b > 0)
        active_mechanics = feature_names[mask]
        weights_a = vec_a[mask]
        weights_b = vec_b[mask]

        # Ordenar para visualización
        total_weight = weights_a + weights_b
        sorted_indices = np.argsort(total_weight)[-15:] # Top 15

        final_mechanics = active_mechanics[sorted_indices]
        final_wa = weights_a[sorted_indices]
        final_wb = weights_b[sorted_indices]

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        y = np.arange(len(final_mechanics))
        height = 0.35
        ax.barh(y - height/2, final_wa, height, label=name_a, color='#00ffcc', alpha=0.8)
        ax.barh(y + height/2, final_wb, height, label=name_b, color='#ff0055', alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(final_mechanics, color='white', fontsize=9)
        ax.set_xlabel('Presencia (1 = Sí, 0 = No)', color='gray')
        ax.set_title(f'ADN EXACTO: {name_a} vs {name_b}', color='white', pad=20)
        ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')
        plt.tight_layout()
        plt.show()