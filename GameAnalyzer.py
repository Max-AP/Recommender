from scipy import stats
from collections import Counter
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class GameAnalyzer:
    """
    Motor de análisis estadístico y visualización para el dataset de juegos.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.df = dataset.df

    def correlation_weight_rating(self) -> dict:
        """Calcula correlación entre Complejidad y Rating (Pearson y Spearman)."""
        complexity = self.df['complexity']
        rating = self.df['avg_rating']

        pearson = stats.pearsonr(complexity, rating)
        spearman = stats.spearmanr(complexity, rating)

        return {
            'pearson': {'val': pearson[0], 'p': pearson[1]},
            'spearman': {'val': spearman[0], 'p': spearman[1]},
            'n': len(complexity)
        }

    def mechanics_by_decade(self) -> dict:
        """Analiza la frecuencia de mecánicas agrupadas por década."""
        # Filtramos años 0 y calculamos década
        valid_games = self.df[self.df['release_year'] > 0].copy()
        valid_games['decade'] = (valid_games['release_year'] // 10) * 10

        trends = {}
        for decade, group in valid_games.groupby('decade'):
            # Aplanamos todas las listas de mecánicas de esa década
            # Asumimos que la columna 'mechanics' ya fue procesada a listas en pasos previos
            # Si no, hacemos split aquí. Para seguridad usamos try/split.
            raw_mechanics = group['mechanics'].fillna('').astype(str)
            all_mechs = [m.strip() for sublist in raw_mechanics.str.split(';') for m in sublist if m.strip()]
            trends[decade] = Counter(all_mechs).most_common(5)

        return trends

    def efficiency_test(self) -> dict:
        """Benchmark: Bucle Python vs NumPy Vectorizado."""
        ratings = self.df['avg_rating'].values

        # 1. Bucle Nativo
        start = time.time()
        _ = sum(ratings) / len(ratings)
        t_loop = time.time() - start

        # 2. NumPy
        start = time.time()
        _ = np.mean(ratings)
        t_numpy = time.time() - start

        return {'loop_sec': t_loop, 'numpy_sec': t_numpy, 'speedup': t_loop/t_numpy}

    def plot_dark_correlation(self):
        """Genera un gráfico de dispersión."""
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot con transparencia para ver densidad
        sns.scatterplot(
            data=self.df,
            x='complexity',
            y='avg_rating',
            alpha=0.4,
            color='#00ffcc', # Cian neón para contraste
            edgecolor=None,
            s=15,
            ax=ax
        )

        # Línea de tendencia roja intensa
        sns.regplot(
            data=self.df,
            x='complexity',
            y='avg_rating',
            scatter=False,
            color='#ff0055', # Rojo neón
            line_kws={'linewidth': 2},
            ax=ax
        )

        # Personalización
        ax.set_title('CORRELACIÓN: COMPLEJIDAD vs CALIDAD', fontsize=14, color='white', pad=20)
        ax.set_xlabel('Nivel de Complejidad (Weight)', fontsize=10, color='gray')
        ax.set_ylabel('Rating Promedio BGG', fontsize=10, color='gray')
        ax.grid(color='gray', linestyle=':', linewidth=0.3, alpha=0.5)

        # Eliminar bordes innecesarios
        sns.despine(left=True, bottom=True)

        plt.show()

    def __repr__(self):
        return f"<GameAnalyzer conectado a {len(self.df)} registros>"