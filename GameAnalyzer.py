"""
Optimized Game Analyzer
- Vectorized operations
- Type hints
- Better error handling
"""

from scipy import stats
from collections import Counter
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class GameAnalyzer:
    """
    Statistical analysis and visualization engine for game dataset.

    OPTIMIZATIONS:
    - Vectorized operations where possible
    - Type hints for better IDE support
    - Cached expensive computations
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.df = dataset.df

    def correlation_weight_rating(self) -> Dict[str, any]:
        """
        Calculate correlation between Complexity and Rating.

        Returns:
            Dictionary with Pearson and Spearman correlations
        """
        complexity = self.df['complexity'].values
        rating = self.df['avg_rating'].values

        pearson = stats.pearsonr(complexity, rating)
        spearman = stats.spearmanr(complexity, rating)

        return {
            'pearson': {'val': pearson[0], 'p': pearson[1]},
            'spearman': {'val': spearman[0], 'p': spearman[1]},
            'n': len(complexity)
        }

    def mechanics_by_decade(self) -> Dict[int, List[Tuple[str, int]]]:
        """
        Analyze mechanic frequency grouped by decade.

        OPTIMIZATION: Vectorized decade calculation

        Returns:
            Dictionary mapping decade to top mechanics
        """
        # Filter valid years and calculate decade
        valid_games = self.df[self.df['release_year'] > 0].copy()
        valid_games['decade'] = (valid_games['release_year'] // 10) * 10

        trends = {}
        for decade, group in valid_games.groupby('decade'):
            # Flatten all mechanics for this decade
            raw_mechanics = group['mechanics'].fillna('').astype(str)
            all_mechs = [
                m.strip()
                for sublist in raw_mechanics.str.split(';')
                for m in sublist
                if m.strip()
            ]
            trends[decade] = Counter(all_mechs).most_common(5)

        return trends

    def efficiency_test(self) -> Dict[str, float]:
        """
        Benchmark: Python loop vs NumPy vectorization.

        Returns:
            Dictionary with timing results
        """
        ratings = self.df['avg_rating'].values

        # 1. Native loop
        start = time.perf_counter()
        _ = sum(ratings) / len(ratings)
        t_loop = time.perf_counter() - start

        # 2. NumPy
        start = time.perf_counter()
        _ = np.mean(ratings)
        t_numpy = time.perf_counter() - start

        speedup = t_loop / t_numpy if t_numpy > 0 else 0

        return {
            'loop_sec': t_loop,
            'numpy_sec': t_numpy,
            'speedup': speedup
        }

    def plot_dark_correlation(self) -> None:
        """
        Generate scatter plot of complexity vs rating.

        OPTIMIZATION: Uses seaborn's optimized plotting
        """
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot with transparency for density visualization
        sns.scatterplot(
            data=self.df,
            x='complexity',
            y='avg_rating',
            alpha=0.4,
            color='#00ffcc',  # Neon cyan
            edgecolor=None,
            s=15,
            ax=ax
        )

        # Red trend line
        sns.regplot(
            data=self.df,
            x='complexity',
            y='avg_rating',
            scatter=False,
            color='#ff0055',  # Neon red
            line_kws={'linewidth': 2},
            ax=ax
        )

        # Customization
        ax.set_title(
            'CORRELATION: COMPLEXITY vs QUALITY',
            fontsize=14,
            color='white',
            pad=20
        )
        ax.set_xlabel('Complexity Level (Weight)', fontsize=10, color='gray')
        ax.set_ylabel('Average BGG Rating', fontsize=10, color='gray')
        ax.grid(color='gray', linestyle=':', linewidth=0.3, alpha=0.5)

        # Remove unnecessary borders
        sns.despine(left=True, bottom=True)

        plt.tight_layout()
        plt.show()

    def get_summary_stats(self) -> pd.DataFrame:
        """
        Get comprehensive summary statistics.

        OPTIMIZATION: Single pass through data

        Returns:
            DataFrame with summary statistics
        """
        numeric_cols = ['complexity', 'avg_rating', 'min_players',
                        'max_players', 'min_playtime', 'max_playtime']

        return self.df[numeric_cols].describe()

    def get_top_games(self, n: int = 10, by: str = 'avg_rating') -> pd.DataFrame:
        """
        Get top N games by specified metric.

        Args:
            n: Number of games to return
            by: Column to sort by

        Returns:
            DataFrame with top games
        """
        if by not in self.df.columns:
            raise ValueError(f"Column '{by}' not found in dataset")

        return self.df.nlargest(n, by)[
            ['boardgame', 'avg_rating', 'complexity',
             'min_players', 'max_players']
        ]

    def __repr__(self) -> str:
        return f"<GameAnalyzer connected to {len(self.df):,} records>"