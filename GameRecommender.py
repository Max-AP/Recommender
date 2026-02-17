"""
Optimized Game Recommender System
- Lazy similarity computation (saves ~3GB RAM)
- Cached preprocessing
- Vectorized operations
- Type hints for better IDE support
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Optional, Tuple, List, Dict, Set
from functools import lru_cache


class GameRecommender:
    """
    Recommendation system based on exact tag matching (MultiLabelBinarizer).
    Preserves compound mechanics (e.g., 'Worker Placement') as unique entities.

    OPTIMIZATIONS:
    - Lazy similarity computation (compute on-demand, not upfront)
    - Cached keyword preprocessing
    - No unnecessary dataframe copies
    - Type hints for better maintainability
    """

    def __init__(self, internal_dataset):
        print("Initializing Recommender Engine (Mode: MultiLabelBinarizer)...")
        self.dataset = internal_dataset
        # OPTIMIZATION: Use reference instead of copy (saves memory)
        self.df = internal_dataset.df

        self.family_stop_words: Set[str] = {
            'the', 'a', 'an', 'of', 'and', 'or', 'in', 'on', 'at', 'to', 'for',
            'edition', 'volume', 'vol', 'deluxe', 'legacy', 'second', 'third',
            'expansion', 'game', 'boardgame', 'pack', 'set', 'box'
        }

        # Sparse matrix for memory efficiency
        self.mlb = MultiLabelBinarizer(sparse_output=True)

        # OPTIMIZATION: Cache for preprocessed keywords
        self._game_keywords_cache: Dict[int, Set[str]] = {}

        # CRITICAL: Index-to-position mappings for mechanics_matrix
        self._idx_to_pos: Dict[int, int] = {}
        self._pos_to_idx: Dict[int, int] = {}

        self._build_model()
        print(f"Engine ready. {len(self.mlb.classes_)} unique tags (Mechanics + Categories).")

    def _build_model(self):
        """
        Build mechanics matrix preserving complete phrases.
        IMPROVEMENT: Merge 'mechanics' and 'categories' for complete profile.
        """
        # CRITICAL: Create mapping from dataframe index to matrix position
        # The mechanics_matrix is always 0-indexed, but df.index might not be
        self._idx_to_pos = {idx: pos for pos, idx in enumerate(self.df.index)}
        self._pos_to_idx = {pos: idx for pos, idx in enumerate(self.df.index)}

        # 1. Merge Mechanics + Categories text
        combined_tags = (
            self.df['mechanics'].fillna('') + ";" +
            self.df['categories'].fillna('')
        )

        # 2. Clean and convert to list
        tags_list = combined_tags.apply(
            lambda x: [m.strip() for m in x.split(';') if m.strip()]
        )

        # 3. Train MultiLabelBinarizer with extended set
        self.mechanics_matrix = self.mlb.fit_transform(tags_list)

        # OPTIMIZATION: Don't compute full similarity matrix upfront!
        # Was: self.similarity_matrix = cosine_similarity(self.mechanics_matrix)
        # This saves ~3.2GB for 20k games

        # 4. OPTIMIZATION: Pre-compute keyword cache for filtering
        print("Pre-computing keyword cache for fast filtering...")
        for idx in self.df.index:
            self._game_keywords_cache[idx] = self._get_clean_keywords(
                self.df.loc[idx, 'boardgame']
            )

        print(f"Model built with {len(self.mlb.classes_)} unique tags.")

    @lru_cache(maxsize=1000)
    def _get_clean_keywords(self, text: str) -> Set[str]:
        """
        Extract clean keywords from text (cached).

        OPTIMIZATION: Using LRU cache to avoid re-processing same titles
        """
        text_clean = re.sub(r'[^\w\s]', '', text.lower())
        return set(text_clean.split()) - self.family_stop_words

    def _resolve_game_id(self, game_name: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Find game index by name with fuzzy matching.

        Returns:
            (index, resolved_name) or (None, None) if not found
        """
        mask = self.df['boardgame'].str.contains(
            re.escape(game_name),
            case=False,
            na=False
        )
        matches = self.df[mask]

        if matches.empty:
            return None, None

        # Try exact match first
        exact_match = matches[matches['boardgame'].str.lower() == game_name.lower()]
        if not exact_match.empty:
            return exact_match.index[0], exact_match.iloc[0]['boardgame']

        # Otherwise, pick shortest title (likely base game)
        best_match = matches.loc[matches['boardgame'].str.len().idxmin()]
        return best_match.name, best_match['boardgame']

    def _compute_similarity_for_game(self, idx: int) -> np.ndarray:
        """
        OPTIMIZATION: Compute similarities on-demand for a single game.

        This replaces storing the full NxN matrix (saves ~3GB RAM).

        Args:
            idx: Dataframe index of the game
        """
        # Convert dataframe index to matrix position
        pos = self._idx_to_pos[idx]
        base_vector = self.mechanics_matrix[pos]
        similarities = cosine_similarity(
            base_vector,
            self.mechanics_matrix
        ).flatten()
        return similarities

    def recommend_similar_games(
        self,
        game_name: str,
        n: int = 5,
        complexity_tolerance: float = 0.5,
        exclude_family: bool = True
    ) -> pd.DataFrame:
        """
        Find games similar to the given game.

        Args:
            game_name: Name of the base game
            n: Number of recommendations
            complexity_tolerance: Max difference in complexity
            exclude_family: Exclude expansions/variants

        Returns:
            DataFrame with recommended games sorted by final_score
        """
        idx, resolved_name = self._resolve_game_id(game_name)
        if idx is None:
            return pd.DataFrame()  # Return empty instead of string error

        base_game = self.df.loc[idx]
        base_keywords = self._game_keywords_cache[idx]
        base_weight = base_game['complexity']

        print(f"Analyzing: {resolved_name} (Complexity: {base_weight:.2f})")

        # OPTIMIZATION: Compute similarities on-demand
        sim_scores = self._compute_similarity_for_game(idx)

        ratings = self.df['avg_rating'].fillna(0)
        norm_ratings = ratings / 10.0

        # Vectorized weight scoring
        weight_diff = np.abs(self.df['complexity'].values - base_weight)
        weight_score = 1 / (1 + weight_diff)

        # Hybrid formula: 50% mechanics, 30% rating, 20% complexity
        final_score = (sim_scores * 0.50) + (norm_ratings * 0.30) + (weight_score * 0.20)

        # OPTIMIZATION: Create result using proper indexing
        # Build series aligned with dataframe index
        res = self.df.copy()
        res['similarity'] = sim_scores
        res['weight_score'] = weight_score
        res['final_score'] = final_score

        # Filter out the base game itself
        res = res[res.index != idx]

        # Filter by complexity tolerance
        res = res[np.abs(res['complexity'] - base_weight) <= complexity_tolerance]

        # OPTIMIZATION: Vectorized family exclusion using cached keywords
        if exclude_family:
            exclude_mask = [
                len(base_keywords.intersection(self._game_keywords_cache[i])) == 0
                for i in res.index
            ]
            res = res[exclude_mask]

        return res.nlargest(n, 'final_score')

    def recommend_by_features(
        self,
        mechanics_list: Optional[List[str]] = None,
        min_players: int = 1,
        max_time: int = 120,
        n: int = 5
    ) -> pd.DataFrame:
        """
        Find games by desired mechanics and constraints.

        Args:
            mechanics_list: List of desired mechanics (case-sensitive)
            min_players: Minimum player count
            max_time: Maximum playtime
            n: Number of results

        Returns:
            DataFrame with matching games
        """
        # Filter by player count and time
        mask = (
            (self.df['min_players'] <= min_players) &
            (self.df['max_players'] >= min_players) &
            (self.df['max_playtime'] <= max_time)
        )
        candidates = self.df[mask]

        if mechanics_list and len(mechanics_list) > 0:
            try:
                # Transform query mechanics to vector
                query_vec = self.mlb.transform([mechanics_list])

                # Get candidate indices and their positions
                candidate_indices = candidates.index
                candidate_positions = [self._idx_to_pos[idx] for idx in candidate_indices]
                candidate_matrix = self.mechanics_matrix[candidate_positions]

                # Compute similarities
                sims = cosine_similarity(candidate_matrix, query_vec).flatten()

                # OPTIMIZATION: Create result without full copy
                result = candidates.copy()
                result['match'] = sims
                return result.nlargest(n, 'match')

            except ValueError as e:
                print(f"Warning: Some mechanics not found in dataset: {e}")
                return pd.DataFrame()

        # No mechanics specified, sort by rating
        return candidates.nlargest(n, 'avg_rating')

    def compare_games(
        self,
        game_a: str,
        game_b: str
    ) -> Dict:
        """
        Compare two games using set logic.

        Returns:
            Dictionary with comparison results
        """
        idx_a, name_a = self._resolve_game_id(game_a)
        idx_b, name_b = self._resolve_game_id(game_b)

        if idx_a is None or idx_b is None:
            return {"error": "Game not found"}

        # OPTIMIZATION: Compute similarity only for these two games
        pos_a = self._idx_to_pos[idx_a]
        pos_b = self._idx_to_pos[idx_b]
        base_vec = self.mechanics_matrix[pos_a]
        target_vec = self.mechanics_matrix[pos_b]
        sim = cosine_similarity(base_vec, target_vec)[0, 0]

        # Extract mechanic sets
        mechs_a = set(m.strip() for m in str(self.df.loc[idx_a, 'mechanics']).split(';') if m.strip())
        mechs_b = set(m.strip() for m in str(self.df.loc[idx_b, 'mechanics']).split(';') if m.strip())

        return {
            'names': (name_a, name_b),
            'similarity': float(sim),
            'shared': mechs_a.intersection(mechs_b),
            'unique_a': mechs_a - mechs_b,
            'unique_b': mechs_b - mechs_a
        }

    # Plotting methods (unchanged, but with better type hints)

    def plot_radar(self, game_a: str, game_b: str) -> None:
        """Radar chart comparing two games."""
        idx_a, name_a = self._resolve_game_id(game_a)
        idx_b, name_b = self._resolve_game_id(game_b)

        if idx_a is None or idx_b is None:
            print("Error: Game not found")
            return

        g1 = self.df.loc[idx_a]
        g2 = self.df.loc[idx_b]

        categories = ['Complexity', 'Rating', 'Min Players', 'Max Players']
        val1 = [
            g1['complexity'],
            g1['avg_rating']/2,
            g1['min_players'],
            min(g1['max_players'], 6)
        ]
        val2 = [
            g2['complexity'],
            g2['avg_rating']/2,
            g2['min_players'],
            min(g2['max_players'], 6)
        ]

        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        val1 += val1[:1]
        val2 += val2[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        plt.style.use('dark_background')

        ax.plot(angles, val1, linewidth=2, linestyle='solid',
                label=name_a, color='#00ffcc')
        ax.fill(angles, val1, '#00ffcc', alpha=0.2)

        ax.plot(angles, val2, linewidth=2, linestyle='solid',
                label=name_b, color='#ff0055')
        ax.fill(angles, val2, '#ff0055', alpha=0.2)

        plt.xticks(angles[:-1], categories, color='white', size=10)
        ax.set_rlabel_position(0)
        plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"],
                   color="gray", size=8)
        plt.ylim(0, 6)
        plt.title(f"PROFILE COMPARISON\n{name_a} vs {name_b}",
                  color='white', y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                   facecolor='#222', edgecolor='white', labelcolor='white')
        plt.show()

    def plot_vector_comparison(self, game_a: str, game_b: str) -> None:
        """Visualize tag matches between two games."""
        idx_a, name_a = self._resolve_game_id(game_a)
        idx_b, name_b = self._resolve_game_id(game_b)

        if idx_a is None or idx_b is None:
            print("Error: Game not found")
            return

        # Binary vectors from MLB
        pos_a = self._idx_to_pos[idx_a]
        pos_b = self._idx_to_pos[idx_b]
        vec_a = self.mechanics_matrix[pos_a].toarray().flatten()
        vec_b = self.mechanics_matrix[pos_b].toarray().flatten()

        feature_names = self.mlb.classes_

        # Find active mechanics
        mask = (vec_a > 0) | (vec_b > 0)
        active_mechanics = feature_names[mask]
        weights_a = vec_a[mask]
        weights_b = vec_b[mask]

        # Sort by total presence
        total_weight = weights_a + weights_b
        sorted_indices = np.argsort(total_weight)[-15:]  # Top 15

        final_mechanics = active_mechanics[sorted_indices]
        final_wa = weights_a[sorted_indices]
        final_wb = weights_b[sorted_indices]

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))

        y = np.arange(len(final_mechanics))
        height = 0.35

        ax.barh(y - height/2, final_wa, height,
                label=name_a, color='#00ffcc', alpha=0.8)
        ax.barh(y + height/2, final_wb, height,
                label=name_b, color='#ff0055', alpha=0.8)

        ax.set_yticks(y)
        ax.set_yticklabels(final_mechanics, color='white', fontsize=9)
        ax.set_xlabel('Presence (1 = Yes, 0 = No)', color='gray')
        ax.set_title(f'EXACT DNA: {name_a} vs {name_b}',
                     color='white', pad=20)
        ax.legend(facecolor='#222', edgecolor='white', labelcolor='white')

        plt.tight_layout()
        plt.show()