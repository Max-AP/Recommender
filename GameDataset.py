import pandas as pd

class GameDataset:
    """
    Clase para el manejo, protección y manipulación del dataset de juegos de mesa.
    Permite realizar operaciones de filtrado encadenables.
    """
    def __init__(self, dataframe: pd.DataFrame):
        """Inicializa el dataset con una copia de seguridad para permitir reseteos."""
        self.df = dataframe.copy()
        self.original_df = dataframe.copy()

    def get_stats(self) -> dict:
        """Calcula estadísticas descriptivas rápidas del estado actual del dataset."""
        if self.df.empty:
            return {"error": "El dataset está vacío."}

        return {
            'total_juegos': len(self.df),
            'rango_años': (int(self.df['release_year'].min()), int(self.df['release_year'].max())),
            'rating_promedio': round(self.df['avg_rating'].mean(), 2),
            'complejidad_media': round(self.df['complexity'].mean(), 2),
            'mecánicas_únicas': self.df['mechanics'].str.split('; ').explode().nunique()
        }

    def filter_by_year(self, min_year: int = None, max_year: int = None):
        """Filtra juegos por un rango de años específico."""
        filtered_df = self.df.copy()
        if min_year is not None:
            filtered_df = filtered_df[filtered_df['release_year'] >= min_year]
        if max_year is not None:
            filtered_df = filtered_df[filtered_df['release_year'] <= max_year]
        return GameDataset(filtered_df)

    def filter_by_players(self, num_players: int, exact: bool = False):
        """
        Filtra juegos por número de jugadores.

        Args:
            num_players: El número de jugadores deseado.
            exact: Si es True, busca juegos diseñados ÚNICAMENTE para ese número.
                   Si es False, busca juegos que soporten ese número dentro de su rango.
        """
        if exact:
            # Caso "Exactamente X": min == max == num_players
            filtered_df = self.df[
                (self.df['min_players'] == num_players) &
                (self.df['max_players'] == num_players)
            ]
        else:
            # Caso "Soporta X": num_players está en el rango [min, max]
            filtered_df = self.df[
                (self.df['min_players'] <= num_players) &
                (self.df['max_players'] >= num_players)
            ]
        return GameDataset(filtered_df)

    def filter_by_playtime(self, max_time: int):
        """Filtra juegos que no excedan el tiempo de juego máximo proporcionado."""
        filtered_df = self.df[self.df['max_playtime'] <= max_time]
        return GameDataset(filtered_df)

    def filter_by_rating(self, min_rating: float):
        """Filtra por calidad mínima basada en el rating promedio de la comunidad."""
        filtered_df = self.df[self.df['avg_rating'] >= min_rating]
        return GameDataset(filtered_df)

    def reset(self):
        """Restaura el dataset a su estado original post-limpieza."""
        self.df = self.original_df.copy()
        return self

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return f"<GameDataset: {len(self.df)} juegos cargados>"