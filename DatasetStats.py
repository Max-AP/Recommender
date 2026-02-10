import DatasetLoader

class DatasetStats:
    def __init__(self, df):
        self.df = df

    def print_general_stats(self):
        print("ESTADÍSTICAS GENERALES DEL DATASET")
        print("="*60)
        print(f"Rango de Años:         {self.df['release_year'].min()} - {self.df['release_year'].max()}")
        print(f"Rating Promedio:       {self.df['avg_rating'].mean():.2f} (±{self.df['avg_rating'].std():.2f})")
        print(f"Complejidad (Weight):  {self.df['complexity'].mean():.2f} (±{self.df['complexity'].std():.2f})")

        print(f"\nConfiguración de Jugadores:")
        print(f"  Mínimo: {self.df['min_players'].min()} | Máximo: {self.df['max_players'].max()}")

        print(f"\nTiempos de Juego (Minutos):")
        print(f"  Rango Mínimo: {self.df['min_playtime'].min()} - {self.df['min_playtime'].max()}")
        print(f"  Rango Máximo: {self.df['max_playtime'].min()} - {self.df['max_playtime'].max()}")

    def print_mechanics_stats(self):
        self.df['mechanics_list'] = self.df['mechanics'].fillna('').apply(lambda x: [m.strip() for m in x.split(';') if m.strip()])

        # Expandimos la lista para contar frecuencias
        all_mechanics = self.df['mechanics_list'].explode()

        print("\nPERFIL DE DISEÑO (MECHANICS)")
        print("-" * 60)
        print(f"Mecánicas únicas identificadas: {all_mechanics.nunique()}")
        print(f"\nTop 10 mecánicas más utilizadas:")
        print(all_mechanics.value_counts().head(10))
