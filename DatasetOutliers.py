class DatasetOutliers:
    def __init__(self, df):
        self.df = df

    def analyze_year_zero(self):
        year_zero = self.df[self.df['release_year'] == 0]
        print(f"\n[1] JUEGOS CON AÑO 0 (N={len(year_zero)})")
        print("-" * 40)
        print(year_zero[['boardgame', 'avg_rating', 'complexity', 'categories']].head())

    def analyze_time_zero(self):
        time_zero = self.df[(self.df['min_playtime'] == 0) | (self.df['max_playtime'] == 0)]
        print(f"\n[2] JUEGOS CON TIEMPO 0 (N={len(time_zero)})")
        print("-" * 40)
        print(time_zero[['boardgame', 'min_playtime', 'max_playtime', 'avg_rating']])

    def analyze_massive_players(self):
        massive_players = self.df[self.df['max_players'] >= 20].sort_values('max_players', ascending=False)
        print(f"\n[3] JUEGOS MASIVOS (Max Players >= 20, N={len(massive_players)})")
        print("-" * 40)
        # Mostramos los top 10 para identificar si son errores o Party Games legítimos
        print(massive_players[['boardgame', 'min_players', 'max_players', 'categories']].head(10))

    def analyze(self):
        print("="*75)
        print("ANÁLISIS DE CASOS ESPECIALES (OUTLIERS)")
        print("="*75)
        self.analyze_year_zero()
        self.analyze_time_zero()
        self.analyze_massive_players()
