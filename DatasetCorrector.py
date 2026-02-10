class DatasetCorrector:
    def __init__(self, df, correcciones_tiempo=None):
        if correcciones_tiempo is None:
            correcciones_tiempo = {
                'Chess': (10, 180),
                'Disney Lorcana': (20, 45),
                'BattleTech': (60, 180)
            }
        self.df = df
        self.correcciones_tiempo = correcciones_tiempo

    def correct_playtimes(self):
        # 1. Corrección de Tiempos de Juego
        for juego, (t_min, t_max) in self.correcciones_tiempo.items():
            mask = self.df['boardgame'] == juego
            if mask.any():
                self.df.loc[mask, 'min_playtime'] = t_min
                self.df.loc[mask, 'max_playtime'] = t_max

    def correct_player_counts(self):
        # 2. Corrección de Jugadores (EXIT: Advent Calendar)
        # Usamos regex para atrapar el nombre completo sin importar variaciones menores
        exit_mask = self.df['boardgame'].str.contains('EXIT.*Advent', case=False, na=False)
        if exit_mask.any():
            self.df.loc[exit_mask, 'max_players'] = 4

    def print_summary(self):
        # 3. Verificación de Cambios
        print("RESUMEN DE LAS CORRECCIONES:")
        print("-" * 60)
        print(f"Tiempos corregidos para: {', '.join(self.correcciones_tiempo.keys())}")
        print(f"Máximo de jugadores ajustado para EXIT: Advent Calendar")
        print(f"Dataset listo con {len(self.df)} registros.")

        # Comprobación rápida de una de las correcciones
        print("\nValidación post-limpieza (Ejemplo Chess):")
        print(self.df[self.df['boardgame'] == 'Chess'][['boardgame', 'min_playtime', 'max_playtime']])

    def correct(self):
        self.correct_playtimes()
        self.correct_player_counts()
        self.print_summary()
        return self.df
