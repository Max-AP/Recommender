import DatasetLoader

class DatasetChecker:
    def __init__(self, df=None):
        self.df = df
        self.columnas_clave = {
            'Identificación': ['row_id', 'boardgame', 'release_year'],
            'Correlación Peso vs Rating': ['complexity', 'avg_rating'],
            'Motor de Recomendación': ['min_players', 'max_players', 'min_playtime', 'max_playtime'],
            'Feature Engineering': ['mechanics'],
            'Filtrado y Segmentación': ['num_ratings', 'minimum_age']
        }

    def print_data_quality(self):
        print("VERIFICACIÓN DE DISPONIBILIDAD Y CALIDAD DE DATOS:")
        print("="*65)

        for categoria, columnas in self.columnas_clave.items():
            print(f"\nCategoría: {categoria}")
            for col in columnas:
                if col in self.df.columns:
                    # Cálculo de completitud
                    no_nulos = self.df[col].notna().sum()
                    porcentaje = (no_nulos / len(self.df)) * 100

                    print(f"  {col:20} | {no_nulos:,} valores | {porcentaje:>6.1f}% completo")
                else:
                    print(f"  {col:20} | NO ENCONTRADA EN EL DATASET")

    def print_mechanics_preview(self):
        print("\n" + "="*65)
        print("ESTRUCTURA DE MECÁNICAS (Dato crudo):")
        print("-" * 65)
        print(self.df['mechanics'].iloc[0] if 'mechanics' in self.df.columns else "Columna 'mechanics' no disponible")
