import os
import pandas as pd

class DatasetSave:
    def __init__(self, dataset):
        self.dataset = dataset

    def save(self):
        print("EJECUTANDO PERSISTENCIA DE DATOS")
        print("="*60)

        # 1. Definimos el nombre del archivo de salida
        output_file = 'boardgames_cleaned_final.csv'

        # 2. Guardamos el dataframe limpio (sin el índice numérico)
        self.dataset.df.to_csv(output_file, index=False)

        # 3. Verificación de seguridad
        if os.path.exists(output_file):
            file_size_kb = os.path.getsize(output_file) / 1024
            print(f"  Éxito: El dataset procesado se ha guardado en '{output_file}'.")
            print(f"  Tamaño del archivo: {file_size_kb:.2f} KB")
            print(f"  Total de juegos guardados: {len(self.dataset.df)}")
            print("\nEste archivo cumple con el requisito de 'Almacenamiento en CSV' de la propuesta.")
        else:
            print("Error: No se pudo crear el archivo.")
