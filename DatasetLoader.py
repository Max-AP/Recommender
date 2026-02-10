import pandas as pd

class DatasetLoader:
    def __init__(self, file_path='boardgame-geek-dataset_organized.csv'):
        self.file_path = file_path
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.file_path)
        self.df = self.df.drop_duplicates(subset='boardgame').reset_index(drop=True)

        return self.df
