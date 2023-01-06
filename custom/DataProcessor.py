import pandas as pd


class DataProcessor:
    def __init__(self):
        self.loaded_data = {}
        self.extra_data_processed = []
        self.preproc_funcs = {}
        pass

    def get_data(self, alias):
        return self.loaded_data[alias]

    def load_from_csv(self, path, alias):
        # print(f"Loaded id: {alias}")
        try:
            self.loaded_data[alias] = pd.read_csv(path)
            self.preproc_funcs[alias] = None
        except FileNotFoundError:
            print(f"DataProcessor: Could not load CSV file from {path}")

    def attach_preproc_func(self, alias, preproc_func):
        self.preproc_funcs[alias] = preproc_func

    def preprocess_data(self):
        for key, value in enumerate(self.preproc_funcs):
            result = self.preproc_funcs[value](self.loaded_data[value])
            if result is not None:
                self.loaded_data[value] = result[0]
            for i in range(1, len(result)):
                self.extra_data_processed.append(result[i])

    @staticmethod
    def replace_df_column(col, r_str, s_str, regex=False):
        return col.str.replace(r_str, s_str, regex=False)
