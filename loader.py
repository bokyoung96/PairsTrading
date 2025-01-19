import os
import pandas as pd    


class DataLoader:
    def __call__(self, data_name: str) -> pd.DataFrame:
        valid_data = {'price': 'data_price', 
                      'sector': 'data_sector',
                      'name': 'data_name'}
        if data_name not in valid_data:
            raise ValueError(f"Invalid data_name: {data_name}. Choose from {list(valid_data.keys())}")

        data_path = f'./DATA/{valid_data[data_name]}.pkl'

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        return pd.read_pickle(data_path)