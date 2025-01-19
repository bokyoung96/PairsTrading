import pandas as pd


class PreProcess:
    def __init__(self,
                 file_name: str,
                 sheet_name: str,
                 **kwargs):
        self.file_name = file_name
        self.sheet_name = sheet_name

        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def get_QW_res_data(self) -> pd.DataFrame:
        data = pd.read_excel(f"./DATA/{self.file_name}.xlsx",
                             index_col=0,
                             sheet_name=self.sheet_name,
                             header=7).iloc[6:, :]
        data.index = pd.to_datetime(data.index)
        data.index.name = 'Date'
        return data
    
    def get_sector_data(self) -> pd.DataFrame:
        data = pd.read_excel(f'./DATA/{self.file_name}.xlsx',
                             index_col=0,
                             sheet_name='sector',
                             header=0)
        data.index = pd.to_datetime(data.index)
        data.index.name = 'Date'
        return data
    
    def get_QW_name_data(self) -> pd.DataFrame:
        data = pd.read_excel(f'./DATA/{self.file_name}.xlsx',
                             index_col=0,
                             sheet_name=self.sheet_name,
                             header=0).iloc[6: 8, :].T.reset_index(drop=True)
        data.columns.name = None
        return data
    
    def save_data(self, method=None):
        if method is None:
            method = 'get_QW_res_data'

        try:
            if hasattr(self, method):
                data_method = getattr(self, method)
                data = data_method()
                if method == 'get_QW_name_data':
                    data.to_pickle("./DATA/data_name.pkl")
                else:
                    data.to_pickle(f"./DATA/data_{self.sheet_name}.pkl")
        except Exception as e:
            print(f"Error: {e}")
        return data
    
def preprocess_all(file_name: str = 'DATA'):
    try:
        pp_price = PreProcess(file_name=file_name, sheet_name="price")
        price_data = pp_price.save_data(method='get_QW_res_data')
        print("Price preprocessed and saved.")
        
        pp_sector = PreProcess(file_name=file_name, sheet_name="sector")
        sector_data = pp_sector.save_data(method='get_sector_data')
        print("Sector preprocessed and saved.")
        
        pp_name = PreProcess(file_name=file_name, sheet_name="price")
        name_data = pp_name.save_data(method='get_QW_name_data')
        print("Name preprocessed and saved.")
        return True
    
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return False
    