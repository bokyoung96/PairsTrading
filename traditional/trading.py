import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import List, Tuple, Optional

from loader import DataLoader
from pairs import PairTradeData, PairTradeAnalyzer


class SinglePairTrader:
    def __init__(self,
                 pairs: Tuple[str, str],
                 price_data_range: str,
                 loader: Optional[DataLoader] = None):
        self.pairs = pairs
        
        self.price_data_range = price_data_range
        
        loader = loader or DataLoader()
        self.price = loader('price')
        self.name = loader('name')

    def get_price(self, df: pd.DataFrame) -> pd.DataFrame:
        end = df.index.max()
        num = int(self.price_data_range[:-1])
        unit = self.price_data_range[-1].upper()

        if unit == "Y":
            start = pd.Timestamp(end) - pd.DateOffset(years=num)
        elif unit == "M":
            start = pd.Timestamp(end) - pd.DateOffset(months=num)
        else:
            raise ValueError(f"Unsupported range: {self.price_data_range}")
        start = max(start, df.index.min())
        return df.loc[start: end]

    def get_pair(self, index: int) -> pd.Series:
        if index not in [0, 1]:
            raise IndexError("Index must be 0 or 1.")
        symbol = self.pairs[index]
        return self.get_price(self.price)[symbol]

    @property
    def pair_1(self) -> pd.Series:
        return self.get_pair(0)

    @property
    def pair_2(self) -> pd.Series:
        return self.get_pair(1)
    
    @property
    def log_pair_1(self) -> pd.Series:
        return self.pair_1.apply(lambda x: np.log(x))
    
    @property
    def log_pair_2(self) -> pd.Series:
        return self.pair_2.apply(lambda x: np.log(x))


if __name__ == "__main__":
    data = PairTradeData(sector_name='Financials',
                         price_data_range='1Y',
                         nan_handling='drop',
                         log_transform=True,
                         loader=None)
    
    analyzer = PairTradeAnalyzer(data=data,
                                 pipeline=None,
                                 coint_pvalue=0.05,
                                 adf_alpha=0.05,
                                 corr_threshold=0.5)
    res = analyzer.analyze_pairs()
    
    trader = SinglePairTrader(pairs=res[0],
                              price_data_range="1Y",
                              loader=None)
    
    p1 = trader.log_pair_1
    p2 = trader.log_pair_2