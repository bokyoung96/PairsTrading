import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from itertools import combinations
from abc import ABC, abstractmethod
from typing import List, Tuple

from loader import DataLoader
from spreads import SpreadAnalyzer


class TimeSeriesCombinationFilter(ABC):
    @abstractmethod
    def apply(self, data: pd.DataFrame, fixed_sets: List[Tuple[str, str]] = None) -> List[Tuple[str, str]]:
        pass


class CointegrationFilter(TimeSeriesCombinationFilter):
    def __init__(self, pvalue_threshold: float = 0.05):
        self.pvalue_threshold = pvalue_threshold
        self.coint_pvalues_matrix: pd.DataFrame = pd.DataFrame()
        self._stats_data = {}

    def apply(self, data: pd.DataFrame, fixed_sets: List[Tuple[str, str]] = None) -> List[Tuple[str, str]]:
        data = data.where(data > 0)
        cols = data.columns
        pvals = pd.DataFrame(np.nan, index=cols, columns=cols)
        self._stats_data = {}
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                x, y = cols[i], cols[j]
                sorted_pair = tuple(sorted((x, y)))
                df_xy = data[[x, y]].dropna()
                if df_xy.empty:
                    continue
                try:
                    _, pval, _ = coint(df_xy[x], df_xy[y])
                    pvals.loc[x, y] = pval
                    pvals.loc[y, x] = pval
                    self._stats_data[sorted_pair] = pval
                except:
                    pass

        self.coint_pvalues_matrix = pvals
        results = []
        combs = fixed_sets if fixed_sets else list(combinations(cols, 2))
        for x, y in combs:
            sorted_pair = tuple(sorted((x, y)))
            val = pvals.loc[x, y]
            if pd.notna(val) and (val < self.pvalue_threshold):
                results.append(sorted_pair)
        return results

    @property
    def stats_data(self):
        return self._stats_data


class SpreadStationarityFilter(TimeSeriesCombinationFilter):
    def __init__(self, adf_alpha: float = 0.05):
        self.adf_alpha = adf_alpha
        self._stats_data = {}

    def apply(self, data: pd.DataFrame, fixed_sets: List[Tuple[str, str]] = None):
        results = []
        self._stats_data = {}
        cols = data.columns
        combs = fixed_sets if fixed_sets else list(combinations(cols, 2))
        for x, y in combs:
            sorted_pair = tuple(sorted((x, y)))
            df_xy = data[[x, y]].dropna()
            if df_xy.empty:
                continue
            X_ = sm.add_constant(df_xy[x])
            Y_ = df_xy[y]
            try:
                model = sm.OLS(Y_, X_).fit()
                resid = model.resid
                _, pval, *_ = adfuller(resid, autolag="AIC")
                self._stats_data[sorted_pair] = pval
                if pval < self.adf_alpha:
                    results.append(sorted_pair)
            except:
                pass
        return results

    @property
    def stats_data(self):
        return self._stats_data


class CorrelationFilter(TimeSeriesCombinationFilter):
    def __init__(self, corr_threshold: float = 0.5, use_abs: bool = True):
        self.corr_threshold = corr_threshold
        self.use_abs = use_abs
        self._stats_data = {}

    def apply(self, data: pd.DataFrame, fixed_sets: List[Tuple[str, str]] = None):
        results = []
        self._stats_data = {}
        cols = data.columns
        combs = fixed_sets if fixed_sets else list(combinations(cols, 2))
        corr_matrix = data.corr()
        for x, y in combs:
            sorted_pair = tuple(sorted((x, y)))
            c = corr_matrix.loc[x, y]
            self._stats_data[sorted_pair] = c
            if self.use_abs:
                if abs(c) >= self.corr_threshold:
                    results.append(sorted_pair)
            else:
                if c >= self.corr_threshold:
                    results.append(sorted_pair)
        return results

    @property
    def stats_data(self):
        return self._stats_data


class TimeSeriesFilterPipeline:
    def __init__(self):
        self.steps: List[Tuple[TimeSeriesCombinationFilter, pd.DataFrame, bool]] = []
        self._filter_history = []

    def add_filter(self, _filter: TimeSeriesCombinationFilter, data: pd.DataFrame, enabled: bool = True):
        self.steps.append((_filter, data, enabled))

    def run(self, initial_pairs: List[Tuple[str, str]] = None) -> List[Tuple[str, str]]:
        def get_stat(stats_data: dict, pair: Tuple[str, str]) -> float:
            sorted_pair = tuple(sorted(pair))
            return stats_data.get(sorted_pair, None)
        
        combs = initial_pairs
        for (flt, df, enabled) in self.steps:
            if not enabled:
                self._filter_history.append((flt.__class__.__name__, None, combs if combs else [], {}))
                continue

            new_combs = flt.apply(df, fixed_sets=combs)
            stats_data = flt.stats_data if hasattr(flt, "stats_data") else {}

            final_stats = {p: get_stat(stats_data, p) for p in new_combs if get_stat(stats_data, p) is not None}

            self._filter_history.append((flt.__class__.__name__, len(new_combs), new_combs, final_stats))
            combs = new_combs
            if not combs:
                break
        return combs if combs else []

    @property
    def filter_history_df(self) -> pd.DataFrame:
        data = []
        for step_info in self._filter_history:
            fname, num_pairs, pairs_list, stats_data = step_info
            data.append({
                "Filter": fname,
                "RemainingPairs": num_pairs if num_pairs is not None else "Skipped",
                "PairsSample": pairs_list[:5] if pairs_list else [],
                "StatsData": stats_data
            })
        return pd.DataFrame(data)


class PairTradeData:
    def __init__(self,
                 sector_name: str,
                 price_data_range: str = "1Y",
                 nan_handling: str = "drop",
                 log_transform: bool = True,
                 loader: DataLoader = None):
        self.sector_name = sector_name
        self.price_data_range = price_data_range
        self.nan_handling = nan_handling
        self.log_transform = log_transform
        self.loader = loader or DataLoader()
        
        self.price = None
        self.sector = None
        self.name = None
        self.sector_map = {}
        
        self.filtered_price = pd.DataFrame()
        self.sector_groups = {}

        self.prepare_data()

    def prepare_data(self):
        self.load_data()
        self.check_sector()
        self.make_sector_map()
        
        self.filtered_price = self.filter_price()
        self.handle_nan()
        self.transform_log()
        
        self.sector_groups = self.group_by_sector()
    
    def load_data(self):
        try:
            self.price = self.loader("price")
            self.sector = self.loader("sector")
            self.name = self.loader("name")
        except Exception as e:
            raise ValueError("Failed to load price / sector / name") from e

    def check_sector(self):
        if self.sector.empty or self.sector.shape[0] < 1:
            raise ValueError("Invalid sector data.")

    def make_sector_map(self):
        self.sector_map = self.sector.iloc[0].to_dict()

    def filter_price(self) -> pd.DataFrame:
        if not self.price_data_range:
            return self.price

        df = self.price
        end = df.index.max()

        if self.price_data_range.endswith("Y"):
            start = pd.Timestamp(end) - pd.DateOffset(years=int(self.price_data_range[:-1]))
        elif self.price_data_range.endswith("M"):
            start = pd.Timestamp(end) - pd.DateOffset(months=int(self.price_data_range[:-1]))
        else:
            raise ValueError(f"Unsupported range: {self.price_data_range}")

        return df.loc[start: end]

    def handle_nan(self):
        if self.nan_handling == "drop":
            self.filtered_price = self.filtered_price.dropna(axis=1, how="any")
        else:
            self.filtered_price = self.filtered_price.fillna(method="ffill")
            self.filtered_price = self.filtered_price.fillna(method="bfill")

    def transform_log(self):
        if self.log_transform:
            self.filtered_price = (self.filtered_price
                                   .where(self.filtered_price > 0)
                                   .apply(lambda x: x.map(np.log))
                                   )

    def group_by_sector(self) -> dict:
        groups = {}
        for ticker, sec in self.sector_map.items():
            if ticker in self.filtered_price.columns:
                groups.setdefault(sec, []).append(ticker)
        return {s: self.filtered_price[tickers]
                for s, tickers in groups.items() if tickers}

    def get_sector_data(self) -> pd.DataFrame:
        if self.sector_name not in self.sector_groups:
            raise ValueError(f"Sector '{self.sector_name}' not found or no data.")
        return self.sector_groups[self.sector_name]


class PairTradeAnalyzer:
    def __init__(self,
                 data: PairTradeData,
                 pipeline: TimeSeriesFilterPipeline = None,
                 coint_pvalue: float = 0.05,
                 adf_alpha: float = 0.05,
                 corr_threshold: float = 0.5):
        self.data = data
        self.sector_name = self.data.sector_name
        
        if pipeline == None:
            pipeline = TimeSeriesFilterPipeline()
        self.pipeline = pipeline
        
        self.coint_pvalue = coint_pvalue
        self.adf_alpha = adf_alpha
        self.corr_threshold = corr_threshold

        self.coint_filter = CointegrationFilter(pvalue_threshold=self.coint_pvalue)
        self.spread_filter = SpreadStationarityFilter(adf_alpha=self.adf_alpha)
        self.corr_filter = CorrelationFilter(corr_threshold=self.corr_threshold,
                                             use_abs=True)

        self.filter_history = []
        self.all_stats = {}

        self.spread_objects: List[SpreadAnalyzer] = []
        self.spread_stats: pd.DataFrame = pd.DataFrame()
        self.spread_dates: dict = {}

    def run_filter_pipeline(self,
                            use_coint: bool = True,
                            use_spread: bool = True,
                            use_corr: bool = True) -> List[Tuple[str, str]]:
        sector_df = self.data.get_sector_data()
        if sector_df.empty:
            print("[run_filter_pipeline] No valid data.")
            return []

        self.pipeline.add_filter(self.coint_filter, sector_df, enabled=use_coint)
        self.pipeline.add_filter(self.spread_filter, sector_df, enabled=use_spread)
        self.pipeline.add_filter(self.corr_filter, sector_df, enabled=use_corr)

        final_pairs = self.pipeline.run()
        self.filter_history = self.pipeline._filter_history
        return final_pairs
    
    def get_filter_stats(self, final_pairs: List[Tuple[str, str]]):
        for step_info in self.filter_history:
            fname, _, _, stats_data = step_info
            for pair in final_pairs:
                if pair in stats_data:
                    self.all_stats.setdefault(pair, {})[fname] = stats_data[pair]
                else:
                    rev_pair = (pair[1], pair[0])
                    if rev_pair in stats_data:
                        self.all_stats.setdefault(pair, {})[fname] = stats_data[rev_pair]

    def get_pair_ranks(self) -> List[Tuple[str, str]]:
        df_stats = pd.DataFrame.from_dict(self.all_stats, orient='index')

        ranks = {}
        if 'CointegrationFilter' in df_stats.columns:
            ranks['CointegrationFilter'] = df_stats['CointegrationFilter'].rank(method='min', ascending=True)
        if 'SpreadStationarityFilter' in df_stats.columns:
            ranks['SpreadStationarityFilter'] = df_stats['SpreadStationarityFilter'].rank(method='min', ascending=True)
        if 'CorrelationFilter' in df_stats.columns:
            ranks['CorrelationFilter'] = df_stats['CorrelationFilter'].rank(method='min', ascending=False)

        df_ranks = pd.DataFrame(ranks)
        df_ranks['TotalRank'] = df_ranks.sum(axis=1)
        df_sorted = df_ranks.sort_values(by='TotalRank')
        
        res = df_sorted.index.tolist()
        return res
    
    def get_spread_objs(self, 
                        final_pairs: List[Tuple[str, str]]) -> List[SpreadAnalyzer]:
        sector_df = self.data.get_sector_data()
        spreads = []
        for x, y in final_pairs:
            x_series = sector_df[x]
            y_series = sector_df[y]
            analyzer = SpreadAnalyzer(x=x_series, y=y_series)
            spreads.append(analyzer)
        self.spread_objects = spreads
        return spreads
    
    def analyze_spreads(self, spreads: List[SpreadAnalyzer]):
        stats_list = []
        for spread in spreads:
            stats = spread.get_spread_stats()
            stats_list.append(stats)

            pair_key = f"{spread.x.name}-{spread.y.name}"
            self.spread_dates[pair_key] = spread.get_spread_dates()
            print(f"Pair: {pair_key}, TriggerDates:\n{self.spread_dates[pair_key]}\n")

        self.spread_stats = pd.DataFrame(stats_list)
    
    def analyze_pairs(self,
                      use_coint=True,
                      use_spread=True,
                      use_corr=True) -> List[Tuple[str, str]]:
        final_pairs = self.run_filter_pipeline(use_coint, 
                                               use_spread,
                                               use_corr)
        if not final_pairs:
            return []

        self.get_filter_stats(final_pairs)
        ranked_pairs = self.get_pair_ranks()

        spreads = self.get_spread_objs(ranked_pairs)
        self.analyze_spreads(spreads)
        return ranked_pairs

    @property
    def pipeline_history_df(self) -> pd.DataFrame:
        if not self.pipeline:
            return pd.DataFrame()
        return self.pipeline.filter_history_df


class Temp(PairTradeAnalyzer):
    def __init__(self, data, pipeline = None, coint_pvalue = 0.05, adf_alpha = 0.05, corr_threshold = 0.5):
        super().__init__(data, pipeline, coint_pvalue, adf_alpha, corr_threshold)

if __name__ == "__main__":
    data = PairTradeData(sector_name='Financials',
                         price_data_range='1Y',
                         nan_handling='drop',
                         log_transform=True,
                         loader=None)
    
    analyzer = PairTradeAnalyzer(
        data=data,
        pipeline=None,
        coint_pvalue=0.05,
        adf_alpha=0.05,
        corr_threshold=0.5
    )
    sorted_pairs = analyzer.analyze_pairs(use_coint=True, 
                                          use_spread=True, 
                                          use_corr=True)

