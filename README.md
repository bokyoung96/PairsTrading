# Pairs Trading Module

A sophisticated Python module for statistical arbitrage and pairs trading strategies, implementing cointegration-based pair selection, spread analysis, and backtesting capabilities.

## Overview

This module provides a comprehensive framework for pairs trading, including:
- Statistical pair selection using cointegration, stationarity tests, and correlation analysis
- Spread analysis and mean-reversion signal generation
- Portfolio backtesting with transaction cost modeling
- Visualization tools for trading signals and performance metrics

## Features

### 1. Pair Selection Filters
- **Cointegration Filter**: Identifies statistically cointegrated pairs using Engle-Granger test
- **Spread Stationarity Filter**: Tests for mean-reverting spreads using ADF test
- **Correlation Filter**: Filters pairs based on correlation thresholds

### 2. Data Management
- Automatic data loading and preprocessing
- Sector-based filtering for industry-specific pair trading
- Log transformation and missing data handling
- Configurable time periods (1Y, 6M, 3M, etc.)

### 3. Spread Analysis
- Z-score based entry/exit signals
- Customizable threshold parameters
- Trading date generation for backtesting

### 4. Transaction Cost Modeling
- Korea stock market specific costs
- Customizable transaction cost models
- Zero-cost option for theoretical analysis

## Installation

### Prerequisites
```bash
pip install numpy pandas statsmodels matplotlib seaborn
```

### Directory Structure
```
PairsTrading/
├── traditional/
│   ├── __init__.py
│   ├── pairs.py         # Core pair selection and analysis
│   ├── spreads.py       # Spread calculation and signal generation
│   ├── loader.py        # Data loading utilities
│   ├── trading.py       # Trading simulation and backtesting
│   ├── cost.py          # Transaction cost models
│   ├── plots.py         # Visualization utilities
│   ├── preprocess.py    # Data preprocessing
│   ├── uploader.py      # Configuration management
│   └── config.json      # Configuration file
└── README.md
```

## Quick Start

### Basic Usage

```python
from traditional.pairs import PairTradeData, PairTradeAnalyzer

# Initialize data loader with sector filtering
data = PairTradeData(
    sector_name='Financials',
    price_data_range='1Y',      # Use 1 year of historical data
    nan_handling='drop',         # Drop stocks with missing data
    log_transform=True           # Apply log transformation
)

# Create analyzer with statistical thresholds
analyzer = PairTradeAnalyzer(
    data=data,
    coint_pvalue=0.05,          # Cointegration p-value threshold
    adf_alpha=0.05,              # ADF test significance level
    corr_threshold=0.5           # Minimum correlation requirement
)

# Run analysis and get ranked pairs
sorted_pairs = analyzer.analyze_pairs(
    use_coint=True,              # Apply cointegration filter
    use_spread=True,             # Apply spread stationarity filter
    use_corr=True                # Apply correlation filter
)

# View results
print(analyzer.pipeline_history_df)  # Filter pipeline results
print(analyzer.spread_stats)         # Spread statistics for selected pairs
```

### Custom Filter Pipeline

```python
from traditional.pairs import TimeSeriesFilterPipeline, CointegrationFilter, CorrelationFilter

# Create custom pipeline
pipeline = TimeSeriesFilterPipeline()

# Add filters with custom parameters
coint_filter = CointegrationFilter(pvalue_threshold=0.01)
corr_filter = CorrelationFilter(corr_threshold=0.7, use_abs=True)

pipeline.add_filter(coint_filter, data_df, enabled=True)
pipeline.add_filter(corr_filter, data_df, enabled=True)

# Run pipeline
final_pairs = pipeline.run()
```

## Core Components

### PairTradeData
Handles data loading, preprocessing, and sector-based filtering.

**Parameters:**
- `sector_name`: Industry sector for pair selection
- `price_data_range`: Historical data period ('1Y', '6M', '3M', etc.)
- `nan_handling`: Method for handling missing data ('drop' or 'fill')
- `log_transform`: Apply logarithmic transformation to prices
- `loader`: Custom data loader instance

### PairTradeAnalyzer
Main analysis class for pair selection and spread analysis.

**Parameters:**
- `data`: PairTradeData instance
- `pipeline`: Custom filter pipeline (optional)
- `coint_pvalue`: Cointegration test p-value threshold
- `adf_alpha`: ADF test significance level
- `corr_threshold`: Minimum correlation requirement

### TimeSeriesFilterPipeline
Sequential filter application system for pair selection.

**Methods:**
- `add_filter()`: Add a filter to the pipeline
- `run()`: Execute all filters in sequence
- `filter_history_df`: Get DataFrame of filter results

## Filter Types

### CointegrationFilter
Tests for cointegration between time series pairs using the Engle-Granger method.

### SpreadStationarityFilter
Verifies spread stationarity using Augmented Dickey-Fuller test on OLS residuals.

### CorrelationFilter
Filters pairs based on Pearson correlation coefficients.

## Advanced Features

### Spread Analysis
```python
from traditional.spreads import SpreadAnalyzer

# Create spread analyzer for a specific pair
analyzer = SpreadAnalyzer(x=price_series_x, y=price_series_y)

# Get spread statistics
stats = analyzer.get_spread_stats()

# Get trading signals
dates = analyzer.get_spread_dates()
```

### Trading Simulation
```python
from traditional.trading import SinglePairTrader
from traditional.cost import KoreaTransactionCost

# Initialize trader with Korean market costs
trader = SinglePairTrader(
    x_series=stock_x_prices,
    y_series=stock_y_prices,
    transaction_cost=KoreaTransactionCost()
)

# Run backtest
results = trader.backtest()
```

## Configuration

The module uses `config.json` for default parameters:

```json
{
    "data_path": "path/to/data",
    "sectors": ["Financials", "Technology", "Healthcare"],
    "default_range": "1Y",
    "z_score_entry": 2.0,
    "z_score_exit": 0.5
}
```

## Performance Metrics

The analyzer provides comprehensive statistics:
- Cointegration p-values
- ADF test statistics
- Correlation coefficients
- Spread mean, std, and z-scores
- Trading signal dates
- Pair rankings based on combined metrics

## Best Practices

1. **Data Quality**: Ensure clean, adjusted price data
2. **Sector Selection**: Focus on economically related securities
3. **Parameter Tuning**: Optimize thresholds based on market conditions
4. **Risk Management**: Implement position sizing and stop-loss rules
5. **Out-of-Sample Testing**: Validate strategy on unseen data

## Example Output

```
Filter Pipeline Results:
                    Filter  RemainingPairs            PairsSample
0     CointegrationFilter              15  [(AAA, BBB), (CCC, DDD), ...]
1  SpreadStationarityFilter             8  [(AAA, BBB), (EEE, FFF), ...]
2       CorrelationFilter              5  [(AAA, BBB), (GGG, HHH), ...]

Top Ranked Pairs:
1. (AAA, BBB) - Coint: 0.01, ADF: 0.02, Corr: 0.85
2. (CCC, DDD) - Coint: 0.02, ADF: 0.01, Corr: 0.78
...
```

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Run existing tests before submitting PRs

## License

[Specify your license here]

## Contact

[Your contact information]

## Acknowledgments

This module uses:
- statsmodels for statistical testing
- pandas for data manipulation
- numpy for numerical computations
