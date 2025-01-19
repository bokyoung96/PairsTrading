import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go


class SpreadAnalyzer:
    def __init__(self, x: pd.Series, y: pd.Series):
        if len(x) != len(y):
            raise ValueError("The length of two assets does not equal.")
        self.x = x
        self.y = y
        self.weights = None
        self.spread = None
        self.calculate_spread()
        self.spread_dates = None

    def calculate_weights(self):
        X = sm.add_constant(self.x)
        model = sm.OLS(self.y, X).fit()
        self.weights = model.params.values
        return self.weights

    def calculate_spread(self):
        if self.weights is None:
            self.calculate_weights()
        const, slope = self.weights
        fitted_y = const + slope * self.x
        self.spread = self.y - fitted_y
        return self.spread

    def get_spread_stats(self):
        if self.spread is None:
            self.calculate_spread()
        
        mean = self.spread.mean()
        std = self.spread.std()
        
        adf_result = sm.tsa.stattools.adfuller(self.spread, autolag='AIC')
        adf_stat = adf_result[0]
        pvalue = adf_result[1]
        usedlag = adf_result[2]
        nobs = adf_result[3]
        critical_values = adf_result[4]
        icbest = adf_result[5]
        
        stats = {
            "Pair": (self.x.name, self.y.name),
            "Mean": mean,
            "Std Dev": std,
            "ADF Statistic": adf_stat,
            "ADF p-value": pvalue,
            "Number of Observations": nobs,
        }
        return stats
    
    def get_spread_dates(self):
        if self.spread is None:
            self.calculate_spread()
        
        mean = self.spread.mean()
        std = self.spread.std()
        
        spread_dates = pd.DataFrame(index=self.spread.index)
        spread_dates['Spread'] = self.spread
        spread_dates['Above_1_std'] = self.spread > (mean + std)
        spread_dates['Below_1_std'] = self.spread < (mean - std)
        spread_dates['Above_2_std'] = self.spread > (mean + 2 * std)
        spread_dates['Below_2_std'] = self.spread < (mean - 2 * std)
        
        spread_dates = spread_dates[
            spread_dates['Above_1_std'] | 
            spread_dates['Below_1_std'] |
            spread_dates['Above_2_std'] |
            spread_dates['Below_2_std']
        ]
        
        return spread_dates

    def plot_spread(self):
        if self.spread is None:
            self.calculate_spread()
        
        mean = self.spread.mean()
        std = self.spread.std()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.spread.index, 
            y=self.spread.values, 
            mode='lines', 
            name='Spread', 
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.spread.index, 
            y=[mean] * len(self.spread), 
            mode='lines', 
            name='Mean', 
            line=dict(color='red', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.spread.index, 
            y=[mean + std] * len(self.spread), 
            mode='lines', 
            name='+1 Std Dev', 
            line=dict(color='gray', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.spread.index, 
            y=[mean - std] * len(self.spread), 
            mode='lines', 
            name='-1 Std Dev', 
            line=dict(color='gray', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.spread.index, 
            y=[mean + 2 * std] * len(self.spread), 
            mode='lines', 
            name='+2 Std Dev', 
            line=dict(color='black', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.spread.index, 
            y=[mean - 2 * std] * len(self.spread), 
            mode='lines', 
            name='-2 Std Dev', 
            line=dict(color='black', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Spread Time Series: {self.x.name} & {self.y.name}',
            xaxis_title='Date',
            yaxis_title='Spread',
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                x=1,
                y=-0.2,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)'
            )
        )        
        return fig