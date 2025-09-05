import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple

class GraphPlotter:
    def __init__(self,
                 sector_name: str,
                 date_filtered_price: pd.DataFrame):
        self.sector_name = sector_name
        self.date_filtered_price = date_filtered_price

    def plot_cointegration_heatmap(self, coint_pvalues_matrix: pd.DataFrame):
        if coint_pvalues_matrix.empty:
            print("[GraphPlotter] No matrix to plot.")
            return None

        mask = np.triu(np.ones_like(coint_pvalues_matrix, dtype=bool), k=1)
        masked_matrix = coint_pvalues_matrix.where(mask, np.nan)

        fig = px.imshow(
            masked_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=0,
            zmax=1,
            labels=dict(x="Ticker2", y="Ticker1", color="p-value"),
            x=coint_pvalues_matrix.columns,
            y=coint_pvalues_matrix.index,
            aspect="auto",
            origin='upper',
            title=f"Cointegration Heatmap: {self.sector_name}"
        )

        fig.update_layout(
            coloraxis_colorbar=dict(title="p-value"),
            plot_bgcolor='white',
            paper_bgcolor='white',
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        fig.update_traces(
            xgap=1,
            ygap=1,
            hoverongaps=False
        )
        return fig

    def plot_top_pairs(self, pairs: List[Tuple[str, str]], top_n: int = 5):
        if not pairs:
            print("[GraphPlotter] No pairs to plot.")
            return None
        selected = pairs[:top_n]
        fig = make_subplots(
            rows=len(selected),
            cols=1,
            shared_xaxes=True,
            subplot_titles=[f"Pair {i}" for i in range(1, len(selected) + 1)],
            specs=[[{"secondary_y": True}] for _ in range(len(selected))],
            vertical_spacing=0.03
        )
        for i, (t1, t2) in enumerate(selected, start=1):
            if (t1 not in self.date_filtered_price.columns) or (t2 not in self.date_filtered_price.columns):
                continue
            s1 = self.date_filtered_price[t1].dropna()
            s2 = self.date_filtered_price[t2].dropna()

            fig.add_trace(
                go.Scatter(x=s1.index, y=s1.values, mode='lines', name=f"{t1}", line_color='blue'),
                row=i, col=1, secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=s2.index, y=s2.values, mode='lines', name=f"{t2}", line_color='skyblue'),
                row=i, col=1, secondary_y=True
            )

        fig.update_layout(
            height=300 * len(selected),
            title_text=f"Top {top_n} Pairs: {self.sector_name}",
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True
        )
        return fig
    
    def plot_top_pairs_individual(self, pairs: List[Tuple[str, str]], top_n: int = 5):
        figures = []
        selected = pairs[:top_n]
        for i, (t1, t2) in enumerate(selected, start=1):
            if (t1 not in self.date_filtered_price.columns) or (t2 not in self.date_filtered_price.columns):
                print(f"[GraphPlotter] Pair {t1} - {t2} data not available.")
                continue
            s1 = self.date_filtered_price[t1].dropna()
            s2 = self.date_filtered_price[t2].dropna()

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=s1.index, y=s1.values, mode='lines', name=f"{t1}", line=dict(color='blue')),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=s2.index, y=s2.values, mode='lines', name=f"{t2}", line=dict(color='skyblue')),
                secondary_y=True,
            )

            fig.update_layout(
                title_text=f"Pair {i}: {t1} & {t2}",
                plot_bgcolor='white',
                paper_bgcolor='white',
                width=1200,
                height=600
            )
            fig.update_yaxes(title_text=t1, secondary_y=False)
            fig.update_yaxes(title_text=t2, secondary_y=True)

            figures.append((fig, (t1, t2))) 
        return figures
    
    def plot_pairs(self, s1: pd.Series, s2: pd.Series, t1: str, t2: str):
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=s1.index, y=s1.values, mode='lines', name=t1, line=dict(color='blue')),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=s2.index, y=s2.values, mode='lines', name=t2, line=dict(color='skyblue')),
            secondary_y=True,
        )

        fig.update_layout(
            title_text=f"Price Time Series: {t1} & {t2}",
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1200,
            height=600
        )
        fig.update_yaxes(title_text=t1, secondary_y=False)
        fig.update_yaxes(title_text=t2, secondary_y=True)
        return fig
