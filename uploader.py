import os
import json
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

from loader import DataLoader
from pairs import PairTradeData, PairTradeAnalyzer
from plots import GraphPlotter
from spreads import SpreadAnalyzer
from preprocess import preprocess_all


SECTORS = [
    "Financials",
    "Communication Services",
    "Energy",
    "Industrials",
    "Health Care",
    "Information Technology",
    "Consumer Staples",
    "Utilities",
    "Materials",
    "Consumer Discretionary"
]

class ConfigLoader:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            st.error(f"Configuration file not found: {self.config_path}")
            st.stop()
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)

def load_config(config_path='config.json'):
    if not os.path.exists(config_path):
        st.error(f"Configuration file not found: {config_path}")
        st.stop()
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

class PairTradingApp:
    def __init__(self):
        st.set_page_config(layout="wide")
        self.config_loader = ConfigLoader()
        self.loader = DataLoader()
        
        if "admin_authenticated" not in st.session_state:
            st.session_state["admin_authenticated"] = False
        if "password_key" not in st.session_state:
            st.session_state["password_key"] = 0
        if "analysis_done" not in st.session_state:
            st.session_state["analysis_done"] = False
        if "final_pairs" not in st.session_state:
            st.session_state["final_pairs"] = []
        if "pipeline_history" not in st.session_state:
            st.session_state["pipeline_history"] = pd.DataFrame()
        if "date_filtered_df" not in st.session_state:
            st.session_state["date_filtered_df"] = None
        if "sector_name" not in st.session_state:
            st.session_state["sector_name"] = ""
        if "coint_pvalues_matrix" not in st.session_state:
            st.session_state["coint_pvalues_matrix"] = pd.DataFrame()
        if "ticker_mapping" not in st.session_state:
            st.session_state["ticker_mapping"] = {}
        if "all_stats" not in st.session_state:
            st.session_state["all_stats"] = pd.DataFrame()
        if "spread_dates" not in st.session_state:
            st.session_state["spread_dates"] = {}

    def run(self):
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] {
                min-width: 350px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.title("Pair Trading")

        (
            name_df, price_df,
            use_coint, use_spread, use_corr,
            coint_pval, adf_alpha, corr_threshold,
            sector_name, date_range, log_transform, nan_method,
            top_n_val, run_button
        ) = self.render_sidebar()

        if run_button:
            self.handle_analysis(
                name_df=name_df,
                price_df=price_df,
                use_coint=use_coint,
                use_spread=use_spread,
                use_corr=use_corr,
                coint_pval=coint_pval,
                adf_alpha=adf_alpha,
                corr_threshold=corr_threshold,
                sector_name=sector_name,
                date_range=date_range,
                log_transform=log_transform,
                nan_method=nan_method
            )

        if st.session_state["analysis_done"]:
            self.show_results(top_n_val)
        else:
            st.info("No analysis has been run yet.")

    def render_sidebar(self):
        with st.sidebar:
            try:
                name_df = self.loader('name')
                price_df = self.loader('price')
            except Exception as e:
                st.error(f"Data loading failed: {e}")
                st.stop()

            last_updated = price_df.index[-1]
            st.write(f"#### Last updated: {datetime.strftime(last_updated, '%Y-%m-%d')}")
            st.write("#### ShinhanAMC Quant Investment Center")

            st.header("Filter Settings")
            use_coint = st.checkbox("Use Cointegration Filter", value=True)
            use_spread = st.checkbox("Use ADF Spread Filter", value=True)
            use_corr = st.checkbox("Use Correlation Filter", value=True)

            coint_pval = st.number_input(
                "Cointegration p-value threshold", 
                value=0.05, 
                min_value=0.0, 
                max_value=1.0, 
                step=0.01,
                help="The p-value threshold for the cointegration test."
            )
            adf_alpha = st.number_input(
                "ADF alpha", 
                value=0.05, 
                min_value=0.0, 
                max_value=1.0, 
                step=0.01,
                help="The significance level for the Augmented Dickey-Fuller test on the spread."
            )
            corr_threshold = st.number_input(
                "Correlation threshold", 
                value=0.5,
                min_value=0.0, 
                max_value=1.0, 
                step=0.05,
                help="Pairs with correlation above this threshold will be considered."
            )

            sector_name = st.selectbox(
                "Sector Name", 
                SECTORS, 
                index=0,
                help="Data from Bloomberg, mainly focused in GICS sector classification."
            )

            date_range = st.selectbox(
                "Date Range", 
                ["3Y", "1Y", "6M", "3M", "1M"], 
                index=0,
                help="The length of data used for analysis."
            )
            log_transform = st.checkbox("Log Transform", value=True)
            nan_method = st.selectbox(
                "NaN Handling", 
                ["drop", "fill"], 
                index=0,
                help="Determines how to handle missing values in the price data: either remove them or fill them."
            )

            st.markdown("---")
            st.header("Output Settings")
            top_n_val = st.number_input(
                "How many top pairs to show?", 
                value=5, 
                min_value=1, 
                step=1
            )

            run_button = st.button("Run Analysis")

            st.markdown("---")
            self.render_admin_section()

        return (
            name_df, price_df,
            use_coint, use_spread, use_corr,
            coint_pval, adf_alpha, corr_threshold,
            sector_name, date_range, log_transform, nan_method,
            top_n_val, run_button
        )

    def render_admin_section(self):
        st.header("Data Management", 
                  help="Administrator only. Enter the password below.")

        with st.form(key='admin_update_form'):
            password = st.text_input(
                "Enter Admin Password", 
                type="password", 
                key=f"admin_password_{st.session_state['password_key']}",
                help="Enter the administrator password to update data."
            )
            submit_button = st.form_submit_button(label="Update All Data")

            if submit_button:
                try:
                    config = load_config()
                    if password == config.get("admin_password"):
                        st.session_state["admin_authenticated"] = True
                        st.success("Authentication successful. Updating data...")
                        success = preprocess_all(file_name="DATA")
                        if success:
                            st.success("Data preprocessing and saving completed successfully.")
                            st.session_state["admin_authenticated"] = False
                            st.session_state["password_key"] += 1
                        else:
                            st.error("Data preprocessing failed. Check logs for details.")
                    else:
                        st.error("Incorrect password. Access denied.")
                except Exception as e:
                    st.error(f"Error loading configuration: {e}")

    def handle_analysis(
        self,
        name_df,
        price_df,
        use_coint,
        use_spread,
        use_corr,
        coint_pval,
        adf_alpha,
        corr_threshold,
        sector_name,
        date_range,
        log_transform,
        nan_method
    ):
        with st.spinner("Running analysis..."):
            try:
                if not st.session_state["ticker_mapping"]:
                    st.session_state["ticker_mapping"] = dict(zip(name_df['Code'], name_df['Name']))

                data = PairTradeData(
                    sector_name=sector_name,
                    price_data_range=date_range,
                    nan_handling=nan_method,
                    log_transform=log_transform,
                    loader=None
                )
                
                analyzer = PairTradeAnalyzer(
                    data=data,
                    pipeline=None,
                    coint_pvalue=coint_pval,
                    adf_alpha=adf_alpha,
                    corr_threshold=corr_threshold
                )

                final_pairs = analyzer.analyze_pairs(
                    use_coint=use_coint,
                    use_spread=use_spread,
                    use_corr=use_corr
                )

                all_stats = pd.DataFrame(columns=["Filter", "Ticker1", "Ticker2", "Value"])
                ph = analyzer.pipeline_history_df
                for idx, row_data in ph.iterrows():
                    filter_name = row_data["Filter"]
                    stats_dict = row_data["StatsData"]

                    stats_list = []
                    for (t1, t2), v in stats_dict.items():
                        stats_list.append([filter_name, t1, t2, v])
                    
                    df_stats = pd.DataFrame(stats_list, columns=["Filter", "Ticker1", "Ticker2", "Value"])
                    all_stats = pd.concat([all_stats, df_stats], ignore_index=True)

                st.session_state["analysis_done"] = True
                st.session_state["final_pairs"] = final_pairs
                st.session_state["pipeline_history"] = analyzer.pipeline_history_df
                st.session_state["date_filtered_df"] = data.filtered_price
                st.session_state["sector_name"] = sector_name
                st.session_state["coint_pvalues_matrix"] = analyzer.coint_filter.coint_pvalues_matrix
                st.session_state["all_stats"] = all_stats
                st.session_state["spread_dates"] = analyzer.spread_dates

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.session_state["analysis_done"] = False

    def show_results(self, top_n_val):
        final_pairs = st.session_state["final_pairs"]
        sector_name = st.session_state["sector_name"]

        if not final_pairs:
            st.warning(f"No final pairs for sector: {sector_name}")
            return

        st.write(f"### Final Pairs ({sector_name})")
        df_pairs = pd.DataFrame(final_pairs, columns=["Ticker1", "Ticker2"])
        
        pair1_display = (
            df_pairs['Ticker1'].astype(str) + ", " + 
            df_pairs['Ticker1'].map(st.session_state['ticker_mapping']).fillna('Unknown')
        )
        pair2_display = (
            df_pairs['Ticker2'].astype(str) + ", " + 
            df_pairs['Ticker2'].map(st.session_state['ticker_mapping']).fillna('Unknown')
        )

        df_pairs_display = pd.DataFrame({
            "Pair1": pair1_display,
            "Pair2": pair2_display
        })

        st.dataframe(df_pairs_display, use_container_width=True)

        with st.expander("Show Pipeline History"):
            st.write("### Pipeline History")
            all_stats = st.session_state["all_stats"]
            if all_stats.empty:
                st.write("No Pipeline History available.")
            else:
                grouped = all_stats.groupby("Filter")
                for filter_name, group in grouped:
                    st.write(f"**{filter_name}** ({len(group)} pairs)")

                    pair1 = (
                        group['Ticker1'].astype(str) + ", " +
                        group['Ticker1'].map(st.session_state['ticker_mapping']).fillna('Unknown')
                    )
                    pair2 = (
                        group['Ticker2'].astype(str) + ", " +
                        group['Ticker2'].map(st.session_state['ticker_mapping']).fillna('Unknown')
                    )
                    group_display = pd.DataFrame({
                        "Pair1": pair1,
                        "Pair2": pair2,
                        "Value": group['Value']
                    })
                    st.dataframe(group_display, use_container_width=True)

        st.write("### Cointegration Heatmap")
        plotter = GraphPlotter(
            sector_name=sector_name,
            date_filtered_price=st.session_state["date_filtered_df"]
        )
        fig_heatmap = plotter.plot_cointegration_heatmap(st.session_state["coint_pvalues_matrix"])
        if fig_heatmap:
            fig_heatmap.update_layout(width=1200, height=800)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("No data available for Cointegration Heatmap.")

        st.write("### Top Pairs Graph")
        df_date_filtered = st.session_state["date_filtered_df"]
        figures = plotter.plot_top_pairs_individual(final_pairs, top_n=top_n_val)
        if figures:
            for fig, (t1, t2) in figures:
                st.plotly_chart(fig, use_container_width=True)
                
                try:
                    pair_stats = st.session_state["all_stats"][
                        (st.session_state["all_stats"]["Ticker1"] == t1) &
                        (st.session_state["all_stats"]["Ticker2"] == t2)
                    ]

                    if not pair_stats.empty:
                        df_pair_stats = pair_stats.pivot_table(
                            index=["Ticker1", "Ticker2"], 
                            columns="Filter", 
                            values="Value"
                        ).reset_index()

                        pair1 = (
                            df_pair_stats['Ticker1'].astype(str) + ", " + 
                            df_pair_stats['Ticker1'].map(st.session_state['ticker_mapping']).fillna('Unknown')
                        )
                        pair2 = (
                            df_pair_stats['Ticker2'].astype(str) + ", " + 
                            df_pair_stats['Ticker2'].map(st.session_state['ticker_mapping']).fillna('Unknown')
                        )
                        df_pair_stats_display = pd.DataFrame({
                            "Pair1": pair1,
                            "Pair2": pair2,
                            **df_pair_stats.drop(['Ticker1', 'Ticker2'], axis=1)
                        })

                        st.write(f"**Detailed Stats for {t1} & {t2}**")
                        st.dataframe(df_pair_stats_display, use_container_width=True)
                    else:
                        st.write("No Stats found for this pair in all_stats.")
                except Exception as e:
                    st.error(f"Error retrieving stats for pair {t1}-{t2}: {e}")
        else:
            st.warning("No Top Pairs Graph available.")

        st.write("### Select a Pair for Detailed View")
        pair_str_options = [f"{p[0]} - {p[1]}" for p in final_pairs]
        selected_pair_str = st.selectbox("Select a pair", pair_str_options)
        if selected_pair_str:
            t1, t2 = selected_pair_str.split(" - ")
            st.write(f"**Selected Pair**: {t1} vs {t2}")

            if (t1 in df_date_filtered.columns) and (t2 in df_date_filtered.columns):
                s1 = df_date_filtered[t1].dropna()
                s2 = df_date_filtered[t2].dropna()
                s1 = pd.to_numeric(s1, errors='coerce').dropna()
                s2 = pd.to_numeric(s2, errors='coerce').dropna()
                common_index = s1.index.intersection(s2.index)

                st.write("#### Price Time Series")
                fig_price = plotter.plot_pairs(s1, s2, t1, t2)
                if fig_price:
                    st.plotly_chart(fig_price, use_container_width=True)
                else:
                    st.warning("Price data not available for plotting.")

                try:
                    s1_log = s1.loc[common_index].apply(np.log)
                    s2_log = s2.loc[common_index].apply(np.log)
            
                    spread_analyzer = SpreadAnalyzer(x=s1_log, y=s2_log)
                    st.write("#### Spread Time Series (Log Prices)")
                    spread_fig = spread_analyzer.plot_spread()
                    st.plotly_chart(spread_fig, use_container_width=True)

                    st.write("#### Spread Statistics")
                    spread_stats = spread_analyzer.get_spread_stats()
                    spread_stats_flat = {k: v for k, v in spread_stats.items() if not isinstance(v, dict)}
                    spread_stats_df = pd.DataFrame([spread_stats_flat])
                    st.dataframe(spread_stats_df, use_container_width=True)
                    
                    pair_key = f"{t1}-{t2}"
                    spread_dates = st.session_state["spread_dates"].get(pair_key)
                    if spread_dates is not None and not spread_dates.empty:
                        st.write("#### Spread Out-of-Std Dates")
                        spread_dates_display = spread_dates.reset_index().rename(columns={
                            'index': 'Date',
                            'Above_1_std': 'Above 1σ',
                            'Below_1_std': 'Below 1σ',
                            'Above_2_std': 'Above 2σ',
                            'Below_2_std': 'Below 2σ'
                        })
                        spread_dates_display['Date'] = spread_dates_display['Date'].dt.strftime('%Y-%m-%d')
                        st.dataframe(spread_dates_display, use_container_width=True)
                    else:
                        st.write("No Spread Out-of-Std dates found for this pair.")
                except ValueError as ve:
                    st.error(f"Spread analysis failed: {ve}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during spread analysis: {e}")
            else:
                st.write("Data not found for that pair.")


def main():
    app = PairTradingApp()
    app.run()
