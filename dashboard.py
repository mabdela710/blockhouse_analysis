import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
from scipy import stats

# Set up paths
data_dir = './data'
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

st.set_page_config(page_title="Temporary Impact Dashboard", layout="wide")
st.title("📊 Temporary Impact Analysis & Optimal Allocation Dashboard")
st.markdown("""
This dashboard allows you to:
- Explore order book data for FROG, SOUN, CRWV
- Visualize temporary impact models
- See optimal order allocation and recommendations
""")

# Helper functions (from your script, simplified for dashboard)
def load_data(ticker, date):
    file_path = os.path.join(data_dir, f"{ticker}/{ticker}_{date} 00:00:00+00:00.csv")
    df = pd.read_csv(file_path)
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df['mid_price'] = (df['bid_px_00'] + df['ask_px_00']) / 2
    buy_market_orders = df[(df['action'] == 'T') & (df['rtype'] == 10) & (df['side'] == 'B')]
    df.loc[buy_market_orders.index, 'slippage'] = df.loc[buy_market_orders.index, 'price'] - df.loc[buy_market_orders.index, 'mid_price']
    sell_market_orders = df[(df['action'] == 'T') & (df['rtype'] == 10) & (df['side'] == 'A')]
    df.loc[sell_market_orders.index, 'slippage'] = df.loc[sell_market_orders.index, 'mid_price'] - df.loc[sell_market_orders.index, 'price']
    return df

def power_law_model(x, alpha, beta):
    return alpha * np.power(x, beta)

def fit_power_law_model(sizes, slippages):
    mask = (sizes > 0) & (slippages > 0)
    if sum(mask) < 2:
        return None, None
    sizes_filtered = sizes[mask]
    slippages_filtered = slippages[mask]
    log_sizes = np.log(sizes_filtered)
    log_slippages = np.log(slippages_filtered)
    slope, intercept, *_ = stats.linregress(log_sizes, log_slippages)
    beta = slope
    alpha = np.exp(intercept)
    return alpha, beta

def analyze_temporary_impact(df):
    market_orders = df[(df['action'] == 'T') & (df['rtype'] == 10) & (df['slippage'].notna())]
    if market_orders.empty:
        return None, None, None, None
    sizes = market_orders['size'].values
    slippages = np.abs(market_orders['slippage'].values)
    alpha, beta = fit_power_law_model(sizes, slippages)
    return sizes, slippages, alpha, beta

def plot_impact(sizes, slippages, alpha, beta):
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(sizes, slippages, alpha=0.5, label='Data')
    if alpha is not None and beta is not None:
        x_range = np.linspace(min(sizes), max(sizes), 100)
        y_model = power_law_model(x_range, alpha, beta)
        ax.plot(x_range, y_model, 'r-', label=f'Power-law: α={alpha:.3g}, β={beta:.2f}')
        linear_alpha = np.mean(slippages) / np.mean(sizes)
        ax.plot(x_range, linear_alpha * x_range, 'g--', label=f'Linear: α={linear_alpha:.3g}')
    ax.set_xlabel('Order Size (X)')
    ax.set_ylabel('Slippage (g(X))')
    ax.set_title('Temporary Impact Function')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def show_recommendations(alpha, beta):
    st.subheader('Recommendations')
    if alpha is None or beta is None:
        st.warning('Not enough data to fit a model.')
        return
    if abs(beta-1) < 0.1:
        st.info('The linear model fits well. You can use a simple linear impact model for execution.')
    else:
        st.info('The nonlinear (power-law) model fits better. Consider using nonlinear impact in your execution strategy.')
    st.markdown(f"**Estimated Model:** $g_t(x) = {alpha:.3g} \, x^{{{beta:.2f}}}$")

def main():
    tickers = ['FROG', 'SOUN', 'CRWV']
    date = st.sidebar.text_input('Date (YYYY-MM-DD)', '2025-04-03')
    ticker = st.sidebar.selectbox('Select Ticker', tickers)
    st.sidebar.markdown('---')
    st.sidebar.markdown('Upload your own CSV if you want:')
    uploaded = st.sidebar.file_uploader('Upload CSV', type='csv')
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success('Custom data loaded!')
    else:
        try:
            df = load_data(ticker, date)
        except Exception as e:
            st.error(f'Error loading data: {e}')
            return
    st.header(f"Data Preview: {ticker}")
    st.dataframe(df.head(20))
    st.subheader('Summary Statistics')
    st.write(df.describe())
    sizes, slippages, alpha, beta = analyze_temporary_impact(df)
    if sizes is not None:
        st.subheader('Temporary Impact Model')
        plot_impact(sizes, slippages, alpha, beta)
        show_recommendations(alpha, beta)
    else:
        st.warning('No valid market orders with slippage found.')

if __name__ == '__main__':
    main()
