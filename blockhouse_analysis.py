"""
Blockhouse Work Trial Task - Temporary Impact Analysis and Optimal Order Allocation
Author: [Your Name]
Date: July 28, 2025

This script implements:
1. Analysis of order book data to understand temporary impact
2. Modeling of the temporary impact function g_t(X)
3. Formulation of an optimal order allocation algorithm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
from datetime import datetime
import seaborn as sns
from scipy import stats

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Define paths
DATA_DIR = './data'  # Update this to your data directory
OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Part 1: Data Loading and Preprocessing
def load_data(ticker, date):
    """
    Load order book data for a specific ticker and date.
    
    Parameters:
    ticker (str): Ticker symbol (e.g., 'FROG', 'SOUN', 'CRWV')
    date (str): Date in format 'YYYY-MM-DD'
    
    Returns:
    pd.DataFrame: Loaded and preprocessed data
    """
    file_path = os.path.join(DATA_DIR, f"{ticker}/{ticker}_{date} 00:00:00+00:00.csv")
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    
    # Calculate mid-price
    df['mid_price'] = (df['bid_px_00'] + df['ask_px_00']) / 2
    
    # Calculate slippage for market orders
    # Buy market orders: slippage = executed_price - mid_price
    buy_market_orders = df[(df['action'] == 'T') & (df['rtype'] == 10) & (df['side'] == 'B')]
    df.loc[buy_market_orders.index, 'slippage'] = df.loc[buy_market_orders.index, 'price'] - df.loc[buy_market_orders.index, 'mid_price']
    
    # Sell market orders: slippage = mid_price - executed_price
    sell_market_orders = df[(df['action'] == 'T') & (df['rtype'] == 10) & (df['side'] == 'A')]
    df.loc[sell_market_orders.index, 'slippage'] = df.loc[sell_market_orders.index, 'mid_price'] - df.loc[sell_market_orders.index, 'price']
    
    return df

def calculate_order_book_metrics(df):
    """
    Calculate various order book metrics.
    
    Parameters:
    df (pd.DataFrame): Order book data
    
    Returns:
    pd.DataFrame: Data with additional metrics
    """
    # Calculate spread
    df['spread'] = df['ask_px_00'] - df['bid_px_00']
    
    # Calculate order book imbalance
    df['bid_volume'] = df['bid_sz_00'] + df['bid_sz_01'] + df['bid_sz_02'] + df['bid_sz_03'] + df['bid_sz_04']
    df['ask_volume'] = df['ask_sz_00'] + df['ask_sz_01'] + df['ask_sz_02'] + df['ask_sz_03'] + df['ask_sz_04']
    df['imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
    
    # Calculate volatility (rolling 5-minute standard deviation of mid-price returns)
    df['mid_price_return'] = df['mid_price'].pct_change()
    df['volatility'] = df['mid_price_return'].rolling(window=300).std().fillna(0)
    
    return df

def resample_to_time_periods(df, period='5min'):
    """
    Resample data to specified time periods.
    
    Parameters:
    df (pd.DataFrame): Order book data
    period (str): Resampling period (e.g., '5min', '1min')
    
    Returns:
    pd.DataFrame: Resampled data
    """
    # Group by time period
    df_resampled = df.set_index('ts_event').resample(period).agg({
        'mid_price': 'mean',
        'spread': 'mean',
        'imbalance': 'mean',
        'volatility': 'mean',
        'slippage': ['mean', 'std', 'count'],
        'size': ['mean', 'sum', 'count']
    })
    
    # Flatten multi-level columns
    df_resampled.columns = ['_'.join(col).strip() for col in df_resampled.columns.values]
    
    return df_resampled

# Part 2: Temporary Impact Modeling
def power_law_model(x, alpha, beta):
    """
    Power-law model for temporary impact.
    
    Parameters:
    x (float or array): Order size
    alpha (float): Scale parameter
    beta (float): Power-law exponent
    
    Returns:
    float or array: Temporary impact
    """
    return alpha * np.power(x, beta)

def fit_power_law_model(sizes, slippages):
    """
    Fit power-law model to slippage data.
    
    Parameters:
    sizes (array): Order sizes
    slippages (array): Corresponding slippages
    
    Returns:
    tuple: (alpha, beta) parameters
    """
    # Filter out non-positive values for log transformation
    mask = (sizes > 0) & (slippages > 0)
    if sum(mask) < 2:
        return None, None
    
    sizes_filtered = sizes[mask]
    slippages_filtered = slippages[mask]
    
    # Log transformation for linear regression
    log_sizes = np.log(sizes_filtered)
    log_slippages = np.log(slippages_filtered)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_slippages)
    
    # Convert back to power-law parameters
    beta = slope
    alpha = np.exp(intercept)
    
    return alpha, beta

def analyze_temporary_impact(df, ticker):
    """
    Analyze and visualize temporary impact for a ticker.
    
    Parameters:
    df (pd.DataFrame): Order book data
    ticker (str): Ticker symbol
    
    Returns:
    tuple: (alpha, beta) parameters
    """
    # Filter market orders with valid slippage
    market_orders = df[(df['action'] == 'T') & (df['rtype'] == 10) & (df['slippage'].notna())]
    
    if market_orders.empty:
        print(f"No market orders with valid slippage found for {ticker}")
        return None, None
    
    # Fit power-law model
    sizes = market_orders['size'].values
    slippages = market_orders['slippage'].abs().values  # Use absolute slippage
    
    alpha, beta = fit_power_law_model(sizes, slippages)
    
    if alpha is None or beta is None:
        print(f"Could not fit power-law model for {ticker}")
        return None, None
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.scatter(sizes, slippages, alpha=0.5, label='Data')
    
    # Generate model curve
    x_range = np.linspace(min(sizes), max(sizes), 100)
    y_model = power_law_model(x_range, alpha, beta)
    plt.plot(x_range, y_model, 'r-', label=f'Power-law model: α={alpha:.6f}, β={beta:.2f}')
    
    # Generate linear model for comparison
    linear_alpha = np.mean(slippages) / np.mean(sizes)
    y_linear = linear_alpha * x_range
    plt.plot(x_range, y_linear, 'g--', label=f'Linear model: α={linear_alpha:.6f}')
    
    plt.title(f'Temporary Impact Function for {ticker}')
    plt.xlabel('Order Size (X)')
    plt.ylabel('Slippage (g(X))')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(OUTPUT_DIR, f'{ticker}_temporary_impact.png'))
    
    return alpha, beta

# Part 3: Optimal Order Allocation
def objective_function(x, alphas, betas, sigmas):
    """
    Objective function for optimization: sum of temporary impact costs.
    
    Parameters:
    x (array): Allocation vector
    alphas (array): Alpha parameters for each period
    betas (array): Beta parameters for each period
    sigmas (array): Volatility parameters for each period
    
    Returns:
    float: Total temporary impact cost
    """
    total_cost = 0
    for i in range(len(x)):
        if x[i] > 0:  # Avoid issues with power function when x=0
            total_cost += alphas[i] * np.power(x[i], betas[i]) * sigmas[i]
    return total_cost

def constraint_sum(x, total_size):
    """
    Constraint: sum of allocations equals total order size.
    
    Parameters:
    x (array): Allocation vector
    total_size (float): Total order size
    
    Returns:
    float: Constraint value (should be 0)
    """
    return sum(x) - total_size

def optimal_allocation(total_size, alphas, betas, sigmas):
    """
    Calculate optimal order allocation.
    
    Parameters:
    total_size (float): Total order size
    alphas (array): Alpha parameters for each period
    betas (array): Beta parameters for each period
    sigmas (array): Volatility parameters for each period
    
    Returns:
    array: Optimal allocation vector
    """
    n_periods = len(alphas)
    
    # Initial guess: equal allocation
    x0 = np.ones(n_periods) * total_size / n_periods
    
    # Bounds: non-negative allocations
    bounds = [(0, None) for _ in range(n_periods)]
    
    # Constraint: sum of allocations equals total order size
    constraint = {'type': 'eq', 'fun': constraint_sum, 'args': (total_size,)}
    
    # Solve optimization problem
    result = minimize(
        objective_function,
        x0,
        args=(alphas, betas, sigmas),
        bounds=bounds,
        constraints=constraint,
        method='SLSQP'
    )
    
    if not result.success:
        print(f"Optimization failed: {result.message}")
        return None
    
    return result.x

def visualize_allocation(allocation, alphas, sigmas, ticker):
    """
    Visualize optimal allocation.
    
    Parameters:
    allocation (array): Optimal allocation vector
    alphas (array): Alpha parameters for each period
    sigmas (array): Volatility parameters for each period
    ticker (str): Ticker symbol
    """
    # Calculate impact parameters
    impact_params = [alpha * sigma for alpha, sigma in zip(alphas, sigmas)]
    
    # Create DataFrame for visualization
    df_alloc = pd.DataFrame({
        'Period': range(1, len(allocation) + 1),
        'Impact Parameter': impact_params,
        'Allocation': allocation,
        'Percentage': allocation / sum(allocation) * 100
    })
    
    # Sort by impact parameter
    df_alloc = df_alloc.sort_values('Impact Parameter')
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Impact parameters
    ax1.bar(df_alloc['Period'], df_alloc['Impact Parameter'])
    ax1.set_title('Impact Parameters by Period')
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Impact Parameter (α × σ)')
    ax1.grid(True, alpha=0.3)
    
    # Allocation
    ax2.bar(df_alloc['Period'], df_alloc['Allocation'])
    ax2.set_title('Optimal Allocation by Period')
    ax2.set_xlabel('Period')
    ax2.set_ylabel('Allocation (shares)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{ticker}_optimal_allocation.png'))

# Part 4: Main Analysis
def main():
    """
    Main analysis function.
    """
    # Define tickers and dates
    tickers = ['FROG', 'SOUN', 'CRWV']
    date = '2025-04-03'  # Example date
    
    # Store results
    results = {}
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        
        # Load and preprocess data
        try:
            df = load_data(ticker, date)
            df = calculate_order_book_metrics(df)
            
            # Analyze temporary impact
            alpha, beta = analyze_temporary_impact(df, ticker)
            
            if alpha is not None and beta is not None:
                # Resample to 5-minute periods
                df_resampled = resample_to_time_periods(df, period='5min')
                
                # Extract parameters for each period
                alphas = []
                betas = []
                sigmas = []
                
                for _, row in df_resampled.iterrows():
                    # Use fitted beta for all periods
                    alphas.append(alpha)
                    betas.append(beta)
                    sigmas.append(row['volatility_mean'])
                
                # Ensure we have at least 3 periods for demonstration
                if len(alphas) < 3:
                    alphas = alphas * (3 // len(alphas) + 1)
                    betas = betas * (3 // len(betas) + 1)
                    sigmas = sigmas * (3 // len(sigmas) + 1)
                    alphas = alphas[:3]
                    betas = betas[:3]
                    sigmas = sigmas[:3]
                
                # Calculate optimal allocation
                total_size = 1000  # Example total order size
                allocation = optimal_allocation(total_size, alphas, betas, sigmas)
                
                if allocation is not None:
                    # Visualize allocation
                    visualize_allocation(allocation, alphas, sigmas, ticker)
                    
                    # Store results
                    results[ticker] = {
                        'alpha': alpha,
                        'beta': beta,
                        'alphas': alphas[:3],  # Store first 3 periods for simplicity
                        'sigmas': sigmas[:3],
                        'allocation': allocation[:3]
                    }
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    # Print summary
    print("\nSummary of Results:")
    print("===================")
    
    for ticker, result in results.items():
        print(f"\n{ticker}:")
        print(f"  Power-law exponent (β): {result['beta']:.4f}")
        print("  Optimal allocation:")
        for i, alloc in enumerate(result['allocation']):
            print(f"    Period {i+1}: {alloc:.2f} shares ({alloc/total_size*100:.1f}%)")
    
    # Create numerical example visualization
    if 'FROG' in results:
        create_numerical_example(results['FROG'], total_size)

def create_numerical_example(result, total_size):
    """
    Create visualization for numerical example.
    
    Parameters:
    result (dict): Results for a ticker
    total_size (float): Total order size
    """
    alphas = result['alphas'][:3]
    beta = result['beta']
    sigmas = result['sigmas'][:3]
    allocation = result['allocation'][:3]
    
    # Calculate impact parameters
    impact_params = [alpha * sigma for alpha, sigma in zip(alphas, sigmas)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Period': [1, 2, 3],
        'Impact Parameter': impact_params,
        'Allocation': allocation,
        'Percentage': allocation / sum(allocation) * 100
    })
    
    # Visualize
    plt.figure(figsize=(10, 6))
    
    # Create table
    table_data = [
        ['Period', 'Impact Parameter', 'Allocation', 'Percentage'],
        [1, f"{impact_params[0]:.1e}", f"{allocation[0]:.0f}", f"{allocation[0]/total_size*100:.1f}%"],
        [2, f"{impact_params[1]:.1e}", f"{allocation[1]:.0f}", f"{allocation[1]/total_size*100:.1f}%"],
        [3, f"{impact_params[2]:.1e}", f"{allocation[2]:.0f}", f"{allocation[2]/total_size*100:.1f}%"]
    ]
    
    table = plt.table(cellText=table_data[1:], colLabels=table_data[0], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.axis('off')
    plt.title('Numerical Example: Optimal Allocation', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'numerical_example.png'))

if __name__ == "__main__":
    main()

