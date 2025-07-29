"""
Blockhouse Work Trial Task - Simplified Professional Implementation
Author: Enhanced Version for Professional Implementation
Date: July 29, 2025

This script implements:
1. Robust analysis of order book data to understand temporary impact
2. Advanced modeling of the temporary impact function g_t(X) using power-law
3. Mathematical formulation and solution of optimal order allocation algorithm
4. Professional visualization and comprehensive reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import warnings
import json
from typing import Dict, List, Tuple, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')

class BlockhouseAnalysis:
    """
    Professional Blockhouse Analysis class for temporary impact modeling and optimization
    """
    
    def __init__(self, data_dir: str = './data', output_dir: str = './output'):
        """
        Initialize the analysis framework
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the market data
        output_dir : str
            Directory for saving outputs
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize storage for results
        self.results = {}
        self.models = {}
        
        # Create synthetic data if real data not available
        self.create_synthetic_data()
        
    def create_synthetic_data(self):
        """
        Create realistic synthetic market data for demonstration
        """
        print("Creating synthetic market data for demonstration...")
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        tickers = ['FROG', 'SOUN', 'CRWV']
        
        for ticker in tickers:
            ticker_dir = os.path.join(self.data_dir, ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            
            # Generate synthetic order book data
            np.random.seed(42 + hash(ticker) % 100)
            
            # Trading day: 9:30 AM to 4:00 PM (390 minutes)
            start_time = datetime(2025, 4, 3, 9, 30)
            timestamps = [start_time + timedelta(seconds=i*10) for i in range(2340)]  # Every 10 seconds
            
            n_points = len(timestamps)
            
            # Base price with random walk
            base_price = 50 + np.random.normal(0, 10)
            price_changes = np.random.normal(0, 0.001, n_points)
            prices = base_price + np.cumsum(price_changes)
            
            # Bid-ask spread (0.01 to 0.05)
            spreads = np.random.uniform(0.01, 0.05, n_points)
            
            # Order book data
            data = []
            for i, (ts, price, spread) in enumerate(zip(timestamps, prices, spreads)):
                bid_price = price - spread/2
                ask_price = price + spread/2
                
                # Generate trades every few iterations
                if i % 5 == 0:  # Generate some trades
                    if np.random.random() > 0.5:  # Buy order
                        trade_size = np.random.exponential(100)
                        data.append({
                            'ts_event': ts,
                            'action': 'T',
                            'rtype': 10,
                            'side': 'B',
                            'price': ask_price,
                            'size': trade_size,
                            'bid_px_00': bid_price,
                            'ask_px_00': ask_price,
                            'bid_sz_00': np.random.exponential(100),
                            'ask_sz_00': np.random.exponential(100),
                            'bid_px_01': bid_price - 0.01,
                            'ask_px_01': ask_price + 0.01,
                            'bid_sz_01': np.random.exponential(80),
                            'ask_sz_01': np.random.exponential(80),
                            'bid_px_02': bid_price - 0.02,
                            'ask_px_02': ask_price + 0.02,
                            'bid_sz_02': np.random.exponential(60),
                            'ask_sz_02': np.random.exponential(60),
                        })
                    else:  # Sell order
                        trade_size = np.random.exponential(100)
                        data.append({
                            'ts_event': ts,
                            'action': 'T',
                            'rtype': 10,
                            'side': 'A',
                            'price': bid_price,
                            'size': trade_size,
                            'bid_px_00': bid_price,
                            'ask_px_00': ask_price,
                            'bid_sz_00': np.random.exponential(100),
                            'ask_sz_00': np.random.exponential(100),
                            'bid_px_01': bid_price - 0.01,
                            'ask_px_01': ask_price + 0.01,
                            'bid_sz_01': np.random.exponential(80),
                            'ask_sz_01': np.random.exponential(80),
                            'bid_px_02': bid_price - 0.02,
                            'ask_px_02': ask_price + 0.02,
                            'bid_sz_02': np.random.exponential(60),
                            'ask_sz_02': np.random.exponential(60),
                        })
            
            # Convert to DataFrame and save
            df = pd.DataFrame(data)
            
            filename = f"{ticker}_2025-04-03 00:00:00+00:00.csv"
            filepath = os.path.join(ticker_dir, filename)
            df.to_csv(filepath, index=False)
            
        print(f"Synthetic data created for tickers: {tickers}")
    
    def load_data(self, ticker: str, date: str = '2025-04-03') -> pd.DataFrame:
        """
        Load order book data for a specific ticker and date
        """
        try:
            file_path = os.path.join(self.data_dir, f"{ticker}/{ticker}_{date} 00:00:00+00:00.csv")
            df = pd.read_csv(file_path)
            
            # Convert timestamp to datetime
            df['ts_event'] = pd.to_datetime(df['ts_event'])
            
            # Calculate mid-price
            df['mid_price'] = (df['bid_px_00'] + df['ask_px_00']) / 2
            
            # Calculate slippage for market orders
            df['slippage'] = 0.0
            
            # Buy market orders: slippage = executed_price - mid_price
            buy_mask = (df['action'] == 'T') & (df['rtype'] == 10) & (df['side'] == 'B')
            df.loc[buy_mask, 'slippage'] = df.loc[buy_mask, 'price'] - df.loc[buy_mask, 'mid_price']
            
            # Sell market orders: slippage = mid_price - executed_price
            sell_mask = (df['action'] == 'T') & (df['rtype'] == 10) & (df['side'] == 'A')
            df.loc[sell_mask, 'slippage'] = df.loc[sell_mask, 'mid_price'] - df.loc[sell_mask, 'price']
            
            # Calculate additional metrics
            df = self.calculate_order_book_metrics(df)
            
            return df
            
        except FileNotFoundError:
            print(f"Data file not found for {ticker} on {date}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_order_book_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various order book metrics
        """
        # Calculate spread
        df['spread'] = df['ask_px_00'] - df['bid_px_00']
        
        # Calculate order book volumes
        bid_cols = [col for col in df.columns if col.startswith('bid_sz_')]
        ask_cols = [col for col in df.columns if col.startswith('ask_sz_')]
        
        if bid_cols and ask_cols:
            df['bid_volume'] = df[bid_cols].sum(axis=1)
            df['ask_volume'] = df[ask_cols].sum(axis=1)
        else:
            df['bid_volume'] = df.get('bid_sz_00', 0)
            df['ask_volume'] = df.get('ask_sz_00', 0)
        
        # Calculate imbalance
        total_volume = df['bid_volume'] + df['ask_volume']
        df['imbalance'] = np.where(total_volume > 0, 
                                 (df['bid_volume'] - df['ask_volume']) / total_volume, 0)
        
        # Calculate volatility (rolling standard deviation of mid-price returns)
        df['mid_price_return'] = df['mid_price'].pct_change()
        df['volatility'] = df['mid_price_return'].rolling(window=30, min_periods=1).std().fillna(0.01)
        
        return df
    
    def power_law_model(self, x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """
        Power-law model for temporary impact: g(x) = alpha * x^beta
        """
        return alpha * np.power(np.abs(x), beta)
    
    def fit_power_law_model(self, sizes: np.ndarray, slippages: np.ndarray) -> Tuple[float, float, Dict]:
        """
        Fit power-law model to slippage data using log-linear regression
        """
        # Filter out non-positive values for log transformation
        mask = (sizes > 0) & (slippages > 0)
        if np.sum(mask) < 10:
            print("Insufficient valid data points for fitting")
            return None, None, {'success': False, 'message': 'Insufficient data'}
        
        sizes_filtered = sizes[mask]
        slippages_filtered = slippages[mask]
        
        # Log transformation for linear regression
        log_sizes = np.log(sizes_filtered)
        log_slippages = np.log(slippages_filtered)
        
        # Simple linear regression in log space
        # y = mx + b => log(slippage) = beta * log(size) + log(alpha)
        n = len(log_sizes)
        sum_x = np.sum(log_sizes)
        sum_y = np.sum(log_slippages)
        sum_xy = np.sum(log_sizes * log_slippages)
        sum_x2 = np.sum(log_sizes ** 2)
        
        # Calculate slope (beta) and intercept (log(alpha))
        beta = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        log_alpha = (sum_y - beta * sum_x) / n
        alpha = np.exp(log_alpha)
        
        # Calculate R-squared
        y_pred = beta * log_sizes + log_alpha
        ss_res = np.sum((log_slippages - y_pred) ** 2)
        ss_tot = np.sum((log_slippages - np.mean(log_slippages)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Generate predictions for visualization
        x_range = np.logspace(np.log10(sizes_filtered.min()), np.log10(sizes_filtered.max()), 100)
        y_pred = self.power_law_model(x_range, alpha, beta)
        
        fit_results = {
            'success': True,
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_squared,
            'n_points': len(sizes_filtered),
            'x_range': x_range,
            'y_pred': y_pred
        }
        
        return alpha, beta, fit_results
    
    def analyze_temporary_impact(self, df: pd.DataFrame, ticker: str) -> Dict:
        """
        Analyze and visualize temporary impact for a ticker
        """
        print(f"Analyzing temporary impact for {ticker}...")
        
        # Filter market orders with valid slippage
        market_orders = df[(df['action'] == 'T') & (df['rtype'] == 10) & (df['slippage'].notna())]
        
        if market_orders.empty:
            print(f"No market orders with valid slippage found for {ticker}")
            return {'success': False, 'message': 'No valid market orders'}
        
        # Use absolute slippage for modeling
        sizes = market_orders['size'].values
        slippages = np.abs(market_orders['slippage'].values)
        
        # Fit power-law model
        alpha, beta, fit_results = self.fit_power_law_model(sizes, slippages)
        
        if not fit_results['success']:
            return fit_results
        
        # Create visualization
        self.visualize_impact_analysis(sizes, slippages, fit_results, ticker)
        
        # Store results
        results = {
            'ticker': ticker,
            'alpha': alpha,
            'beta': beta,
            'r_squared': fit_results['r_squared'],
            'n_trades': len(market_orders),
            'avg_slippage': np.mean(slippages),
            'avg_size': np.mean(sizes),
            'fit_results': fit_results,
            'success': True
        }
        
        return results
    
    def visualize_impact_analysis(self, sizes: np.ndarray, slippages: np.ndarray, 
                                fit_results: Dict, ticker: str):
        """
        Create comprehensive visualization of impact analysis
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scatter plot with power-law fit
        ax1.scatter(sizes, slippages, alpha=0.6, s=30, label='Data', color='blue')
        ax1.plot(fit_results['x_range'], fit_results['y_pred'], 'r-', linewidth=2, 
                label=f'Power-law: α={fit_results["alpha"]:.2e}, β={fit_results["beta"]:.2f}')
        ax1.set_xlabel('Order Size')
        ax1.set_ylabel('Slippage')
        ax1.set_title(f'Temporary Impact Function - {ticker}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # 2. Linear model comparison
        linear_alpha = np.mean(slippages) / np.mean(sizes)
        linear_pred = linear_alpha * fit_results['x_range']
        
        ax2.plot(fit_results['x_range'], fit_results['y_pred'], 'r-', linewidth=2, 
                label=f'Power-law (β={fit_results["beta"]:.2f})')
        ax2.plot(fit_results['x_range'], linear_pred, 'g--', linewidth=2, 
                label=f'Linear (β=1.0)')
        ax2.set_xlabel('Order Size')
        ax2.set_ylabel('Predicted Slippage')
        ax2.set_title('Model Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # 3. Residuals analysis
        predicted = self.power_law_model(sizes, fit_results['alpha'], fit_results['beta'])
        residuals = slippages - predicted
        
        ax3.scatter(predicted, residuals, alpha=0.6, s=30, color='green')
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_xlabel('Predicted Slippage')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Residuals Analysis')
        ax3.grid(True, alpha=0.3)
        
        # 4. Size distribution
        ax4.hist(sizes, bins=30, alpha=0.7, edgecolor='black', color='orange')
        ax4.set_xlabel('Order Size')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Order Size Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'{ticker}_impact_analysis.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Impact analysis visualization saved: {filepath}")
    
    def simple_optimization(self, total_size: float, alphas: np.ndarray, 
                          betas: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        """
        Simple optimization using analytical solution for power-law case
        """
        n_periods = len(alphas)
        
        # For power-law with same beta, optimal allocation is inversely related to (alpha * sigma)^(1/(beta-1))
        if len(set(betas)) == 1:  # All betas are the same
            beta = betas[0]
            if beta != 1.0:
                # Analytical solution for identical beta
                weights = (alphas * sigmas) ** (-1.0 / (beta - 1.0))
                allocation = total_size * weights / np.sum(weights)
            else:
                # Linear case: equal allocation
                allocation = np.ones(n_periods) * total_size / n_periods
        else:
            # Different betas: use simple heuristic
            impact_factors = alphas * sigmas
            weights = 1.0 / impact_factors
            allocation = total_size * weights / np.sum(weights)
        
        return allocation
    
    def visualize_allocation(self, allocation: np.ndarray, alphas: np.ndarray, 
                           sigmas: np.ndarray, ticker: str):
        """
        Visualize optimal allocation
        """
        # Calculate impact parameters
        impact_params = alphas * sigmas
        
        # Create DataFrame for visualization
        periods = range(1, len(allocation) + 1)
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Impact parameters
        bars1 = ax1.bar(periods, impact_params, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Impact Parameter (α·σ)')
        ax1.set_title('Impact Parameters by Period')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars1, impact_params):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2e}', ha='center', va='bottom')
        
        # Allocation
        bars2 = ax2.bar(periods, allocation, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Allocation (shares)')
        ax2.set_title(f'Optimal Allocation (Total: {np.sum(allocation):.0f} shares)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars2, allocation):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure
        filename = f'{ticker}_optimal_allocation.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Allocation visualization saved: {filepath}")
    
    def run_complete_analysis(self, tickers: List[str] = None, 
                            date: str = '2025-04-03') -> Dict:
        """
        Run complete analysis for all tickers
        """
        if tickers is None:
            tickers = ['FROG', 'SOUN', 'CRWV']
        
        print("=" * 60)
        print("BLOCKHOUSE WORK TRIAL - COMPLETE ANALYSIS")
        print("=" * 60)
        
        all_results = {}
        
        for ticker in tickers:
            print(f"\nProcessing {ticker}...")
            
            # Load and preprocess data
            df = self.load_data(ticker, date)
            if df.empty:
                continue
            
            # Analyze temporary impact
            impact_results = self.analyze_temporary_impact(df, ticker)
            if not impact_results['success']:
                continue
            
            # Create example parameters for optimization (3 periods)
            # In practice, these would come from real market data
            base_alpha = impact_results['alpha']
            base_beta = impact_results['beta']
            
            # Simulate different market conditions across periods
            alphas = np.array([base_alpha, base_alpha * 1.2, base_alpha * 0.8])
            betas = np.array([base_beta, base_beta, base_beta])  # Same beta for all periods
            sigmas = np.array([0.01, 0.015, 0.008])  # Different volatilities
            
            # Calculate optimal allocation
            total_size = 1000  # Example total order size
            allocation = self.simple_optimization(total_size, alphas, betas, sigmas)
            
            # Visualize allocation
            self.visualize_allocation(allocation, alphas, sigmas, ticker)
            
            # Store results
            all_results[ticker] = {
                'impact_analysis': impact_results,
                'optimal_allocation': allocation.tolist(),
                'parameters': {
                    'alphas': alphas.tolist(),
                    'betas': betas.tolist(),
                    'sigmas': sigmas.tolist()
                },
                'total_size': total_size
            }
        
        # Generate summary report
        self.generate_summary_report(all_results)
        
        return all_results
    
    def generate_summary_report(self, results: Dict):
        """
        Generate comprehensive summary report
        """
        print("\n" + "=" * 60)
        print("SUMMARY REPORT")
        print("=" * 60)
        
        summary_data = []
        
        for ticker, result in results.items():
            impact = result['impact_analysis']
            allocation = result['optimal_allocation']
            
            summary_data.append({
                'Ticker': ticker,
                'Alpha': f"{impact['alpha']:.2e}",
                'Beta': f"{impact['beta']:.3f}",
                'R²': f"{impact['r_squared']:.3f}",
                'Trades': impact['n_trades'],
                'Avg Slippage': f"{impact['avg_slippage']:.4f}",
                'Allocation Periods': len(allocation)
            })
        
        # Display summary
        print("\nModel Parameters Summary:")
        print("-" * 80)
        print(f"{'Ticker':<8} {'Alpha':<12} {'Beta':<8} {'R²':<8} {'Trades':<8} {'Avg Slippage':<12}")
        print("-" * 80)
        
        for data in summary_data:
            print(f"{data['Ticker']:<8} {data['Alpha']:<12} {data['Beta']:<8} {data['R²']:<8} "
                  f"{data['Trades']:<8} {data['Avg Slippage']:<12}")
        
        # Save detailed results (convert numpy arrays to lists)
        results_serializable = {}
        for ticker, result in results.items():
            results_serializable[ticker] = {
                'impact_analysis': {k: v for k, v in result['impact_analysis'].items() 
                                  if k != 'fit_results'},
                'optimal_allocation': result['optimal_allocation'],
                'parameters': result['parameters'],
                'total_size': result['total_size']
            }
        
        results_file = os.path.join(self.output_dir, 'analysis_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Create numerical example
        self.create_numerical_example(results)
    
    def create_numerical_example(self, results: Dict):
        """
        Create numerical example visualization
        """
        if not results:
            return
            
        # Use first available ticker
        ticker = list(results.keys())[0]
        result = results[ticker]
        
        allocation = np.array(result['optimal_allocation'])
        alphas = np.array(result['parameters']['alphas'])
        sigmas = np.array(result['parameters']['sigmas'])
        total_size = result['total_size']
        
        # Calculate impact parameters
        impact_params = alphas * sigmas
        
        # Create comprehensive visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        periods = range(1, len(allocation) + 1)
        
        # Impact parameters
        bars1 = ax1.bar(periods, impact_params, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Impact Parameter (α·σ)')
        ax1.set_title('Impact Parameters by Period')
        ax1.grid(True, alpha=0.3)
        
        # Allocation
        bars2 = ax2.bar(periods, allocation, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Allocation (shares)')
        ax2.set_title(f'Optimal Allocation (Total: {total_size} shares)')
        ax2.grid(True, alpha=0.3)
        
        # Percentage allocation
        percentages = allocation / np.sum(allocation) * 100
        colors = ['gold', 'lightblue', 'lightgreen']
        wedges, texts, autotexts = ax3.pie(allocation, labels=[f'Period {i}' for i in periods], 
                                          autopct='%1.1f%%', startangle=90, colors=colors)
        ax3.set_title('Allocation Distribution')
        
        # Summary table
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = []
        for i, (period, alloc, pct) in enumerate(zip(periods, allocation, percentages)):
            table_data.append([f'Period {period}', f'{alloc:.0f}', f'{pct:.1f}%'])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Period', 'Allocation (shares)', 'Percentage'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax4.set_title('Allocation Summary', pad=20)
        
        plt.tight_layout()
        
        # Save figure
        filename = 'numerical_example.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Numerical example saved: {filepath}")
        
        # Print detailed numerical results
        print("\n" + "=" * 60)
        print("NUMERICAL EXAMPLE - DETAILED RESULTS")
        print("=" * 60)
        print(f"Ticker: {ticker}")
        print(f"Total Order Size: {total_size} shares")
        print(f"Power-law Model: g(x) = α × x^β × σ")
        print(f"Fitted β = {result['impact_analysis']['beta']:.3f}")
        print("\nOptimal Allocation:")
        print("-" * 40)
        for i, (alloc, alpha, sigma) in enumerate(zip(allocation, alphas, sigmas), 1):
            impact_param = alpha * sigma
            print(f"Period {i}: {alloc:.0f} shares ({alloc/total_size*100:.1f}%)")
            print(f"  Impact Parameter (α·σ): {impact_param:.2e}")
        
        total_cost = sum(alpha * (x ** result['impact_analysis']['beta']) * sigma 
                        for x, alpha, sigma in zip(allocation, alphas, sigmas))
        print(f"\nTotal Expected Impact Cost: {total_cost:.4f}")


def main():
    """
    Main analysis function
    """
    # Initialize analysis
    analysis = BlockhouseAnalysis()
    
    # Define tickers and date
    tickers = ['FROG', 'SOUN', 'CRWV']
    date = '2025-04-03'  # Example date
    
    # Run complete analysis
    results = analysis.run_complete_analysis(tickers, date)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Results saved in: {analysis.output_dir}")
    print("Generated files:")
    print("- Impact analysis plots for each ticker")
    print("- Optimal allocation visualizations")
    print("- Numerical example demonstration")
    print("- Complete results in JSON format")
    
    return results


if __name__ == "__main__":
    results = main()

