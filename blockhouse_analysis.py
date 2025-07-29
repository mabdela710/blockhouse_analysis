import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from zipfile import ZipFile
import os
from scipy.optimize import curve_fit, minimize

sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)

def load_and_preprocess_data(ticker_paths):
    ticker_dfs = {}
    for ticker, zip_path in ticker_paths.items():
        print(f'\n--- Processing data for {ticker} ---')
        full_zip_path = os.path.join('/home/ubuntu/blockhouse_analysis/blockhouse_analysis-main', zip_path)
        extracted_path = os.path.join('/home/ubuntu/blockhouse_analysis/blockhouse_analysis-main', f'data/{ticker}/extracted')

        df = None
        # First, try to load directly from CSV if it exists alongside the zip
        csv_file_direct = os.path.join(os.path.dirname(full_zip_path), f'{ticker}_2025-04-03 00:00:00+00:00.csv')
        if os.path.exists(csv_file_direct):
            try:
                df = pd.read_csv(csv_file_direct)
                print(f'✅ Data loaded directly from CSV: {csv_file_direct}')
            except Exception as e:
                print(f'❌ Error reading direct CSV {csv_file_direct}: {e}')

        if df is None and os.path.exists(full_zip_path):
            try:
                with ZipFile(full_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extracted_path)
                    csv_files = [f for f in os.listdir(extracted_path) if f.endswith(".csv")]

                    if not csv_files:
                        print(f'❌ No CSV file inside {full_zip_path}')
                        continue

                    # Try to find the main CSV file, or just take the first one
                    main_csv_file = next((f for f in csv_files if f.startswith(ticker)), csv_files[0])
                    df = pd.read_csv(os.path.join(extracted_path, main_csv_file))
                    print(f'✅ Data loaded from zip: {os.path.join(extracted_path, main_csv_file)}')

            except Exception as e:
                print(f'❌ Error processing {ticker} zip file: {e}')

        if df is not None:
            # Rename columns for consistency
            df.rename(columns={'size': 'volume', 'bid_px_00': 'bid_price', 'ask_px_00': 'ask_price', 
                               'bid_sz_00': 'bid_size', 'ask_sz_00': 'ask_size'}, inplace=True)

            # Ensure 'price' and 'volume' columns exist, use existing data or set to 0 if missing
            if 'price' not in df.columns:
                if 'close' in df.columns:
                    df['price'] = df['close']
                elif 'Close' in df.columns:
                    df['price'] = df['Close']
                else:
                    print(f'⚠️ No price column found for {ticker}, setting to 0.')
                    df['price'] = 0 # Set to 0, not dummy data
            
            if 'volume' not in df.columns:
                if 'Volume' in df.columns:
                    df['volume'] = df['Volume']
                else:
                    print(f'⚠️ No volume column found for {ticker}, setting to 0.')
                    df['volume'] = 0 # Set to 0, not dummy data

            ticker_dfs[ticker] = df
            print(f'✅ Data loaded ({len(df)}) rows')
        else:
            print(f'⚠️ No data loaded for {ticker}.')

    return ticker_dfs

def calculate_temporary_impact(df):
    # Ensure necessary columns exist for slippage calculation
    required_slippage_cols = ['bid_price', 'ask_price', 'bid_size', 'ask_size', 'volume']
    if not all(col in df.columns for col in required_slippage_cols):
        print("Warning: Insufficient order book data for accurate slippage calculation. Using simplified impact.")
        # Fallback to simplified impact if order book data is missing
        if 'price' not in df.columns or 'volume' not in df.columns:
            df['impact'] = 0
            return df
        df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2 if 'bid_price' in df.columns and 'ask_price' in df.columns else df['price']
        df['price_change'] = df['mid_price'].diff().fillna(0)
        df['impact'] = df['price_change'] * df['volume']
        return df

    df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
    df['slippage'] = 0.0 # Initialize slippage

    # Simulate execution for each row to calculate slippage
    # This is a simplified simulation for demonstration purposes.
    # A full simulation would involve iterating through all order book levels.
    for index, row in df.iterrows():
        volume_to_execute = row['volume']
        mid_price = row['mid_price']
        current_slippage = 0.0

        # Assuming a buy order (consuming ask side)
        # If 'side' column exists and indicates 'sell', then consume bid side
        # For this example, we'll assume buy side impact calculation
        
        # Consume best ask level
        if volume_to_execute > 0 and row['ask_size'] > 0:
            volume_consumed = min(volume_to_execute, row['ask_size'])
            current_slippage += volume_consumed * (row['ask_price'] - mid_price)
            volume_to_execute -= volume_consumed

        # If volume still remains, consume next ask level (simplified to ask_px_01/ask_sz_01 if available)
        if volume_to_execute > 0 and 'ask_px_01' in row and 'ask_sz_01' in row and row['ask_sz_01'] > 0:
            volume_consumed = min(volume_to_execute, row['ask_sz_01'])
            current_slippage += volume_consumed * (row['ask_px_01'] - mid_price)
            volume_to_execute -= volume_consumed
            
        # For a sell order (consuming bid side)
        # This part would be activated if 'side' column indicates 'sell'
        # For now, assuming buy side impact for demonstration
        
        df.loc[index, 'slippage'] = current_slippage
        
    df['impact'] = df['slippage'] # Use calculated slippage as impact
    return df

def linear_model(x, beta):
    return beta * x

def nonlinear_model(x, alpha, p):
    return alpha * np.power(x, p)

def fit_impact_models(ticker_dfs):
    fit_results = {}
    for ticker, df in ticker_dfs.items():
        if 'price' in df.columns and 'volume' in df.columns:
            print(f'\n--- Model fitting for {ticker} ---')
            df = df.copy()
            df = calculate_temporary_impact(df)

            x = df['volume'].values
            y = df['impact'].values

            non_zero_volume_mask = (x > 0) & (~np.isnan(y))
            x_filtered = x[non_zero_volume_mask]
            y_filtered = y[non_zero_volume_mask]

            if len(x_filtered) < 2:
                print(f'Skipping model fitting for {ticker} due to insufficient valid data after filtering.')
                continue

            try:
                popt_lin, _ = curve_fit(linear_model, x_filtered, y_filtered)
            except Exception as e:
                print(f'Error in linear model for {ticker}: {e}')
                popt_lin = [np.nan]

            try:
                popt_nonlin, _ = curve_fit(nonlinear_model, x_filtered, y_filtered, bounds=([0, 0], [np.inf, 2]))
            except Exception as e:
                print(f'Error in nonlinear model for {ticker}: {e}')
                popt_nonlin = [np.nan, np.nan]

            fit_results[ticker] = {'linear': popt_lin, 'nonlinear': popt_nonlin}

            plt.figure(figsize=(7, 4))
            plt.scatter(x_filtered, y_filtered, alpha=0.3, label='Data')

            if not np.isnan(popt_lin[0]):
                x_fit = np.linspace(x_filtered.min(), x_filtered.max(), 100)
                plt.plot(x_fit, linear_model(x_fit, *popt_lin), label='Linear Model', color='royalblue')
                print(f'Linear model coefficient (beta): {popt_lin[0]:.4g}')

            if not np.isnan(popt_nonlin[0]):
                x_fit = np.linspace(x_filtered.min(), x_filtered.max(), 100)
                plt.plot(x_fit, nonlinear_model(x_fit, *popt_nonlin), label='Nonlinear Model', color='darkorange')
                print(f'Nonlinear model coefficients (alpha, p): {popt_nonlin[0]:.4g}, {popt_nonlin[1]:.4g}')

            plt.title(f'Temporary Impact Fitting for {ticker}')
            plt.xlabel('Volume')
            plt.ylabel('Price Change')
            plt.legend()
            plt.savefig(f'/home/ubuntu/blockhouse_analysis/blockhouse_analysis-main/output/{ticker}_impact_analysis.png')
            plt.close()
    return fit_results

def optimal_execution(S, N, alpha, p, P0):
    def cost(x):
        x = np.array(x)
        return np.sum(P0 * x + alpha * np.power(x, p))

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - S})
    bounds = [(0, None)] * N
    x0 = np.ones(N) * (S / N)

    res = minimize(cost, x0, bounds=bounds, constraints=cons)
    print('Optimal x:', res.x)
    print('Total cost:', res.fun)
    return res.x, res.fun

if __name__ == '__main__':
    ticker_paths = {
        'CRWV': 'data/CRWV/CRWV.zip',
        'FROG': 'data/FROG/FROG.zip',
        'SOUN': 'data/SOUN/SOUN.zip'
    }

    ticker_dfs = load_and_preprocess_data(ticker_paths)

    if ticker_dfs:
        fit_results = fit_impact_models(ticker_dfs)
        print("\n--- Optimal Execution Example ---")
        S_total = 10000
        N_intervals = 10
        P0_initial = 100

        avg_alpha = np.nanmean([res['nonlinear'][0] for res in fit_results.values() if not np.isnan(res['nonlinear'][0])]) if fit_results else 0.01
        avg_p = np.nanmean([res['nonlinear'][1] for res in fit_results.values() if not np.isnan(res['nonlinear'][1])]) if fit_results else 0.8

        if np.isnan(avg_alpha): avg_alpha = 0.01
        if np.isnan(avg_p): avg_p = 0.8

        optimal_x, total_cost = optimal_execution(S_total, N_intervals, avg_alpha, avg_p, P0_initial)

        plt.figure(figsize=(8, 5))
        plt.bar(range(N_intervals), optimal_x)
        plt.title('Optimal Order Allocation')
        plt.xlabel('Time Interval')
        plt.ylabel('Shares to Trade')
        plt.savefig(f'/home/ubuntu/blockhouse_analysis/blockhouse_analysis-main/output/optimal_allocation_example.png')
        plt.close()

    else:
        print('No data loaded. Cannot proceed with analysis.')

