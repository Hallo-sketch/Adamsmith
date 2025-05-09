import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import t as student_t, kurtosis as scipy_kurtosis, skew, probplot
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.tsa.stattools import acf 
import pandas as pd
from multiprocessing import Pool, cpu_count 

# Helper function to calculate unconditional variance for GJR-GARCH(1,1)
def gjr_garch_unconditional_variance(omega, alpha, beta, gamma_leverage, nu):
    """
    Approximate unconditional variance for GJR-GARCH(1,1) with t-distributed errors.
    """
    if nu <= 2: 
        denominator_simple = 1 - alpha - beta - 0.5 * gamma_leverage
        return omega / denominator_simple if denominator_simple > 1e-6 else omega 

    e_z_squared = nu / (nu - 2) 
    denominator = 1 - (alpha * e_z_squared) - beta - (0.5 * gamma_leverage * e_z_squared)
    
    if denominator > 1e-6: 
        return omega / denominator
    else:
        denominator_simple = 1 - alpha - beta - 0.5 * gamma_leverage
        return omega / denominator_simple if denominator_simple > 1e-6 else omega

# Function to calculate maximum drawdown
def calculate_max_drawdown(price_series):
    """Calculates the maximum drawdown from a price series."""
    if len(price_series) == 0:
        return 0.0
    # Ensure price_series is a numpy array for vectorized operations
    price_series_np = np.asarray(price_series)
    roll_max = np.maximum.accumulate(price_series_np)
    # Calculate drawdown; ensure roll_max is not zero to avoid division by zero
    drawdown = np.zeros_like(price_series_np)
    non_zero_roll_max = roll_max > 1e-9 # Avoid division by zero or tiny numbers
    drawdown[non_zero_roll_max] = (price_series_np[non_zero_roll_max] - roll_max[non_zero_roll_max]) / roll_max[non_zero_roll_max]
    
    max_drawdown_val = np.min(drawdown) if len(drawdown) > 0 and np.any(drawdown < 0) else 0.0 
    return max_drawdown_val 

def simulate_smithian_market(params, T=2500, P0=100.0, natural_price_series=None, nlags_acf=50): 
    """
    Simulates the Smithian market model and returns DataFrame, ACF of absolute returns, and Max Drawdown.
    """
    # Unpack parameters
    omega = params['omega']
    alpha_garch = params['alpha']
    beta_garch = params['beta']
    gamma_leverage = params['gamma_leverage']
    nu = params['nu']
    gamma_price_reversion = params['gamma_price_reversion']
    mu_base = params['mu_base'] 

    prices = np.zeros(T)
    returns = np.zeros(T)
    sigmas_sq = np.zeros(T)
    epsilons = np.zeros(T)
    prices[0] = P0

    sigmas_sq[0] = gjr_garch_unconditional_variance(omega, alpha_garch, beta_garch, gamma_leverage, nu)
    if sigmas_sq[0] <= 0 or not np.isfinite(sigmas_sq[0]): 
        sigmas_sq[0] = omega 

    z0 = student_t.rvs(df=nu, size=1)[0]
    if nu > 2:
        z0 *= np.sqrt((nu - 2) / nu) 
    epsilons[0] = np.sqrt(sigmas_sq[0]) * z0

    if natural_price_series is None or len(natural_price_series) != T:
        raise ValueError("A valid natural_price_series of length T must be provided.")

    for t in range(1, T):
        mean_reversion_effect = gamma_price_reversion * (natural_price_series[t-1] - prices[t-1]) / prices[t-1] if prices[t-1] > 1e-9 else 0
        mu_t = mu_base + mean_reversion_effect 

        leverage_term = gamma_leverage * (epsilons[t-1]**2) * (1 if epsilons[t-1] < 0 else 0)
        sigmas_sq[t] = omega + alpha_garch * (epsilons[t-1]**2) + beta_garch * sigmas_sq[t-1] + leverage_term
        if sigmas_sq[t] <= 1e-9: 
            sigmas_sq[t] = 1e-9 

        z_t = student_t.rvs(df=nu, size=1)[0]
        if nu > 2:
            z_t *= np.sqrt((nu - 2) / nu) 

        epsilons[t] = np.sqrt(sigmas_sq[t]) * z_t
        returns[t] = mu_t + epsilons[t]
        
        prices[t] = prices[t-1] * (1 + returns[t])
        if prices[t] < 1e-6: 
            prices[t] = 1e-6 
            
    df = pd.DataFrame({
        'Price': prices,
        'Return': np.concatenate(([np.nan], returns[1:])), 
        'CondVolatility': np.concatenate(([np.nan], np.sqrt(sigmas_sq[1:]))),
        'Shock_epsilon': np.concatenate(([np.nan], epsilons[1:])),
        'CondVariance_sq': np.concatenate(([np.nan], sigmas_sq[1:]))
    })
    df.iloc[0, df.columns.get_loc('CondVolatility')] = np.sqrt(sigmas_sq[0])
    df.iloc[0, df.columns.get_loc('Shock_epsilon')] = epsilons[0]
    df.iloc[0, df.columns.get_loc('CondVariance_sq')] = sigmas_sq[0]
    
    abs_returns_for_acf = np.abs(df['Return'].dropna())
    acf_values = acf(abs_returns_for_acf, nlags=nlags_acf, fft=True) 
    max_dd = calculate_max_drawdown(df['Price'].values)

    return df, acf_values, max_dd 

# Worker function for parallel simulation
def worker_simulation(args):
    params, T, P0, natural_price_series, nlags_acf = args 
    return simulate_smithian_market(params, T=T, P0=P0, natural_price_series=natural_price_series, nlags_acf=nlags_acf)

# --- Define Parameters for Scenarios ---
T_periods = 2500 
N_runs = 500      
P0_initial = 100.0 
NLAGS_FOR_ACF = 50 

fundamental_total_target_return = 0.10 
fundamental_daily_growth_rate = (1 + fundamental_total_target_return)**(1/T_periods) - 1
print(f"Fundamental daily growth rate (for Natural Price & mu_base A & B): {fundamental_daily_growth_rate:.8f}")

natural_price_path = P0_initial * (1 + fundamental_daily_growth_rate)**np.arange(T_periods)

params_A = {
    'omega': 3.6e-6, 
    'alpha': 0.08,
    'beta': 0.85,    
    'gamma_leverage': 0.02,
    'nu': 10,
    'gamma_price_reversion': 0.05, 
    'mu_base': fundamental_daily_growth_rate 
}

params_B = {
    'omega': 1.5e-5, 
    'alpha': 0.07,
    'beta': 0.82,    
    'gamma_leverage': 0.06,
    'nu': 5,
    'gamma_price_reversion': 0.01, 
    'mu_base': fundamental_daily_growth_rate 
}

# --- Run Simulations (Monte Carlo) ---
all_prices_A, all_prices_B = [], []
all_returns_A, all_returns_B = [], [] 
run_stats_A, run_stats_B = [], []     
all_acf_values_A, all_acf_values_B = [], [] 
all_max_dd_A, all_max_dd_B = [], [] 


if __name__ == '__main__':
    num_cores_to_use = cpu_count() - 1 if cpu_count() > 1 else 1 
    print(f"Starting {N_runs} Monte Carlo simulations for each scenario using {num_cores_to_use} cores...")

    tasks_A = [(params_A, T_periods, P0_initial, natural_price_path, NLAGS_FOR_ACF) for _ in range(N_runs)]
    tasks_B = [(params_B, T_periods, P0_initial, natural_price_path, NLAGS_FOR_ACF) for _ in range(N_runs)]

    with Pool(processes=num_cores_to_use) as pool:
        results_A = pool.map(worker_simulation, tasks_A) 
        results_B = pool.map(worker_simulation, tasks_B) 
    
    print("Simulations complete. Processing and aggregating results...")

    final_prices_A, final_prices_B = [], []

    for df_A_run, acf_A_run, max_dd_A_run in results_A:
        all_prices_A.append(df_A_run['Price'].values)
        final_prices_A.append(df_A_run['Price'].iloc[-1])
        returns_A_no_nan_run = df_A_run['Return'].dropna()
        all_returns_A.extend(returns_A_no_nan_run.tolist()) 
        all_acf_values_A.append(acf_A_run)
        all_max_dd_A.append(max_dd_A_run)
        run_stats_A.append({
            'mean_return': returns_A_no_nan_run.mean(),
            'std_return': returns_A_no_nan_run.std(),
            'kurtosis': scipy_kurtosis(returns_A_no_nan_run, fisher=True), 
            'skewness': skew(returns_A_no_nan_run),
            'mean_cond_vol': df_A_run['CondVolatility'].dropna().mean(),
            'max_drawdown': max_dd_A_run
        })

    for df_B_run, acf_B_run, max_dd_B_run in results_B:
        all_prices_B.append(df_B_run['Price'].values)
        final_prices_B.append(df_B_run['Price'].iloc[-1])
        returns_B_no_nan_run = df_B_run['Return'].dropna()
        all_returns_B.extend(returns_B_no_nan_run.tolist())
        all_acf_values_B.append(acf_B_run)
        all_max_dd_B.append(max_dd_B_run)
        run_stats_B.append({
            'mean_return': returns_B_no_nan_run.mean(),
            'std_return': returns_B_no_nan_run.std(),
            'kurtosis': scipy_kurtosis(returns_B_no_nan_run, fisher=True),
            'skewness': skew(returns_B_no_nan_run),
            'mean_cond_vol': df_B_run['CondVolatility'].dropna().mean(),
            'max_drawdown': max_dd_B_run
        })

    all_prices_A_np = np.array(all_prices_A) 
    all_prices_B_np = np.array(all_prices_B) 
    all_acf_values_A = np.array(all_acf_values_A)
    all_acf_values_B = np.array(all_acf_values_B)

    df_run_stats_A = pd.DataFrame(run_stats_A)
    df_run_stats_B = pd.DataFrame(run_stats_B)
    print("Result aggregation complete. Generating plots...")

    mean_acf_A = np.mean(all_acf_values_A, axis=0)
    mean_acf_B = np.mean(all_acf_values_B, axis=0)
    lags_for_plot = np.arange(len(mean_acf_A)) 

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif', 
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'font.size': 10, 
        'axes.titlesize': 12, 
        'axes.labelsize': 10, 
        'xtick.labelsize': 9, 
        'ytick.labelsize': 9, 
        'legend.fontsize': 9, 
        'figure.titlesize': 14,
        'figure.facecolor': 'white', 
        'axes.facecolor': 'white',   
        'axes.edgecolor': 'lightgray', 
        'grid.color': 'lightgray',   
        'grid.linestyle': '--',      
        'patch.edgecolor': 'darkgrey' 
    })

    # --- Plot 1: Simulated Price Paths (Mean & Percentiles) ---
    plt.figure(figsize=(10, 6))
    mean_price_A_path = np.mean(all_prices_A_np, axis=0) 
    percentile_5_price_A = np.percentile(all_prices_A_np, 5, axis=0)
    percentile_95_price_A = np.percentile(all_prices_A_np, 95, axis=0)
    mean_price_B_path = np.mean(all_prices_B_np, axis=0) 
    percentile_5_price_B = np.percentile(all_prices_B_np, 5, axis=0)
    percentile_95_price_B = np.percentile(all_prices_B_np, 95, axis=0)

    plt.plot(mean_price_A_path, label=f"Scenario A Mean Price", color='royalblue', linewidth=1.5) 
    plt.fill_between(range(T_periods), percentile_5_price_A, percentile_95_price_A, color='lightblue', alpha=0.5, label="A: 5th-95th Percentile")
    plt.plot(mean_price_B_path, label=f"Scenario B Mean Price", color='crimson', linewidth=1.5) 
    plt.fill_between(range(T_periods), percentile_5_price_B, percentile_95_price_B, color='lightcoral', alpha=0.5, label="B: 5th-95th Percentile")
    plt.plot(natural_price_path, label=f"Natural Price Path ({fundamental_total_target_return*100:.0f}% growth)", color='green', linestyle='--', lw=1.5) 
    plt.title(f"Simulated Price Paths (Mean & Percentiles, {N_runs} Runs)")
    plt.ylabel("Price")
    plt.xlabel("Time Period")
    plt.legend(loc='upper left')
    plt.grid(True) 
    plt.savefig('plot_01_price_paths.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Plot 2: Aggregated Distribution of Daily Returns ---
    plt.figure(figsize=(8, 6))
    returns_A_all_runs = np.array(all_returns_A) 
    returns_B_all_runs = np.array(all_returns_B)
    sns.histplot(returns_A_all_runs, bins=150, kde=False, stat="density", color='royalblue', alpha=0.7, label=f"Scenario A ($\\nu$={params_A['nu']})", element="step")
    sns.histplot(returns_B_all_runs, bins=150, kde=False, stat="density", color='crimson', alpha=0.6, label=f"Scenario B ($\\nu$={params_B['nu']})", element="step")
    plt.title(f"Aggregated Distribution of Daily Returns ({N_runs*T_periods} points)") 
    plt.xlabel("Daily Return")
    plt.ylabel("Density")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('plot_02_return_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Plot 3: Distribution of Final Prices ---
    plt.figure(figsize=(8, 6))
    sns.histplot(final_prices_A, kde=True, stat="density", color='royalblue', label='Scenario A', element="step")
    sns.histplot(final_prices_B, kde=True, stat="density", color='crimson', label='Scenario B', element="step", alpha=0.7)
    plt.title(f'Distribution of Final Prices ({N_runs} runs)')
    plt.xlabel('Final Price')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('plot_03_final_price_dist.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Plot 4: Distribution of Kurtosis per Run ---
    plt.figure(figsize=(8, 6))
    sns.histplot(df_run_stats_A['kurtosis'], kde=True, stat="density", color='royalblue', label='Scenario A', element="step")
    sns.histplot(df_run_stats_B['kurtosis'], kde=True, stat="density", color='crimson', label='Scenario B', element="step", alpha=0.7)
    plt.title(f'Distribution of Kurtosis per Run ({N_runs} runs)')
    plt.xlabel('Kurtosis (Excess)')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('plot_04_kurtosis_dist.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- Plot 5: Distribution of Skewness per Run ---
    plt.figure(figsize=(8, 6))
    sns.histplot(df_run_stats_A['skewness'], kde=True, stat="density", color='royalblue', label='Scenario A', element="step")
    sns.histplot(df_run_stats_B['skewness'], kde=True, stat="density", color='crimson', label='Scenario B', element="step", alpha=0.7)
    plt.title(f'Distribution of Skewness per Run ({N_runs} runs)')
    plt.xlabel('Skewness')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('plot_05_skewness_dist.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Plot 6: Distribution of Avg. Cond. Vol. per Run ---
    plt.figure(figsize=(8, 6))
    sns.histplot(df_run_stats_A['mean_cond_vol'], kde=True, stat="density", color='royalblue', label='Scenario A', element="step")
    sns.histplot(df_run_stats_B['mean_cond_vol'], kde=True, stat="density", color='crimson', label='Scenario B', element="step", alpha=0.7)
    plt.title(f'Distribution of Avg. Cond. Vol. per Run ({N_runs} runs)')
    plt.xlabel('Average Conditional Volatility per Run')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('plot_06_avg_cond_vol_dist.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Plot 7: Distribution of Max Drawdowns per Run ---
    plt.figure(figsize=(8, 6))
    # Max drawdowns are negative, multiply by -100 for positive percentage display
    sns.histplot(-100 * df_run_stats_A['max_drawdown'], kde=True, stat="density", color='royalblue', label='Scenario A', element="step")
    sns.histplot(-100 * df_run_stats_B['max_drawdown'], kde=True, stat="density", color='crimson', label='Scenario B', element="step", alpha=0.7)
    plt.title(f'Distribution of Max Drawdowns per Run ({N_runs} runs)')
    plt.xlabel('Maximum Drawdown (%)')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('plot_07_max_drawdown_dist.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Plot 8: Mean ACF of Abs. Returns (Scenario A) ---
    plt.figure(figsize=(8, 6))
    markerline_A, stemlines_A, baseline_A = plt.stem(lags_for_plot[1:], mean_acf_A[1:], linefmt='royalblue', markerfmt='o', basefmt="gray")
    plt.setp(markerline_A, markersize=4, color='royalblue')
    plt.setp(stemlines_A, linewidth=1.5, color='royalblue')
    plt.title(f"Mean ACF of Abs. Returns (Scenario A, {N_runs} runs)")
    plt.ylabel("Mean ACF")
    plt.xlabel("Lag")
    plt.grid(True)
    plt.savefig('plot_08_mean_acf_A.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Plot 9: Mean ACF of Abs. Returns (Scenario B) ---
    plt.figure(figsize=(8, 6))
    markerline_B, stemlines_B, baseline_B = plt.stem(lags_for_plot[1:], mean_acf_B[1:], linefmt='crimson', markerfmt='o', basefmt="gray")
    plt.setp(markerline_B, markersize=4, color='crimson')
    plt.setp(stemlines_B, linewidth=1.5, color='crimson')
    plt.title(f"Mean ACF of Abs. Returns (Scenario B, {N_runs} runs)")
    plt.xlabel("Lag")
    plt.ylabel("Mean ACF") 
    plt.grid(True)
    plt.savefig('plot_09_mean_acf_B.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- Plot 10: QQ Plot of Aggregated Daily Returns (Scenario A vs Normal) ---
    plt.figure(figsize=(7, 7))
    probplot(returns_A_all_runs, dist="norm", plot=plt)
    plt.title(f"QQ Plot of Aggregated Daily Returns (Scenario A vs Normal, {N_runs} runs)")
    plt.xlabel("Theoretical Quantiles (Normal)")
    plt.ylabel("Sample Quantiles (Scenario A)")
    plt.grid(True)
    plt.savefig('plot_10_qqplot_A.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Plot 11: QQ Plot of Aggregated Daily Returns (Scenario B vs Normal) ---
    plt.figure(figsize=(7, 7))
    probplot(returns_B_all_runs, dist="norm", plot=plt)
    plt.title(f"QQ Plot of Aggregated Daily Returns (Scenario B vs Normal, {N_runs} runs)")
    plt.xlabel("Theoretical Quantiles (Normal)")
    plt.ylabel("Sample Quantiles (Scenario B)")
    plt.grid(True)
    plt.savefig('plot_11_qqplot_B.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("All plots generated and saved.")

    # --- Save Generated Data ---
    print("\\n--- Saving Generated Data ---")
    np.save('output_all_prices_A.npy', all_prices_A_np)
    np.save('output_all_prices_B.npy', all_prices_B_np)
    np.save('output_all_returns_A.npy', np.array(all_returns_A))
    np.save('output_all_returns_B.npy', np.array(all_returns_B))
    np.save('output_final_prices_A.npy', np.array(final_prices_A))
    np.save('output_final_prices_B.npy', np.array(final_prices_B))
    np.save('output_all_acf_values_A.npy', all_acf_values_A)
    np.save('output_all_acf_values_B.npy', all_acf_values_B)
    np.save('output_all_max_dd_A.npy', np.array(all_max_dd_A))
    np.save('output_all_max_dd_B.npy', np.array(all_max_dd_B))
    df_run_stats_A.to_csv('output_df_run_stats_A.csv', index=False)
    df_run_stats_B.to_csv('output_df_run_stats_B.csv', index=False)
    np.save('output_mean_acf_A.npy', mean_acf_A)
    np.save('output_mean_acf_B.npy', mean_acf_B)
    np.save('output_natural_price_path.npy', natural_price_path)
    print("All data saved to .npy and .csv files.")
    # --- End Save Generated Data ---

    print(f"\\n--- Aggregated Statistics over {N_runs} runs ---")
    mean_A_agg = np.mean(returns_A_all_runs)
    std_A_agg = np.std(returns_A_all_runs)
    skew_A_agg = skew(returns_A_all_runs)

    mean_B_agg = np.mean(returns_B_all_runs)
    std_B_agg = np.std(returns_B_all_runs)
    skew_B_agg = skew(returns_B_all_runs)
    kurt_A_agg = scipy_kurtosis(returns_A_all_runs, fisher=True) # Define kurt_A_agg
    kurt_B_agg = scipy_kurtosis(returns_B_all_runs, fisher=True) # Define kurt_B_agg

    print("\nScenario A (More Regulated / Smithian Ideal):")
    print(f"  Underlying Fundamental Total Return Target: {fundamental_total_target_return:.2%}")
    print(f"  Avg. Mean Daily Return (across runs): {df_run_stats_A['mean_return'].mean():.8f} (Std of Means: {df_run_stats_A['mean_return'].std():.8f})")
    print(f"  Avg. Daily Std Dev Return (across runs): {df_run_stats_A['std_return'].mean():.6f} (Std of StdDevs: {df_run_stats_A['std_return'].std():.6f})")
    print(f"  Avg. Kurtosis (across runs): {df_run_stats_A['kurtosis'].mean():.2f} (Std of Kurtoses: {df_run_stats_A['kurtosis'].std():.2f})")
    print(f"  Avg. Skewness (across runs): {df_run_stats_A['skewness'].mean():.2f} (Std of Skewnesses: {df_run_stats_A['skewness'].std():.2f})")
    print(f"  Avg. Mean Cond. Vol (across runs): {df_run_stats_A['mean_cond_vol'].mean():.6f} (Std of MeanVols: {df_run_stats_A['mean_cond_vol'].std():.6f})")
    print(f"  Avg. Max Drawdown (across runs): {df_run_stats_A['max_drawdown'].mean()*100:.2f}% (Std: {df_run_stats_A['max_drawdown'].std()*100:.2f}%)")
    print(f"  Overall Mean (from all aggregated returns): {mean_A_agg:.8f}")
    print(f"  Overall Std Dev (from all aggregated returns): {std_A_agg:.6f}")
    print(f"  Overall Skewness (from all aggregated returns): {skew_A_agg:.2f}")
    print(f"  Overall Kurtosis (from all aggregated returns): {kurt_A_agg:.2f}")

    print("\nScenario B (Less Regulated / Behavioral Impact):")
    print(f"  Underlying Fundamental Total Return Target (same as A): {fundamental_total_target_return:.2%}")
    print(f"  Avg. Mean Daily Return (across runs): {df_run_stats_B['mean_return'].mean():.8f} (Std of Means: {df_run_stats_B['mean_return'].std():.8f})")
    print(f"  Avg. Daily Std Dev Return (across runs): {df_run_stats_B['std_return'].mean():.6f} (Std of StdDevs: {df_run_stats_B['std_return'].std():.6f})")
    print(f"  Avg. Kurtosis (across runs): {df_run_stats_B['kurtosis'].mean():.2f} (Std of Kurtoses: {df_run_stats_B['kurtosis'].std():.2f})")
    print(f"  Avg. Skewness (across runs): {df_run_stats_B['skewness'].mean():.2f} (Std of Skewnesses: {df_run_stats_B['skewness'].std():.2f})")
    print(f"  Avg. Mean Cond. Vol (across runs): {df_run_stats_B['mean_cond_vol'].mean():.6f} (Std of MeanVols: {df_run_stats_B['mean_cond_vol'].std():.6f})")
    print(f"  Avg. Max Drawdown (across runs): {df_run_stats_B['max_drawdown'].mean()*100:.2f}% (Std: {df_run_stats_B['max_drawdown'].std()*100:.2f}%)")
    print(f"  Overall Mean (from all aggregated returns): {mean_B_agg:.8f}")
    print(f"  Overall Std Dev (from all aggregated returns): {std_B_agg:.6f}")
    print(f"  Overall Skewness (from all aggregated returns): {skew_B_agg:.2f}")
    print(f"  Overall Kurtosis (from all aggregated returns): {kurt_B_agg:.2f}")

    print("\n--- GARCH Stationarity Checks (Theoretical) ---")
    e_z_sq_A = params_A['nu']/(params_A['nu']-2) if params_A['nu'] > 2 else 1.0 
    check_A = (params_A['alpha'] + 0.5 * params_A['gamma_leverage']) * e_z_sq_A + params_A['beta']
    print(f"Scenario A GARCH persistence metric: {check_A:.4f} (should be < 1 for stationarity)")

    e_z_sq_B = params_B['nu']/(params_B['nu']-2) if params_B['nu'] > 2 else 1.0 
    check_B = (params_B['alpha'] + 0.5 * params_B['gamma_leverage']) * e_z_sq_B + params_B['beta']
    print(f"Scenario B GARCH persistence metric: {check_B:.4f} (should be < 1 for stationarity)")

    mean_price_A_path_val = np.mean(all_prices_A_np, axis=0) # Use the correct variable name
    mean_price_B_path_val = np.mean(all_prices_B_np, axis=0) # Use the correct variable name

    avg_daily_ret_A_sim = df_run_stats_A['mean_return'].mean()
    net_10yr_ret_A_sim = (1 + avg_daily_ret_A_sim)**T_periods - 1
    print(f"\nProjected 10-year net return for Scenario A (from sim avg daily mean): {net_10yr_ret_A_sim:.4f} (Fundamental Target ~{fundamental_total_target_return:.2f})")
    print(f"  Mean final price for Scenario A: {mean_price_A_path_val[-1]:.2f} (Fundamental Target ~{P0_initial*(1+fundamental_total_target_return):.2f})")

    avg_daily_ret_B_sim = df_run_stats_B['mean_return'].mean()
    net_10yr_ret_B_sim = (1 + avg_daily_ret_B_sim)**T_periods - 1
    print(f"Projected 10-year net return for Scenario B (from sim avg daily mean): {net_10yr_ret_B_sim:.4f} (Fundamental Target ~{fundamental_total_target_return:.2f})")
    print(f"  Mean final price for Scenario B: {mean_price_B_path_val[-1]:.2f} (Fundamental Target ~{P0_initial*(1+fundamental_total_target_return):.2f})")

