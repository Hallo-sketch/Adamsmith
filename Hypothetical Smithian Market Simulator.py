import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn
from scipy.stats import t as student_t, kurtosis as scipy_kurtosis, skew
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


def simulate_smithian_market(params, T=2500, P0=100.0, natural_price_series=None, nlags_acf=50): 
    """
    Simulates the Smithian market model and returns DataFrame and ACF of absolute returns.
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

    return df, acf_values 

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


if __name__ == '__main__':
    num_cores_to_use = cpu_count() - 1 if cpu_count() > 1 else 1 
    print(f"Starting {N_runs} Monte Carlo simulations for each scenario using {num_cores_to_use} cores...")

    tasks_A = [(params_A, T_periods, P0_initial, natural_price_path, NLAGS_FOR_ACF) for _ in range(N_runs)]
    tasks_B = [(params_B, T_periods, P0_initial, natural_price_path, NLAGS_FOR_ACF) for _ in range(N_runs)]

    with Pool(processes=num_cores_to_use) as pool:
        results_A = pool.map(worker_simulation, tasks_A) 
        results_B = pool.map(worker_simulation, tasks_B) 
    
    print("Simulations complete. Processing and aggregating results...")

    final_prices_A = []
    final_prices_B = []

    for df_A_run, acf_A_run in results_A:
        all_prices_A.append(df_A_run['Price'].values)
        final_prices_A.append(df_A_run['Price'].iloc[-1]) # Store final price for this run
        returns_A_no_nan_run = df_A_run['Return'].dropna()
        all_returns_A.extend(returns_A_no_nan_run.tolist()) 
        all_acf_values_A.append(acf_A_run)
        run_stats_A.append({
            'mean_return': returns_A_no_nan_run.mean(),
            'std_return': returns_A_no_nan_run.std(),
            'kurtosis': scipy_kurtosis(returns_A_no_nan_run, fisher=True), 
            'skewness': skew(returns_A_no_nan_run),
            'mean_cond_vol': df_A_run['CondVolatility'].dropna().mean() 
        })

    for df_B_run, acf_B_run in results_B:
        all_prices_B.append(df_B_run['Price'].values)
        final_prices_B.append(df_B_run['Price'].iloc[-1]) # Store final price for this run
        returns_B_no_nan_run = df_B_run['Return'].dropna()
        all_returns_B.extend(returns_B_no_nan_run.tolist())
        all_acf_values_B.append(acf_B_run)
        run_stats_B.append({
            'mean_return': returns_B_no_nan_run.mean(),
            'std_return': returns_B_no_nan_run.std(),
            'kurtosis': scipy_kurtosis(returns_B_no_nan_run, fisher=True),
            'skewness': skew(returns_B_no_nan_run),
            'mean_cond_vol': df_B_run['CondVolatility'].dropna().mean() 
        })

    all_prices_A_np = np.array(all_prices_A) # Renamed to avoid conflict
    all_prices_B_np = np.array(all_prices_B) # Renamed to avoid conflict
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
        'patch.edgecolor': 'black'   
    })


    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 12)) # Removed sharex='col' for independent x-axes if needed
    fig.suptitle(f"Smithian Market Model Simulation ({N_runs} Runs): Scenario A vs. Scenario B", fontweight='bold')

    mean_price_A_path = np.mean(all_prices_A_np, axis=0) 
    percentile_5_price_A = np.percentile(all_prices_A_np, 5, axis=0)
    percentile_95_price_A = np.percentile(all_prices_A_np, 95, axis=0)

    mean_price_B_path = np.mean(all_prices_B_np, axis=0) 
    percentile_5_price_B = np.percentile(all_prices_B_np, 5, axis=0)
    percentile_95_price_B = np.percentile(all_prices_B_np, 95, axis=0)

    axs[0, 0].plot(mean_price_A_path, label=f"Scenario A Mean Price", color='royalblue', linewidth=1.5) 
    axs[0, 0].fill_between(range(T_periods), percentile_5_price_A, percentile_95_price_A, color='lightblue', alpha=0.5, label="A: 5th-95th Percentile")

    axs[0, 0].plot(mean_price_B_path, label=f"Scenario B Mean Price", color='crimson', linewidth=1.5) 
    axs[0, 0].fill_between(range(T_periods), percentile_5_price_B, percentile_95_price_B, color='lightcoral', alpha=0.5, label="B: 5th-95th Percentile")

    axs[0, 0].plot(natural_price_path, label=f"Natural Price Path ({fundamental_total_target_return*100:.0f}% growth)", color='green', linestyle='--', lw=1.5) 
    axs[0, 0].set_title("Simulated Price Paths (Mean & 5th-95th Percentiles)")
    axs[0, 0].set_ylabel("Price")
    axs[0, 0].set_xlabel("Time Period")
    axs[0, 0].legend(loc='upper left')
    axs[0, 0].grid(True) 
    
    # Plot Distribution of Final Prices
    sns.histplot(final_prices_A, ax=axs[1,0], kde=True, stat="density", color='royalblue', label='Scenario A Final Prices', element="step")
    sns.histplot(final_prices_B, ax=axs[1,0], kde=True, stat="density", color='crimson', label='Scenario B Final Prices', element="step", alpha=0.7)
    axs[1,0].set_title(f'Distribution of Final Prices ({N_runs} runs)')
    axs[1,0].set_xlabel('Final Price')
    axs[1,0].set_ylabel('Density')
    axs[1,0].legend(loc='upper right')
    axs[1,0].grid(True)

    # Plot Distribution of Mean Conditional Volatilities per run
    sns.histplot(df_run_stats_A['mean_cond_vol'], ax=axs[2,0], kde=True, stat="density", color='royalblue', label='Scenario A Avg. Cond. Vol.', element="step")
    sns.histplot(df_run_stats_B['mean_cond_vol'], ax=axs[2,0], kde=True, stat="density", color='crimson', label='Scenario B Avg. Cond. Vol.', element="step", alpha=0.7)
    axs[2,0].set_title(f'Distribution of Avg. Cond. Volatility per Run ({N_runs} runs)')
    axs[2,0].set_xlabel('Average Conditional Volatility per Run')
    axs[2,0].set_ylabel('Density')
    axs[2,0].legend(loc='upper right')
    axs[2,0].grid(True)
    axs[2,0].set_xlabel("Time Period / Avg. Cond. Vol.") # Combined xlabel for bottom row

    returns_A_all_runs = np.array(all_returns_A) 
    returns_B_all_runs = np.array(all_returns_B)

    sns.histplot(returns_A_all_runs, bins=150, ax=axs[0,1], kde=False, stat="density", color='royalblue', alpha=0.7, label=f"Scenario A ($\\nu$={params_A['nu']})", element="step")
    sns.histplot(returns_B_all_runs, bins=150, ax=axs[0,1], kde=False, stat="density", color='crimson', alpha=0.6, label=f"Scenario B ($\\nu$={params_B['nu']})", element="step")
    axs[0, 1].set_title(f"Aggregated Distribution of Daily Returns") # Simpler title
    axs[0, 1].set_xlabel("Daily Return")
    axs[0, 1].set_ylabel("Density")
    axs[0, 1].legend(loc='upper right')
    kurt_A_agg = scipy_kurtosis(returns_A_all_runs, fisher=True) 
    kurt_B_agg = scipy_kurtosis(returns_B_all_runs, fisher=True)
    axs[0, 1].text(0.03, 0.97, f"Agg. Kurtosis A: {kurt_A_agg:.2f}\nAgg. Kurtosis B: {kurt_B_agg:.2f}",
                   transform=axs[0, 1].transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', fc='ivory', alpha=0.8))
    axs[0,1].set_xlim([-0.15, 0.15]) 
    axs[0,1].grid(True)

    markerline_A, stemlines_A, baseline_A = axs[1, 1].stem(lags_for_plot[1:], mean_acf_A[1:], linefmt='royalblue', markerfmt='o', basefmt="gray")
    plt.setp(markerline_A, markersize=3, color='royalblue')
    plt.setp(stemlines_A, linewidth=1, color='royalblue')
    axs[1, 1].set_title(f"Mean ACF of Abs. Returns (Scenario A, {N_runs} runs)")
    axs[1, 1].set_ylabel("Mean ACF")
    axs[1,1].grid(True)

    markerline_B, stemlines_B, baseline_B = axs[2, 1].stem(lags_for_plot[1:], mean_acf_B[1:], linefmt='crimson', markerfmt='o', basefmt="gray")
    plt.setp(markerline_B, markersize=3, color='crimson')
    plt.setp(stemlines_B, linewidth=1, color='crimson')
    axs[2, 1].set_title(f"Mean ACF of Abs. Returns (Scenario B, {N_runs} runs)")
    axs[2, 1].set_xlabel("Lag")
    axs[2, 1].set_ylabel("Mean ACF")
    axs[2,1].grid(True)


    plt.tight_layout(rect=[0, 0.02, 1, 0.95]) 
    plt.savefig('smithian_market_simulation_summary.png', dpi=300, bbox_inches='tight')
    print("Plot saved as smithian_market_simulation_summary.png")

    # Save individual subplots
    subplot_titles = [
        "price_paths", "aggregated_daily_returns",
        "final_price_distribution", "mean_acf_abs_returns_A",
        "avg_cond_vol_distribution", "mean_acf_abs_returns_B"
    ]
    idx = 0
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            ax = axs[i, j]
            # Get the bounding box of the axes in figure coordinates
            bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
            # Add a small padding if desired, though tight_layout should handle most cases
            # bbox = bbox.padded(0.05) # Example padding
            
            filename = f"subplot_{subplot_titles[idx]}.png"
            fig.savefig(filename, dpi=300, bbox_inches=bbox)
            print(f"Subplot saved as {filename}")
            idx += 1

    plt.show()

    print(f"\n--- Aggregated Statistics over {N_runs} runs ---")

    mean_A_agg = np.mean(returns_A_all_runs)
    std_A_agg = np.std(returns_A_all_runs)
    skew_A_agg = skew(returns_A_all_runs)

    mean_B_agg = np.mean(returns_B_all_runs)
    std_B_agg = np.std(returns_B_all_runs)
    skew_B_agg = skew(returns_B_all_runs)

    print("\nScenario A (More Regulated / Smithian Ideal):")
    print(f"  Underlying Fundamental Total Return Target: {fundamental_total_target_return:.2%}")
    print(f"  Avg. Mean Daily Return (across runs): {df_run_stats_A['mean_return'].mean():.8f} (Std of Means: {df_run_stats_A['mean_return'].std():.8f})")
    print(f"  Avg. Daily Std Dev Return (across runs): {df_run_stats_A['std_return'].mean():.6f} (Std of StdDevs: {df_run_stats_A['std_return'].std():.6f})")
    print(f"  Avg. Kurtosis (across runs): {df_run_stats_A['kurtosis'].mean():.2f} (Std of Kurtoses: {df_run_stats_A['kurtosis'].std():.2f})")
    print(f"  Avg. Skewness (across runs): {df_run_stats_A['skewness'].mean():.2f} (Std of Skewnesses: {df_run_stats_A['skewness'].std():.2f})")
    print(f"  Avg. Mean Cond. Vol (across runs): {df_run_stats_A['mean_cond_vol'].mean():.6f} (Std of MeanVols: {df_run_stats_A['mean_cond_vol'].std():.6f})")
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

    avg_daily_ret_A_sim = df_run_stats_A['mean_return'].mean()
    net_10yr_ret_A_sim = (1 + avg_daily_ret_A_sim)**T_periods - 1
    print(f"\nProjected 10-year net return for Scenario A (from sim avg daily mean): {net_10yr_ret_A_sim:.4f} (Fundamental Target ~{fundamental_total_target_return:.2f})")
    print(f"  Mean final price for Scenario A: {mean_price_A_path[-1]:.2f} (Fundamental Target ~{P0_initial*(1+fundamental_total_target_return):.2f})")

    avg_daily_ret_B_sim = df_run_stats_B['mean_return'].mean()
    net_10yr_ret_B_sim = (1 + avg_daily_ret_B_sim)**T_periods - 1
    print(f"Projected 10-year net return for Scenario B (from sim avg daily mean): {net_10yr_ret_B_sim:.4f} (Fundamental Target ~{fundamental_total_target_return:.2f})")
    print(f"  Mean final price for Scenario B: {mean_price_B_path[-1]:.2f} (Fundamental Target ~{P0_initial*(1+fundamental_total_target_return):.2f})")

