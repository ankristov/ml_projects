import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns
from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def run_ttest(control, test):
    
    mean_control = np.mean(control)
    mean_test = np.mean(test)
    diff_mean = mean_test - mean_control
    var_diff_mean = np.var(control)/len(control) + np.var(test)/len(test)
    z_statistics = diff_mean/np.sqrt(var_diff_mean)
    diff_distribution = sps.norm(loc=diff_mean, scale=np.sqrt(var_diff_mean))
    left_bound, right_bound = diff_distribution.ppf([0.025, 0.975])
    ci_length = right_bound - left_bound
    pvalue = 2 * min(diff_distribution.cdf(0), diff_distribution.sf(0))
    effect = diff_mean
    
    return pd.Series({
        'mean_control': mean_control,
        'mean_test': mean_test,
        'z_statistics': z_statistics,
        'pvalue': pvalue,
        'effect': effect,
        'ci_length': ci_length,
        'left_bound': left_bound,
        'right_bound': right_bound
    })


def plot_pvalue_distribution(df):
    fig, ax = plt.subplots(figsize=(12,5))
    fig.patch.set_facecolor('white')
    sns.histplot(x='pvalue_bin', data=df, color='orange',ax=ax)
    plt.xlabel('P-value', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6)
    plt.title('P-value Distribution')
    ax.set_facecolor('white')
    plt.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    plt.show()


def create_pvalue_bins(df):

        x = np.array(df['pvalue'])
        conditions = [
                (0 <= x) & (x < 0.05),
                (0.05 <= x) & (x < 0.1),
                (0.1 <= x) & (x < 0.15),
                (0.15 <= x) & (x < 0.2),
                (0.2 <= x) & (x < 0.25),
                (0.25 <= x) & (x < 0.3),
                (0.3 <= x) & (x < 0.35),
                (0.35 <= x) & (x < 0.4),
                (0.4 <= x) & (x < 0.45),
                (0.45 <= x) & (x < 0.5),
                (0.5 <= x) & (x < 0.55),
                (0.55 <= x) & (x <= 0.6),
                (0.6 <= x) & (x <= 0.65),
                (0.65 <= x) & (x <= 0.7),
                (0.7 <= x) & (x <= 0.75),
                (0.75 <= x) & (x <= 0.8),
                (0.8 <= x) & (x <= 0.85),
                (0.85 <= x) & (x <= 0.9),
                (0.9 <= x) & (x <= 0.95),
                (0.95 <= x) & (x <= 1.0)]
        labels = [
                "[0, 0.05)",
                "[0.05, 0.1]",
                "[0.1, 0.15)",
                "[0.15, 0.2]",
                "[0.2, 0.25)",
                "[0.25, 0.3]",
                "[0.3, 0.35)",
                "[0.35, 0.4]",
                "[0.4, 0.45)",
                "[0.45, 0.5]",
                "[0.5, 0.55)",
                "[0.55, 0.6]",
                "[0.6, 0.65)",
                "[0.65, 0.7]",
                "[0.7, 0.75)",
                "[0.75, 0.8]",
                "[0.8, 0.85)",
                "[0.85, 0.9]",
                "[0.9, 0.95)",
                "[0.95, 1.0]"
                ]
        df['pvalue_bin'] = np.select(conditions, labels, "other")
        df['pvalue_bin'] = pd.Categorical(df['pvalue_bin'], categories=labels, ordered=True)
        
        return df


def plot_exp_mean_distribution(df, bin_limit_left, bin_limit_right):

    df['mean_control_bin'] = pd.cut(df['mean_control'], bins=np.arange(bin_limit_left, bin_limit_right,0.001),right=False)
    df['mean_test_bin'] = pd.cut(df['mean_test'], bins=np.arange(bin_limit_left, bin_limit_right,0.001),right=False)
    df_vis_1 = (
        df.groupby(by='mean_control_bin', observed=False)['mean_control']
        .count()
        .rename("cnt_mean_control")
        .reset_index())
    df_vis_2 = (
        df.groupby(by='mean_test_bin', observed=False)['mean_test']
        .count()
        .rename("cnt_mean_test")
        .reset_index())
    df_vis = pd.merge(left=df_vis_1, 
                      right=df_vis_2, 
                      how='inner', 
                      left_on='mean_control_bin', 
                      right_on='mean_test_bin').drop(columns=['mean_test_bin']).rename({'mean_control_bin':'mean_bin'}, axis=1)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=df_vis.index, y='cnt_mean_control', data=df_vis, color='blue', alpha=0.5)
    sns.barplot(x=df_vis.index, y='cnt_mean_test', data=df_vis, color='red', alpha=0.5)

    ax.set_facecolor('white')
    plt.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    plt.legend().get_frame().set_facecolor('white')
    plt.title('Distribution of Experiment Mean (Conversion Rate)')
    plt.xlabel('Experiment Mean (binned)')
    plt.ylabel('N experiments')

    ticks = [ind for ind in df_vis.index if ind % 2 == 0]
    ticks_labels = df_vis.iloc[ticks, :]['mean_bin'].values
    plt.xticks(ticks=ticks, labels=ticks_labels, rotation=45, fontsize=6)
    plt.yticks(fontsize=8)
    plt.show()


def calculate_sample_size(p1, p2, alpha=0.05, power=0.8):
    """
    Calculate required sample size to achieve a specified power level.
    
    Parameters:
    - p1: Control group probability
    - p2: Test group probability
    - alpha: Significance level (default 0.05)
    - power: Desired power level (default 0.8)
    
    Returns:
    - sample_size: Required sample size per group
    """
    # Calculate effect size
    effect_size = proportion_effectsize(p1, p2)
    
    # Initialize power analysis object
    analysis = NormalIndPower()
    
    # Calculate required sample size per group
    sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided')
    
    return sample_size


def calculate_power(p1, p2, sample_size, alpha=0.05):
    """
    Calculate the achieved power given a sample size.
    
    Parameters:
    - p1: Control group probability
    - p2: Test group probability
    - sample_size: Sample size per group
    - alpha: Significance level (default 0.05)
    
    Returns:
    - power: Achieved power level
    """
    # Calculate effect size
    effect_size = proportion_effectsize(p1, p2)
    
    # Initialize power analysis object
    analysis = NormalIndPower()
    
    # Calculate power with the given sample size
    power = analysis.power(effect_size=effect_size, nobs1=sample_size, alpha=alpha, alternative='two-sided')
    
    return power


def calculate_sample_size_2(p1, p2, alpha=0.05, power=0.8):
    """
    Calculate the required sample size per group for a given alpha and power.
    """
    # Z-scores for alpha and beta (power)
    z_alpha = norm.ppf(1 - alpha / 2)  # two-tailed test
    z_beta = norm.ppf(power)
    
    # Calculate effect size and pooled standard deviation
    pooled_std = ((p1 * (1 - p1) + p2 * (1 - p2)) / 2) ** 0.5
    effect_size = abs(p1 - p2)
    
    # Sample size per group
    n = (((z_alpha + z_beta) * pooled_std) / effect_size) ** 2
    
    return int(n)  # Rounding to nearest whole number for sample size


def calculate_power_2(p1, p2, alpha=0.05, sample_size=1000):
    """Calculate the statistical power given a sample size per group."""
    # Z-score for alpha
    z_alpha = norm.ppf(1 - alpha / 2)  # two-tailed test
    
    # Calculate effect size and pooled standard deviation
    pooled_std = ((p1 * (1 - p1) + p2 * (1 - p2)) / 2) ** 0.5
    effect_size = abs(p1 - p2)
    
    # Calculate z_beta
    z_beta = (effect_size * (sample_size ** 0.5)) / pooled_std - z_alpha
    
    # Convert z_beta to power
    power = norm.cdf(z_beta)
    
    return power


def run_simulation(p1, p2, n_users, n_distributions):


    result_arr = []
    for _ in range(n_distributions):
        control = np.random.binomial(n=1, p=p1, size=n_users)
        test = np.random.binomial(n=1, p=p2, size=n_users)

        result = run_ttest(control, test)
        result_arr.append(result)

    result_df = pd.DataFrame(result_arr)

    result_df['lift'] = (result_df['mean_control'] - result_df['mean_test'])/result_df['mean_control']
    result_df['stat_sign_lift'] = np.where(result_df['pvalue']<=0.05, np.abs(result_df['lift']), np.NaN)
    result_df['is_stat_sign'] = np.where(result_df['pvalue']<=0.05, 1, 0)
    # create bins for pvalue
    result_df = create_pvalue_bins(df=result_df)

    # print summaries
    print('Summaries: ')
    print('N users per variant: ', n_users)
    print('N simulations: ', n_distributions)
    print('CR per control: ', np.round(p1,4))
    print('CR per test: ', np.round(p2,4))
    print('-'*10)
    print('N simulations with stat sign lift: ', np.round(result_df['is_stat_sign'].mean(skipna=False),3))
    print('Max stat sign lift: ', np.round(result_df.query('is_stat_sign == 1')['stat_sign_lift'].max(skipna=True), 3))
    print('Min stat sign lift: ', np.round(result_df.query('is_stat_sign == 1')['stat_sign_lift'].min(skipna=True), 3))
    print('Avg stat sign lift: ', np.round(result_df.query('is_stat_sign == 1')['stat_sign_lift'].mean(skipna=True), 3))

    # plot p-value distribution
    plot_pvalue_distribution(df=result_df)
    # plot experiment mean (conversion rate) distribution
    plot_exp_mean_distribution(df=result_df, bin_limit_left = 0.0, bin_limit_right = 0.1)