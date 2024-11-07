import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon
from scipy.stats.mstats import winsorize


def calculate_metrics_per_user(group):
    """
    Function is applyed to groupped dataframe to calculate per group (user) summaries.

    Parameters
    ----------
    group : _type_
        Result of applying .groupby to a dataframe.

    Returns
    -------
    pd.Series
        Containing summaries per group (user)
    """
    # count steps before registration date
    cnt_steps_before = group.query('steps_date < reg_date')['steps'].sum()
    # count steps after registration date
    cnt_steps_after = group.query('steps_date >= reg_date')['steps'].sum()
    # count days (records) before registration date
    cnt_days_before = group.query('steps_date < reg_date')['steps_date'].nunique()
    # count days (records) after registration date
    cnt_days_after = group.query('steps_date >= reg_date')['steps_date'].nunique()
   
    return pd.Series({'cnt_days_before': cnt_days_before, 
                      'cnt_days_after': cnt_days_after, 
                      'cnt_steps_before': cnt_steps_before,
                      'cnt_steps_after': cnt_steps_after})


def vis_distributions(df):
    """
    Visualize distribution of the two metrics and boxplots.

    Parameters
    ----------
    df : pd.Dataframe
        A dataframe with two columns representing our metrics.
    params: dict
        A dictionary with params for visualization
    """
    assert df.shape[1] == 2, 'df MUST be a dataframe with two columns.'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    col1, col2 = df.columns

    # KDE plots on the first axes (ax1)
    sns.kdeplot(df[col1], fill=True, color="skyblue", alpha=0.5, label='before registration', ax=ax1)
    sns.kdeplot(df[col2], fill=True, color="red", alpha=0.5, label='after registration', ax=ax1)
    ax1.set_xlabel('Average Steps per Day per User')


    # Boxplots on the second axes (ax2)
    sns.boxplot(x='Registration Status', y='Average Steps', 
                data=pd.melt(df, value_vars=[col1, col2], var_name='Registration Status', value_name='Average Steps'), 
                hue='Registration Status',
                palette={col1: 'skyblue', col2: 'red'},
                ax=ax2)
    ax2.set_xlabel('Registration Status')
    ax2.set_ylabel('Average Steps per Day per User')
    
    # get handles for figure level legend
    handles = [
        plt.Line2D([0], [0], color='skyblue', lw=4, label='Before Registration'),
        plt.Line2D([0], [0], color='red', lw=4, label='After Registration'),
    ]
    
    fig.legend(handles, labels=['Before Registration', 'After Registration'], loc='center', bbox_to_anchor=(0.5, -0.05), ncol=2, title='Registration Status')


    # Set the overall title for the figure
    plt.suptitle('Comparison of Average Steps per Day Before and After Registration', fontsize=16)

    # Show the plots
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust top to make space for the suptitle
    plt.show()


def vis_trends(df):
    """
    Visualize trends per sampled user.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with columns 'user_id', 'steps', 'days_diff'
    """
    # sample 100 users for visualization
    user_ids_for_vis = np.random.choice(df['user_id'].unique(), size=100, replace=False)
    df_vis = df[df['user_id'].isin(user_ids_for_vis)]
    # visualize trends for users

    plt.figure(figsize=(20,8))
    sns.lineplot(x='days_diff', y='steps', data=df_vis, color='red')

    plt.axvline(x=0, color='grey', ls='--')
    plt.title('Count of steps before/after registration (users sample)')
    plt.xlabel('Days before/after')
    plt.ylabel('Steps')
    plt.show()


def run_stat_test(var1, var2, remove_outliers=False, verbose=False):
    """
    Calculate statistical tests (paired t-test and wilcoxon) for two variables.

    Parameters
    ----------
    var1 : pd.Series
        First variable.
    var2 : pd.Series
        Second Variable.
    remove_outliers : bool, optional
        If to apply outlier removing technique.
    verbose : bool, optional
        If to show some additional meta data in console (helpful for debugging).
    """

    if verbose:
        avg_steps_before = var1.mean()
        avg_steps_after = var2.mean()
        std_steps_before = var1.std()
        std_steps_after = var2.std()
        print('Descriptive statistics for var1, var2:')
        print(f'var1: mean ={avg_steps_before}, std={std_steps_before}')
        print(f'var2: mean = {avg_steps_after}, std={std_steps_after}')

    if remove_outliers:
        # Winsorize both vars
        var1 = winsorize(var1, limits=[0.05, 0.05])
        var2 = winsorize(var2, limits=[0.05, 0.05])
        if verbose:
            avg_steps_before = var1.mean()
            avg_steps_after = var2.mean()
            std_steps_before = var1.std()
            std_steps_after = var2.std()
            print('Descriptive statistics for var1, var2 (after winsorize): ')
            print(f'var1: mean ={avg_steps_before}, std={std_steps_before}')
            print(f'var2: mean = {avg_steps_after}, std={std_steps_after}')


    # Perform paired t-test
    t__stat, t__p_value = ttest_rel(var1, var2, alternative='two-sided')
    # Perform Wilcoxon signed-rank test
    w__stat, w__p_value = wilcoxon(var1, var2)

    # Interpret results
    alpha = 0.05
    print()
    print('Result (t-test):')
    print(f'T-test statistic: {t__stat}, p-value: {t__p_value}')
    if t__p_value < alpha:
        print("There is a significant difference in physical activity before and after registration.")
    else:
        print("There is no significant difference in physical activity before and after registration.")

    print()
    print('Result (wilcoxon):')
    print(f"Wilcoxon test statistic: {w__stat}, p-value: {w__p_value}")
    if w__p_value < alpha:
        print("There is a significant difference in physical activity before and after registration.")
    else:
        print("There is no significant difference in physical activity before and after registration.")