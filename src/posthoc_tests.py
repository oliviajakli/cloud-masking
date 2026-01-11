from scipy.stats import wilcoxon
from statsmodels.stats import multitest as smm
import pandas as pd
import numpy as np
from cliffs_delta import cliffs_delta

def run_posthoc_wilcoxon(df, pairs):
    # If significant, run pairwise Wilcoxon
    pvals = []
    for (a, b) in pairs:
        stat, p = wilcoxon(df[a], df[b])
        pvals.append(p)
        print(f"{a} vs {b}: Statistic = {stat}, uncorrected p-value = {p:.4f}")

    # Correct for multiple comparisons (Holm-Bonferroni)
    rej, p_corr, _, _ = smm.multipletests(pvals, alpha=0.05, method='holm')
    for (a, b), p_unc, p_c, r in zip(pairs, pvals, p_corr, rej):
        print(f"{a} vs {b}: corrected p-value = {p_c:.4f}, reject = {r}")
    results_df = pd.DataFrame({
        'algorithm_pair': [f"{a} vs {b}" for (a, b) in pairs],
        'uncorrected_p_value': p_unc,
        'corrected_p_value': p_corr,
        'reject_null': rej
    })
    return results_df

# Pairwise effect sizes with bootstrap CIs.
def effect_size_cliffs_delta(df, pairs):
    """
    Calculate Cliff's Delta for two algorithms a and b.
    a, b should be 1D numpy arrays of equal length.
    Returns delta and magnitude string.
    """
    results = []
    for (a, b) in pairs:
        delta, magnitude = cliffs_delta(df[a], df[b])
        results.append({
            "algorithm_a": a,
            "algorithm_b": b,
            "delta": delta,
            "effect_size": magnitude,
            "favors": a if delta > 0 else b if delta < 0 else "Neither"
        })
    return results

def bootstrap_cliffs_delta(x, y, n_boot=5000, ci=95):
    """
    Calculate Cliff's Delta for all pairwise comparisons among algorithms,
    bootstrap CI, and full distribution. x, y should be 1D numpy arrays of
    equal length.
    """
    boot_deltas = []
    n = len(x)

    # Convert Series to numpy arrays to ensure positional indexing for bootstrap.
    x_np = x.values
    y_np = y.values

    for _ in range(n_boot):
        indices = np.random.choice(n, n, replace=True)
        x_boot = x_np[indices]
        y_boot = y_np[indices]
        delta, _ = cliffs_delta(x_boot, y_boot)
        boot_deltas.append(delta)
    # Build confidence intervals.
    alpha = (100 - ci) / 2
    lower = np.percentile(boot_deltas, alpha)
    upper = np.percentile(boot_deltas, 100 - alpha)

    return lower, upper