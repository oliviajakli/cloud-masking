from scipy.stats import wilcoxon    # type: ignore
import statsmodels.stats.multitest as smm   # type: ignore
import pandas as pd     # type: ignore
import numpy as np   # type: ignore
from cliffs_delta import cliffs_delta   # type: ignore
import logging

logger = logging.getLogger(__name__)

def run_posthoc_wilcoxon(df: pd.DataFrame, pairs: list) -> pd.DataFrame:
    """Generate pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction.
    Args:
        df (pd.DataFrame): DataFrame with algorithm results.
        pairs (list): List of tuples with algorithm name pairs to compare.
    Returns:
        pd.DataFrame: DataFrame with pairwise comparison results.
    """
    logger.info("Running pairwise Wilcoxon signed-rank tests.")
    # If results of Friedman test are significant, run pairwise Wilcoxon.
    pvals = []
    for (a, b) in pairs:
        logger.debug(f"Comparing {a} and {b}.")
        stat, p = wilcoxon(df[a], df[b])
        pvals.append(p)
        logger.info(f"{a} vs {b}: Statistic = {stat}, uncorrected p-value = {p:.4f}")

    # Correct for multiple comparisons using Holm-Bonferroni, a stepwise procedure 
    # which adaptively adjusts thresholds. More powerful and less conservative than 
    # plain Bonferroni, while still controlling for family-wise error rate.
    rej, p_corr, _, _ = smm.multipletests(pvals, alpha=0.05, method='holm')
    for (a, b), p_unc, p_c, r in zip(pairs, pvals, p_corr, rej):
        logger.info(f"{a} vs {b}: corrected p-value = {p_c:.4f}, reject = {r}")
    results_df = pd.DataFrame({
        'algorithm_pair': [f"{a} vs {b}" for (a, b) in pairs],
        'uncorrected_p_value': pvals,
        'corrected_p_value': p_corr,
        'reject_null': rej
    })
    logger.info("Pairwise Wilcoxon tests completed.")
    logger.debug(f"Pairwise Wilcoxon results:\n{results_df}")
    return results_df

# Pairwise effect sizes with bootstrap CIs.
def effect_size_cliffs_delta(df: pd.DataFrame, pairs: list) -> list:
    """
    Calculate Cliff's Delta for two algorithms a and b.
    a, b should be 1D numpy arrays of equal length.
    Returns delta and magnitude string.
    Args:
        df (pd.DataFrame): DataFrame with algorithm results.
        pairs (list): List of tuples with algorithm name pairs to compare.
    Returns:
        list: List of dictionaries with effect size results for each pair.
    """
    logger.info("Calculating Cliff's Delta for pairwise comparisons.")
    results = []
    for (a, b) in pairs:
        logger.debug(f"Calculating Cliff's Delta for {a} and {b}.")
        delta, magnitude = cliffs_delta(df[a], df[b])
        results.append({
            "algorithm_a": a,
            "algorithm_b": b,
            "delta": delta,
            "effect_size": magnitude,
            "favors": a if delta > 0 else b if delta < 0 else "Neither"
        })
        logger.info(f"{a} vs {b}: Cliff's Delta = {delta:.4f}, effect size = {magnitude}")
    logger.info("Cliff's Delta calculations completed.")
    return results

def bootstrap_cliffs_delta(x: pd.Series, y: pd.Series, n_boot: int = 5000, ci: int = 95) -> tuple:
    """
    Calculate Cliff's Delta for all pairwise comparisons among algorithms,
    bootstrap CI, and full distribution. x, y should be 1D numpy arrays of
    equal length.
    Args:
        x (pd.Series): Results for algorithm x.
        y (pd.Series): Results for algorithm y.
        n_boot (int): Number of bootstrap samples.
        ci (int): Confidence interval percentage.
    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    logger.info("Calculating bootstrap CI for Cliff's Delta between two algorithms.")
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
    logger.info(f"Bootstrap CI for Cliff's Delta: {ci}% CI = [{lower:.4f}, {upper:.4f}]")
    return lower, upper