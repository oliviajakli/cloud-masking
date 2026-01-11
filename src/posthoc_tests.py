from scipy.stats import wilcoxon
from statsmodels.stats import multitest as smm
from pathlib import Path
import pandas as pd
from src.utils.io import save_csv

def run_posthoc_wilcoxon(df, pairs):
    # Pivot to wide format (scenes x algorithms). Algorithms will be in alphabetical order.
    mcc_wide = df.pivot(index='scene_id', columns='algorithm', values='mcc')
    # If significant, run pairwise Wilcoxon
    pvals = []
    for (a, b) in pairs:
        stat, p = wilcoxon(mcc_wide[a], mcc_wide[b])
        pvals.append(p)
        print(f"{a} vs {b}: Statistic = {stat}, uncorrected p-value = {p:.4f}")

    # Correct for multiple comparisons (Holm-Bonferroni)
    rej, p_corr, _, _ = smm.multipletests(pvals, alpha=0.05, method='holm')
    for (a, b), p_unc, p_c, r in zip(pairs, pvals, p_corr, rej):
        print(f"{a} vs {b}: corrected p-value = {p_c:.4f}, reject = {r}")
    # Save results to CSV file.
    results_df = pd.DataFrame({
        'algorithm_pair': [f"{a} vs {b}" for (a, b) in pairs],
        'uncorrected_p_value': pvals,
        'corrected_p_value': p_corr,
        'reject_null': rej
    })
    save_csv(results_df, path=Path("results/posthoc_wilcoxon_results.csv"))
    return results_df