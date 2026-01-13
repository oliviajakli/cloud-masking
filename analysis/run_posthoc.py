from src.posthoc_tests import run_posthoc_wilcoxon, effect_size_cliffs_delta, bootstrap_cliffs_delta
from src.utils.config import load_config
from pathlib import Path
import pandas as pd
import numpy as np
from utils.io import save_csv

config = load_config()

input_data = Path(config["paths"]["input"])
pairs = config["algorithm_pairs"]
random_seed = config["statistics"]["random_seed"]
output_dir = Path(config["paths"]["output_dir"])

df = pd.read_csv(input_data)

def main():
    # Pivot to wide format (scenes x algorithms). Algorithms will be in alphabetical order.
    mcc_wide = df.pivot(index='scene_id', columns='algorithm', values='mcc')
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Run pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction.
    wilcoxon_results = run_posthoc_wilcoxon(mcc_wide, pairs)

    # Calculate Cliff's Delta effect sizes for each pair.
    effect_size_results = effect_size_cliffs_delta(mcc_wide, pairs)

    # Save results of Wilcoxon, Hol-Bonferroni, effect size and bootstrap 95% CI  to CSV file.
    posthoc_df = pd.DataFrame(wilcoxon_results)
    posthoc_df[['algorithm_a', 'algorithm_b']] = posthoc_df['algorithm_pair'].str.split(' vs ', expand=True)
    # Add effect size results to the posthoc dataframe.
    effect_size_df = pd.DataFrame(effect_size_results)
    posthoc_df = posthoc_df.merge(
        effect_size_df
        [['algorithm_a', 'algorithm_b', 'delta', 'effect_size', 'favors']],
        on=['algorithm_a', 'algorithm_b']
    )
    out_path = output_dir / "posthoc_cliffs_delta_results.csv"

    for (a, b) in pairs:
        print(f"\nBootstrap 95% CI for {a} vs {b}")
        ci_lower, ci_upper = bootstrap_cliffs_delta(
            mcc_wide[a],
            mcc_wide[b]
        )
        print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        posthoc_df.loc[
            (posthoc_df["algorithm_a"] == a) & (posthoc_df["algorithm_b"] == b),
            "ci_lower"
        ] = ci_lower
        posthoc_df.loc[
            (posthoc_df["algorithm_a"] == a) & (posthoc_df["algorithm_b"] == b),
            "ci_upper"
        ] = ci_upper

    save_csv(posthoc_df, path=out_path)
    return f"Post-hoc results saved to {out_path}"

if __name__ == "__main__":
    print(main())