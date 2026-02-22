import pandas as pd

from pipeline.runner import run_pipeline


def test_full_pipeline(tmp_path, mocker):
    fake_config = {
        "paths": {
            "input": str(tmp_path / "input.csv"),
            "output_dir": str(tmp_path),
        },
        "algorithm_pairs": {"A": ["B"]},
        "validation": {
            "required_columns": ["algorithm"],
            "metric_columns": ["mcc"]
        },
    }

    df = pd.DataFrame({
        "algorithm": ["A", "B"],
        "mcc": [0.8, 0.9],
    })

    df.to_csv(tmp_path / "input.csv", index=False)

    mocker.patch("pipeline.runner.load_config", return_value=fake_config)

    def touch(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok")

    def fake_run_evaluation(*args, **kwargs):
        touch(tmp_path / "per_scene_evaluation_metrics.csv")
        touch(tmp_path / "A_confusion_matrix_scene_1.png")

    def fake_run_exploratory(*args, **kwargs):
        touch(tmp_path / "metrics_summary.csv")
        touch(tmp_path / "descriptives" / "mcc_distribution.png")
        touch(tmp_path / "boxplots" / "mcc_boxplot.png")
        touch(tmp_path / "paired_differences" / "mcc_paired_difference.png")
        touch(tmp_path / "bland_altman" / "bland_altman_A_vs_B_cloud_fraction.png")
        touch(tmp_path / "error_maps" / "error_map_A_202506.tif")
        touch(tmp_path / "scatterplots" / "cloud_fraction_scatter_mcc.png")
        touch(tmp_path / "time_series" / "time_series_mcc_by_algorithm.png")

    def fake_run_normality_tests(*args, **kwargs):
        touch(tmp_path / "hypothesis" / "shapiro_wilk.csv")
        touch(tmp_path / "hypothesis" / "pairwise_mcc_differences.csv")
        touch(tmp_path / "hypothesis" / "histogram_kde_mcc_diff_A_vs_B.png")

    def fake_run_friedman(*args, **kwargs):
        touch(tmp_path / "hypothesis" / "friedman_test_results.csv")

    def fake_run_posthoc(*args, **kwargs):
        touch(tmp_path / "posthoc" / "posthoc_cliffs_delta_results.csv")

    def fake_run_bootstrap(*args, **kwargs):
        touch(tmp_path / "bootstrap" / "algorithm_summary.csv")
        touch(tmp_path / "bootstrap" / "pairwise_diff_summary.csv")
        touch(tmp_path / "bootstrap" / "alg_bootstrap_raw.csv")
        touch(tmp_path / "bootstrap" / "pairwise_diffs_raw.csv")

    def fake_run_directional_error(*args, **kwargs):
        touch(tmp_path / "directional_bias" / "directional_error_summary.csv")
        touch(tmp_path / "directional_bias" / "directional_bias_violinplot.png")

    mocker.patch("pipeline.runner.run_evaluation", side_effect=fake_run_evaluation)
    mocker.patch("pipeline.runner.run_exploratory", side_effect=fake_run_exploratory)
    mocker.patch("pipeline.runner.run_normality_tests", side_effect=fake_run_normality_tests)
    mocker.patch("pipeline.runner.run_friedman", side_effect=fake_run_friedman)
    mocker.patch("pipeline.runner.run_posthoc", side_effect=fake_run_posthoc)
    mocker.patch("pipeline.runner.run_bootstrap", side_effect=fake_run_bootstrap)
    mocker.patch("pipeline.runner.run_directional_error", side_effect=fake_run_directional_error)
    
    run_pipeline()

    # Assert expected files created
    assert (tmp_path / "per_scene_evaluation_metrics.csv").exists()
    assert (tmp_path / "A_confusion_matrix_scene_1.png").exists()
    assert (tmp_path / "metrics_summary.csv").exists()
    assert (tmp_path / "descriptives" / "mcc_distribution.png").exists()
    assert (tmp_path / "boxplots" / "mcc_boxplot.png").exists()
    assert (tmp_path / "paired_differences" / "mcc_paired_difference.png").exists()
    assert (
        tmp_path / "bland_altman" / "bland_altman_A_vs_B_cloud_fraction.png").exists()
    assert (tmp_path / "error_maps" / "error_map_A_202506.tif").exists()
    assert (tmp_path / "scatterplots" / "cloud_fraction_scatter_mcc.png").exists()
    assert (tmp_path / "time_series" / "time_series_mcc_by_algorithm.png").exists()
    assert (tmp_path / "hypothesis" / "shapiro_wilk.csv").exists()
    assert (tmp_path / "hypothesis" / "pairwise_mcc_differences.csv").exists()
    assert (tmp_path / "hypothesis" / "histogram_kde_mcc_diff_A_vs_B.png").exists()
    assert (tmp_path / "hypothesis" / "friedman_test_results.csv").exists()
    assert (tmp_path / "posthoc" / "posthoc_cliffs_delta_results.csv").exists()
    assert (tmp_path / "bootstrap" / "algorithm_summary.csv").exists()
    assert (tmp_path / "bootstrap" / "pairwise_diff_summary.csv").exists()
    assert (tmp_path / "bootstrap" / "alg_bootstrap_raw.csv").exists()
    assert (tmp_path / "bootstrap" / "pairwise_diffs_raw.csv").exists()
    assert (tmp_path / "directional_bias" / "directional_error_summary.csv").exists()
    assert (tmp_path / "directional_bias" / "directional_bias_violinplot.png").exists()
