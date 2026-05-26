# Examining the effect of cloud masking algorithms on remote sensing data accuracy and processing to improve water quality in Sandusky Bay, Ohio

## Motivation and problem statement

Cloud masking is a crucial preprocessing step in the KSU VPCA spectral decomposition method, which lends itself to the identification of possible contaminants in water bodies through scientific analysis. Especially useful for regions with frequent cloud cover, the ability to accurately detect and remove clouds and the shadows they cast can greatly increase the availability of usable data for such purposes.

This project evaluates and compares three prominent cloud masking algorithms using sample-level classification metrics, nonparametric hypothesis testing, effect sizes, and bootstrap inference. A reproducible, statistically robust framework for comparing cloud masking methods is established.

## Key Contributions
- End-to-end reproducible pipeline for evaluating cloud masking algorithms
- Sample-level confusion matrix construction and metric computation
- Nonparametric statistical comparison using the Friedman test
- Post-hoc pairwise comparisons with multiple-testing correction
- Pairwise effect size estimation (e.g., Cliff’s delta)
- Bootstrap confidence intervals for inference robustness


## Methodology Overview
The following cloud masking approaches were compared:
- **Hybrid (Sen2Cor-based)** — a custom method combining the MSK_CLDPRB cloud probability layer with MSK_CLASSI opaque and cirrus cloud classification bands.
- **s2cloudless** — a lightweight machine learning cloud detection algorithm based on a LightGBM model.
- **Cloud Score+** — a weakly supervised deep learning approach using temporal information and probabilistic quality assessment bands.

The analysis used the [Harmonized Sentinel-2 Level-2A surface reflectance dataset in Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#bands). Fifteen scenes from 2020–2025 were selected across spring, summer, and fall months to capture a wide range of cloud conditions over Sandusky Bay.

To ensure fair comparison across algorithms:
- All bands were resampled to 10 m spatial resolution
- Continuous reflectance/probability layers used bilinear interpolation
- Categorical mask layers used nearest-neighbor resampling
- Each algorithm was applied to identical spatial and temporal inputs

Reference (“ground truth”) masks were manually generated using the ESA-PhiLab IRIS active-learning segmentation tool. Pixels were classified into binary classes:
- clear
- cloud

Cloud shadows, cirrus clouds, and opaque clouds were grouped into the cloud class because even subtle atmospheric contamination can significantly impact freshwater analysis workflows.

Performance evaluation was conducted at the scene level rather than the pixel level to avoid inflated statistical significance caused by strong spatial autocorrelation among neighboring pixels.

Key evaluation metrics included:
- Matthew’s Correlation Coefficient (MCC)
- F1-score
- Intersection over Union (IoU / Jaccard Score)
- Balanced Accuracy
- Precision
- Recall

MCC was selected as the primary evaluation metric due to its robustness to class imbalance and its balanced use of all confusion matrix components.

## Statistical Analysis Details
The statistical framework was designed to evaluate both overall algorithm performance and systematic error behavior.

### Primary hypothesis testing
The primary hypothesis tested whether significant differences existed among the three algorithms when measured using scene-level MCC values. 
Exploratory analysis and Shapiro-Wilk testing showed that pairwise MCC differences were non-normally distributed. Accordingly, the analysis proceeded using nonparametric methods.

### Friedman Test
A Friedman repeated-measures test was used to compare the three algorithms across the same 15 scenes.
- Test statistic: χ² = 22.933
- Degrees of freedom: 2
- p-value = 0.0001

This result indicated statistically significant differences among the algorithms.

### Post-hoc Pairwise Comparisons
After rejecting the null hypothesis, pairwise comparisons were conducted using:
- Wilcoxon signed-rank tests
- Holm-Bonferroni correction to control familywise error rate

| Comparison | Adjusted p-value |
| ----------- | ----------- |
| Hybrid vs s2cloudless | 0.0085 |
| Hybrid vs Cloud Score+ | 0.0002 |
| s2cloudless vs Cloud Score+ | 0.0085 |

All pairwise comparisons remained statistically significant after correction.

### Effect Size Analysis
To quantify practical significance, Cliff’s Delta was used as a robust nonparametric effect size measure.

| Comparison | Cliff's Delta | Interpretation |
| ----------- | ----------- | ----------- |
| Hybrid vs s2cloudless | -0.502 | Large effect |
| Hybrid vs Cloud Score+ | -0.573 | Large effect |
| s2cloudless vs Cloud Score+ | -0.333 | Medium effect |

Bootstrapped 95% confidence intervals were also computed for all comparisons.

### Bootstrap Inference
A hierarchical tile-based bootstrap approach was implemented to preserve both spatial and scene-level dependence structures.

Bootstrap design:
- Tile size: 256 x 256 pixels
- 1,000 tile-level resamples per scene
- 2,000 global paired resamples across scenes

Median MCC was used as the primary bootstrap statistic because it is more robust to outlier scenes and heterogeneous cloud conditions than the mean.