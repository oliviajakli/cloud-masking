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

## Results
The analysis consistently identified the following ranking: Cloud Score+ > s2cloudless > Hybrid

Bootstrapped median MCC:

| Algorithm | Median MCC | 95% CI |
| ----------- | ----------- | ----------- |
| Hybrid | 0.323 | 0.207 – 0.402 |
| s2cloudless | 0.402 | 0.261 – 0.451 |
| Cloud Score+ | 0.539 | 0.330 – 0.605 |

### Cloud Score+
Cloud Score+ achieved the strongest overall performance across nearly all analyses:
- Highest median MCC
- Highest median F1-score
- Highest median IoU
- Lowest variability across scenes

This algorithm demonstrated strong agreement with the reference masks while maintaining a relatively balanced tradeoff between cloud omission and cloud commission errors.

### s2cloudless
s2cloudless performed consistently well and closely trailed Cloud Score+.

Strengths included:
- Reliable cloud detection
- Strong performance in heavily cloudy scenes
- Conservative cloud removal behavior

However, it tended to aggressively mask clouds, producing more false positives and reducing usable image area.

### Hybrid method
The hybrid Sen2Cor-based method showed the weakest and least stable performance.

Observed weaknesses included:
- Failure to detect thin and sparse clouds
- Poor cloud shadow detection
- High variability between scenes
- Multiple catastrophic failures in low-cloud conditions

Although it occasionally performed adequately under simple cloudy conditions, it was not sufficiently robust for operational freshwater remote sensing workflows.

## Reproducibility
The statistical analysis pipeline in this repository is fully reproducible using the included configuration files, algorithm output masks, and reference masks.
Reference (ground-truth) cloud masks were manually generated using the [IRIS](https://github.com/ESA-PhiLab/iris/tree/master#) software package through expert visual interpretation of satellite imagery. Because this process involves human annotation and interactive labeling decisions, the exact mask-generation process is not fully reproducible programmatically.
To support reproducibility of the downstream analysis, all finalized reference masks and algorithm-generated masks used in the study are included in this repository.

### Environment
It is recommended to do this inside a virtual environment.

- Python 3.12.5
- Key dependencies listed in `requirements.txt`

Environment Setup
1. Clone the repository
2. Configure paths and parameters in ‘config/config.yaml’
3. Execute ‘runner.py’ to reproduce all analyses and figures

Running:

python -m pipeline.runner or python pipeline/runner.py

will reproduce:
- Evaluation metrics
- Descriptive statistics
- Bootstrap confidence intervals
- Friedman test results
- Post-hoc comparisons
- Effect sizes
- Exploratory figures
- Final analysis plots

<br>

>git clone git@github.com:oliviajakli/cloud-masking.git\
cd pipeline\
pip install -r requirements.txt\
python -m runner **or** python runner.py

## Limitations
Several important limitations should be considered when interpreting these results.

### Imperfect Reference Masks
Ground-truth masks were generated through human interpretation with AI assistance using IRIS. Although carefully created, they still contain uncertainty and labeling errors, especially around:
- Bright reflective surfaces
- Thin cloud boundaries
- Water reflections
- Mixed pixels

These imperfections likely contributed to some disagreement between algorithms and reference data.

### Limited Spatial and Temporal Scope
The study focused exclusively on:
- One geographic region (Sandusky Bay, Ohio)
- 15 scenes
- Selected months between 2020–2025

As a result, conclusions may not fully generalize to:
- Other climates
- Different land-cover types
- Snow/ice environments
- Tropical cloud regimes

### Binary Classification Only
Cloud detection was treated as a binary problem:
- cloud
- clear

More detailed semantic classes such as:
- cloud shadow
- thin cloud
- cirrus cloud
- haze

were not evaluated separately.

### Limited Hyperparameter Tuning
Only one threshold configuration was tested for each algorithm.

Additional tuning of:
- cloud probability thresholds
- cloud buffers
- shadow projection distances
- NIR thresholds

could potentially improve performance.

## Future Work
Several extensions could improve the robustness and applicability of this research.

### Improve Reference Mask Quality
Future studies could refine IRIS-generated masks through:
- additional manual correction
- multi-annotator validation
- integration with benchmark datasets such as CloudSEN12+

Improved ground truth would reduce uncertainty in evaluation metrics.

### Expand Geographic and Temporal Coverage
Future work should include:
- multiple freshwater systems
- different climate regions
- additional seasons
- larger sample sizes

### Multi-class Cloud Segmentation
Extending the analysis beyond binary classification would allow more detailed evaluation of:
- cloud shadows
- cirrus clouds
- thin haze
- thick opaque clouds

This would better reflect real-world atmospheric complexity.

### Hyperparameter Optimization
Algorithm-specific thresholds and parameters should be systematically optimized for different downstream applications such as:
- water-quality retrieval
- vegetation monitoring
- land-cover mapping
- composite image generation

### Composite Image Analysis
A future extension could evaluate the impact of cloud masking quality on downstream composite imagery products and environmental monitoring workflows, especially for harmful algal bloom detection and freshwater quality assessment.

## Citation
If you use this code or methodology, please cite:

Olivia Jákli, 2026. *Examining the effect of cloud masking algorithms on remote sensing data accuracy and processing to improve water quality in Sandusky Bay, Ohio*.