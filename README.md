## Examining the effect of cloud masking algorithms on remote sensing data accuracy and processing to improve water quality in Sandusky Bay, Ohio

### Motivation and problem statement

Cloud masking is a crucial preprocessing step in the KSU VPCA spectral decomposition method, which lends itself to the identification of possible contaminants in water bodies through scientific analysis. Especially useful for regions with frequent cloud cover, the ability to accurately detect and remove clouds and the shadows they cast can greatly increase the availability of usable data for such purposes.

This project evaluates and compares three prominent cloud masking algorithms using sample-level classification metrics, nonparametric hypothesis testing, effect sizes, and bootstrap inference. A reproducible, statistically robust framework for comparing cloud masking methods is established.

### Key Contributions
- End-to-end reproducible pipeline for evaluating cloud masking algorithms
- Sample-level confusion matrix construction and metric computation
- Nonparametric statistical comparison using the Friedman test
- Post-hoc pairwise comparisons with multiple-testing correction
- Pairwise effect size estimation (e.g., Cliff’s delta)
- Bootstrap confidence intervals for inference robustness

### Data
- Satellite imagery: Sentinel-2 Level-2A surface reflectance collection, accessible [here](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#bands).
- Spatial resolution: 10 m
- Temporal coverage: 2020-2025
- Reference (ground truth) masks: manually labeled and augmented with AI (a gradient boosted decision tree)

