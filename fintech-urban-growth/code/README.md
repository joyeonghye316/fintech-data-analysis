# Code Guide

## Environment
- Python version: 3.16
- Main packages: pandas, numpy, statsmodels/linearmodels, matplotlib

## Recommended Run Order

1. `01_data_cleaning/` scripts  
   - `00_datacheck.py`: verify the structure, consistency, and key variables of the city-year panel dataset.

2. `02_analysis/` scripts  
   - `01_descriptive_statistics.py`: generate summary statistics and descriptive figures.

3. `03_regression/` scripts  
   - `02_baseline_regression.py`: estimate the baseline fixed-effects model.  
   - `03_robustness_checks.py`: conduct robustness checks (lags/leads, sample exclusions).  
   - `04_heterogeneity_analysis.py`: perform heterogeneity analysis across regions and firm structures.  
   - `05_marginal_effects.py`: compute and visualize marginal effects and turning points.

## Outputs
- Regression tables are saved to `results/tables/`
- Figures are saved to `results/figures/`

## Notes
- Update file paths in the first cell/section of each script if needed.
- Some raw data may not be included; see `data/README.md`.
