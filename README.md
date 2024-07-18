[![DOI](https://zenodo.org/badge/733696122.svg)](https://zenodo.org/doi/10.5281/zenodo.12667032)

# Dome-C-Sensitivity
Sensitivity to changes in the GCR flux using carbon-14 profiles at Dome C, Antarctica.

## Setup Steps
1. Create Folders
models
matlab/burst_models_past
matlab/const_models_2sigma_hull
matlab/delta_fast_models
matlab/delta_neg_models
matlab/linear_models_past
matlab/linear_models_pres
matlab/step_models_past
matlab/step_models_pres

2. Generate 14CO Profiles
Run Matlabs:
Deep_Ice_14CO_age_burst.m
Deep_Ice_14CO_age_constant.m
Deep_Ice_14CO_age_delta_fast.m
Deep_Ice_14CO_age_delta_neg.m
Deep_Ice_14CO_age_linear_past.m
Deep_Ice_14CO_age_linear_pres.m
Deep_Ice_14CO_age_step_past.m
Deep_Ice_14CO_age_step_pres.m

3. Convert Profiles to FITS format
Run csv-to-FITS.ipynb

4. Calculate Null Model Bayes Factors
Run null-generator.py

## iPython Notebooks
gcr-sensitivity : main sensitivity calculation
graph-maker : produces graphs illustrating each step of the calculation process