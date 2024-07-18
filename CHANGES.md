# Dome C Sensitivity Change Log

## v1.1.0

### Major Updates

* Recompute sensitivities by marginalizing over muon interaction rates for both null and alternative hypotheses ([PR #2](https://github.com/14CO/Dome-C-Sensitivity/pull/2)).

### Minor Updates

* Factorize GCR calculation and plotting into two notebooks ([PR #2](https://github.com/14CO/Dome-C-Sensitivity/pull/2)).
* Added option to fix changes to the flux in the present or the past ([PR #2](https://github.com/14CO/Dome-C-Sensitivity/pull/2)).
* Reworked the calculation to enable parallel processing of many profile realizations ([PR #2](https://github.com/14CO/Dome-C-Sensitivity/pull/2)).

## v1.0.0

Initial release. Uses a simple integrator to construct carbon-14 profiles in horizontally uniform ice at Dome C. Then constructs a test statistic using the Bayes Factor to test for variability in the cosmic-ray flux producing the carbon-14.
