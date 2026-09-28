# Formal seed 0-9 ensemble

This directory contains the JSON result files used for the paper's ensemble
statistics. The intended formal set has one file for every combination of:

- family: `bcca` and `bpca`
- aggregate size: `N = 8, 16, 32, 64`
- wavelength: `450`, `550`, and `650` nm where defined by the paper cases
- seed: `0` through `9`

Files containing `seed42` or names such as `seed42_seed0` are historical
demonstration/batch artifacts and must not be included in the formal seed
filter. The analysis scripts filter by the `input_parameters.seed` field as a
second check.
