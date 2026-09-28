# Transport and Polarization Characteristics of Single Scattering

This post-processing experiment derives transport and polarization metrics
from the existing application JSON outputs. It does not rerun the RCR solver.

## Derived Quantities

- `Qtr = Qsca * (1 - g)`: transport scattering efficiency.
- `PL(theta) = -P12(theta)/P11(theta)`: linear polarization.
- `PC(theta) = P34(theta)/P11(theta)`: circular conversion.
- `max_abs_linear_pol` and angle of maximum linear polarization.
- `linear_pol_at_90deg` and `linear_pol_at_180deg`.
- `max_abs_circular_conversion`.
- `forward_backward_p11_ratio`.
- `g_from_P11`: angular-moment consistency check from `P11`.

## Run

```bat
D:\students\jingjia\gpu_bench_env\Scripts\python.exe experiments\transport_and_polarization_characteristics_of_single_scattering\analyze_transport_polarization.py --results experiments\application\results --out experiments\transport_and_polarization_characteristics_of_single_scattering\output --seeds 0-9
```

If only seed-42 results are available, use:

```bat
D:\students\jingjia\gpu_bench_env\Scripts\python.exe experiments\transport_and_polarization_characteristics_of_single_scattering\analyze_transport_polarization.py --results experiments\application\results --out experiments\transport_and_polarization_characteristics_of_single_scattering\output_seed42 --seeds 42
```

The seed-42 run is useful for testing the script, but it should not be used as
the formal ensemble-statistics result.

## Outputs

- `transport_polarization_by_seed.csv`
- `transport_polarization_ensemble.csv`
- `fig_transport_polarization_scaling.png/.pdf`
- `fig_polarization_angle_characteristics.png/.pdf`
- `transport_polarization_report.md`
