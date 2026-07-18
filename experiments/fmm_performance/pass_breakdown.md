# Pass-Level Execution Time Breakdown (p=3)

| N | Solver / Pass | Time (ms) | Fraction of Total |
|---|---------------|-----------|-------------------|
| **128** | | | |
| | Direct JIT kernel/matvec | 2.535 ms | 100.0% |
| | **FMM Speed (Total)** | **87.370 ms** | 100.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- allocation/zero-fill | 0.025 ms | 0.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- input padding | 0.057 ms | 0.1% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- P2M | 7.643 ms | 8.7% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- M2M | 0.190 ms | 0.2% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- M2L | 62.459 ms | 71.5% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- L2L | 0.195 ms | 0.2% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- L2P | 7.538 ms | 8.6% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- near field | 9.122 ms | 10.4% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- output assembly | 0.141 ms | 0.2% |
| | **FMM Reference (Total)** | **520.933 ms** | 100.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- allocation/zero-fill | 0.055 ms | 0.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- input padding | 0.058 ms | 0.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- P2M | 51.340 ms | 9.9% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- M2M | 1.494 ms | 0.3% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- M2L | 407.811 ms | 78.3% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- L2L | 1.186 ms | 0.2% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- L2P | 49.739 ms | 9.5% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- near field | 9.110 ms | 1.7% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- output assembly | 0.140 ms | 0.0% |
| **256** | | | |
| | Direct JIT kernel/matvec | 9.881 ms | 100.0% |
| | **FMM Speed (Total)** | **198.892 ms** | 100.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- allocation/zero-fill | 0.054 ms | 0.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- input padding | 0.113 ms | 0.1% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- P2M | 13.731 ms | 6.9% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- M2M | 7.990 ms | 4.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- M2L | 119.587 ms | 60.1% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- L2L | 8.184 ms | 4.1% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- L2P | 13.453 ms | 6.8% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- near field | 35.541 ms | 17.9% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- output assembly | 0.238 ms | 0.1% |
| | **FMM Reference (Total)** | **1966.089 ms** | 100.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- allocation/zero-fill | 0.131 ms | 0.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- input padding | 0.118 ms | 0.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- P2M | 433.712 ms | 22.1% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- M2M | 62.090 ms | 3.2% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- M2L | 909.733 ms | 46.3% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- L2L | 60.737 ms | 3.1% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- L2P | 419.178 ms | 21.3% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- near field | 79.752 ms | 4.1% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- output assembly | 0.637 ms | 0.0% |
| **512** | | | |
| | Direct JIT kernel/matvec | 36.722 ms | 100.0% |
| | **FMM Speed (Total)** | **1237.322 ms** | 100.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- allocation/zero-fill | 0.122 ms | 0.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- input padding | 0.233 ms | 0.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- P2M | 101.267 ms | 8.2% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- M2M | 33.793 ms | 2.7% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- M2L | 555.873 ms | 44.9% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- L2L | 151.102 ms | 12.2% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- L2P | 98.824 ms | 8.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- near field | 295.638 ms | 23.9% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- output assembly | 0.470 ms | 0.0% |
| | **FMM Reference (Total)** | **17030.838 ms** | 100.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- allocation/zero-fill | 0.301 ms | 0.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- input padding | 0.234 ms | 0.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- P2M | 795.131 ms | 4.7% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- M2M | 997.962 ms | 5.9% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- M2L | 13347.671 ms | 78.4% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- L2L | 966.072 ms | 5.7% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- L2P | 786.773 ms | 4.6% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- near field | 136.214 ms | 0.8% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- output assembly | 0.478 ms | 0.0% |
| **1024** | | | |
| | Direct JIT kernel/matvec | 133.350 ms | 100.0% |
| | **FMM Speed (Total)** | **6633.409 ms** | 100.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- allocation/zero-fill | 0.206 ms | 0.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- input padding | 0.468 ms | 0.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- P2M | 57.213 ms | 0.9% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- M2M | 271.992 ms | 4.1% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- M2L | 5447.056 ms | 82.1% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- L2L | 273.542 ms | 4.1% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- L2P | 57.538 ms | 0.9% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- near field | 524.490 ms | 7.9% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- output assembly | 0.903 ms | 0.0% |
| | **FMM Reference (Total)** | **40448.320 ms** | 100.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- allocation/zero-fill | 0.494 ms | 0.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- input padding | 0.460 ms | 0.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- P2M | 1596.514 ms | 3.9% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- M2M | 1975.140 ms | 4.9% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- M2L | 32077.653 ms | 79.3% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- L2L | 2022.711 ms | 5.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- L2P | 1606.504 ms | 4.0% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- near field | 1167.944 ms | 2.9% |
| | &nbsp;&nbsp;&nbsp;&nbsp;- output assembly | 0.900 ms | 0.0% |