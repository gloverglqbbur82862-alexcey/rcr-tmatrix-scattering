# Unified Backend Benchmark Report

- **CUDA Available**: True

| N | p | Backend | Status | Init Time (s) | Warmup (s) | Avg Matvec (s) | Memory (MB) | Rel Error | Speedup |
|---|---|---------|--------|---------------|------------|----------------|-------------|-----------|---------|
| 128 | 3 | `direct_cpu_serial` | Success | 0.2187 | 0.0412 | 0.0408 | 569.3 | 0.00e+00 | 1.00x |
| 128 | 3 | `direct_cpu_parallel` | Success | 0.2213 | 0.0053 | 0.0054 | 573.2 | 0.00e+00 | 7.50x |
| 128 | 3 | `fmm_cpu_serial` | Success | 1.7443 | 5.4522 | 0.3981 | 683.4 | 8.15e-12 | 0.10x |
| 128 | 3 | `direct_gpu` | Success | 1.1335 | 0.0695 | 0.0182 | 718.2 | 2.62e-16 | 2.25x |
| 128 | 3 | `direct_gpu_dense` | Success | 3.4751 | 0.0910 | 0.0007 | 1056.9 | 4.93e-16 | 62.36x |
| 128 | 3 | `fmm_gpu` | Success | 1.8436 | 0.3995 | 0.4025 | 1150.3 | 8.15e-12 | 0.10x |
| 128 | 3 | `auto` | Success | 1.7093 | 0.3958 | 0.3969 | 1155.0 | 8.15e-12 | 0.10x |
| 128 | 5 | `direct_cpu_serial` | Success | 0.3440 | 0.1263 | 0.1264 | 1093.6 | 0.00e+00 | 1.00x |
| 128 | 5 | `direct_cpu_parallel` | Success | 0.3401 | 0.0116 | 0.0137 | 1093.6 | 0.00e+00 | 9.26x |
| 128 | 5 | `fmm_cpu_serial` | Success | 1.8200 | 0.4536 | 0.4511 | 1173.7 | 1.01e-13 | 0.28x |
| 128 | 5 | `direct_gpu` | Success | 0.6132 | 0.3486 | 0.0652 | 1166.3 | 3.12e-16 | 1.94x |
| 128 | 5 | `direct_gpu_dense` | Success | 1.9092 | 0.0074 | 0.0066 | 3214.4 | 5.10e-16 | 19.05x |
| 128 | 5 | `fmm_gpu` | Success | 1.7877 | 0.4490 | 0.4515 | 3294.2 | 1.01e-13 | 0.28x |
| 128 | 5 | `auto` | Success | 1.7876 | 0.4501 | 0.4554 | 3296.5 | 1.01e-13 | 0.28x |
| 256 | 3 | `direct_cpu_serial` | Success | 0.8877 | 0.1751 | 0.1739 | 3234.7 | 0.00e+00 | 1.00x |
| 256 | 3 | `direct_cpu_parallel` | Success | 0.8786 | 0.0342 | 0.0263 | 3242.4 | 0.00e+00 | 6.61x |
| 256 | 3 | `fmm_cpu_serial` | Success | 3.8106 | 0.8928 | 0.8795 | 3394.2 | 5.51e-10 | 0.20x |
| 256 | 3 | `direct_gpu` | Success | 1.4609 | 0.1489 | 0.0361 | 3256.3 | 2.44e-16 | 4.82x |
| 256 | 3 | `direct_gpu_dense` | Success | 2.3996 | 0.0056 | 0.0051 | 4287.2 | 6.66e-16 | 33.89x |
| 256 | 3 | `fmm_gpu` | Success | 3.8521 | 0.9365 | 0.9032 | 4439.9 | 5.51e-10 | 0.19x |
| 256 | 3 | `auto` | Success | 3.7598 | 0.8889 | 0.8830 | 4445.2 | 5.51e-10 | 0.20x |
| 256 | 5 | `direct_cpu_serial` | Success | 1.3391 | 0.5061 | 0.5075 | 4385.6 | 0.00e+00 | 1.00x |
| 256 | 5 | `direct_cpu_parallel` | Success | 1.3451 | 0.0872 | 0.0811 | 4396.4 | 0.00e+00 | 6.26x |
| 256 | 5 | `fmm_cpu_serial` | Success | 4.1202 | 1.0974 | 1.0925 | 4520.3 | 2.27e-11 | 0.46x |
| 256 | 5 | `direct_gpu` | Success | 2.4045 | 0.3760 | 0.1227 | 4403.1 | 3.03e-16 | 4.14x |
| 256 | 5 | `direct_gpu_dense` | Failed: Out of memory allocating 5,138,022,400 bytes (allocated so far: 238,041,600 bytes). | N/A | N/A | N/A | N/A | N/A | N/A |
| 256 | 5 | `fmm_gpu` | Success | 4.1306 | 1.1249 | 1.1055 | 4539.2 | 2.27e-11 | 0.46x |
| 256 | 5 | `auto` | Success | 4.0628 | 1.1028 | 1.1019 | 4542.0 | 2.27e-11 | 0.46x |
| 512 | 3 | `direct_cpu_serial` | Success | 3.5235 | 0.6565 | 0.6529 | 4510.5 | 0.00e+00 | 1.00x |
| 512 | 3 | `direct_cpu_parallel` | Success | 3.5488 | 0.2923 | 0.1297 | 4529.4 | 0.00e+00 | 5.04x |
| 512 | 3 | `fmm_cpu_serial` | Success | 11.4086 | 4.3730 | 4.3586 | 4882.5 | 1.12e-10 | 0.15x |
| 512 | 3 | `direct_gpu` | Success | 5.8212 | 0.2002 | 0.0654 | 4584.2 | 2.92e-16 | 9.99x |
| 512 | 3 | `direct_gpu_dense` | Failed: Out of memory allocating 3,774,873,600 bytes (allocated so far: 265,435,648 bytes). | N/A | N/A | N/A | N/A | N/A | N/A |
| 512 | 3 | `fmm_gpu` | Success | 11.2965 | 4.4381 | 4.4343 | 4954.9 | 1.12e-10 | 0.15x |
| 512 | 3 | `auto` | Success | 11.2785 | 4.4167 | 4.4115 | 4963.7 | 1.12e-10 | 0.15x |
| 512 | 5 | `direct_cpu_serial` | Success | 5.3917 | 2.0547 | 2.0448 | 4973.6 | 0.00e+00 | 1.00x |
| 512 | 5 | `direct_cpu_parallel` | Success | 5.3973 | 0.3501 | 0.3879 | 4992.1 | 0.00e+00 | 5.27x |
| 512 | 5 | `fmm_cpu_serial` | Success | 11.4599 | 6.1168 | 6.1084 | 4988.5 | 1.96e-12 | 0.33x |
| 512 | 5 | `direct_gpu` | Success | 9.5901 | 0.2058 | 0.1757 | 5056.7 | 3.19e-16 | 11.64x |
| 512 | 5 | `direct_gpu_dense` | Failed: Out of memory allocating 20,552,089,600 bytes (allocated so far: 863,123,968 bytes). | N/A | N/A | N/A | N/A | N/A | N/A |
| 512 | 5 | `fmm_gpu` | Success | 11.3403 | 6.2066 | 6.1575 | 5061.3 | 1.96e-12 | 0.33x |
| 512 | 5 | `auto` | Success | 11.3526 | 6.1469 | 6.1548 | 5070.2 | 1.96e-12 | 0.33x |
| 1024 | 3 | `direct_cpu_serial` | Success | 14.3500 | 3.0999 | 3.0957 | 5466.1 | 0.00e+00 | 1.00x |
| 1024 | 3 | `direct_cpu_parallel` | Success | 14.7724 | 1.0447 | 0.6829 | 5493.3 | 0.00e+00 | 4.53x |
| 1024 | 3 | `fmm_cpu_serial` | Success | 23.0678 | 18.9898 | 18.9631 | 5258.4 | 9.63e-10 | 0.16x |
| 1024 | 3 | `direct_gpu` | Success | 23.5115 | 0.0979 | 0.0846 | 5727.0 | 2.73e-16 | 36.60x |
| 1024 | 3 | `direct_gpu_dense` | Failed: Unable to allocate 14.1 GiB for an array with shape (30720, 30720) and data type complex128 | N/A | N/A | N/A | N/A | N/A | N/A |
| 1024 | 3 | `fmm_gpu` | Success | 22.9171 | 19.0415 | 19.0261 | 5518.8 | 9.63e-10 | 0.16x |
| 1024 | 3 | `auto` | Success | 22.9067 | 19.0497 | 19.0868 | 5518.8 | 9.63e-10 | 0.16x |
| 1024 | 5 | `direct_cpu_serial` | Skipped (Direct large size restriction) | N/A | N/A | N/A | N/A | N/A | N/A |
| 1024 | 5 | `direct_cpu_parallel` | Skipped (Direct large size restriction) | N/A | N/A | N/A | N/A | N/A | N/A |
| 1024 | 5 | `fmm_cpu_serial` | Success | 22.5691 | 25.1145 | 25.0974 | 5456.4 | N/A | N/A |
| 1024 | 5 | `direct_gpu` | Success | 38.9024 | 0.4060 | 0.3909 | 11702.4 | N/A | N/A |
| 1024 | 5 | `direct_gpu_dense` | Failed: Required dense matrix size 76.56 GiB exceeds available GPU free memory 17.69 GiB. | N/A | N/A | N/A | N/A | N/A | N/A |
| 1024 | 5 | `fmm_gpu` | Success | 22.8255 | 25.0956 | 25.0558 | 9825.8 | N/A | N/A |
| 1024 | 5 | `auto` | Success | 22.5995 | 24.9326 | 24.9559 | 9817.1 | N/A | N/A |