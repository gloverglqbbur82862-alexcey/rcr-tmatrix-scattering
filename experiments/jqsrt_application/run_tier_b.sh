#!/bin/bash
# Run Tier B cases
# Aligned thread settings to prevent oversubscription:
# For N <= 32: workers=1, threads-per-case=8 (runs sequentially, since individual runs are fast)
# For N = 64: workers=1, threads-per-case=24 (uses full parallel core budget)

echo "=== Running JQSRT Tier B Size-Scaling Sweep ==="

# 1. Run N <= 32 cases
echo "Running N <= 32 cases..."
./.venv/bin/python experiments/jqsrt_application/run_batch.py --cases experiments/jqsrt_application/cases.json --case-id bcca_n8_w550_seed42 --workers 1 --threads-per-case 8
./.venv/bin/python experiments/jqsrt_application/run_batch.py --cases experiments/jqsrt_application/cases.json --case-id bpca_n8_w550_seed42 --workers 1 --threads-per-case 8
./.venv/bin/python experiments/jqsrt_application/run_batch.py --cases experiments/jqsrt_application/cases.json --case-id bcca_n32_w550_seed42 --workers 1 --threads-per-case 8
./.venv/bin/python experiments/jqsrt_application/run_batch.py --cases experiments/jqsrt_application/cases.json --case-id bpca_n32_w550_seed42 --workers 1 --threads-per-case 8

# 2. Run N = 64 cases
echo "Running N = 64 cases..."
./.venv/bin/python experiments/jqsrt_application/run_batch.py --cases experiments/jqsrt_application/cases.json --case-id bcca_n64_w550_seed42 --workers 1 --threads-per-case 24
./.venv/bin/python experiments/jqsrt_application/run_batch.py --cases experiments/jqsrt_application/cases.json --case-id bpca_n64_w550_seed42 --workers 1 --threads-per-case 24

echo "=== Tier B Sweep Complete ==="
