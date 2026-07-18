#!/bin/bash
# Run Tier A cases (medium size N=16)
# Recommended defaults: workers=4, threads-per-case=8, NUMBA_NUM_THREADS=1

export JQSRT_CASE_WORKERS=4
export JQSRT_THREADS_PER_CASE=8

echo "Running JQSRT Tier A validation sweep with $JQSRT_CASE_WORKERS workers, $JQSRT_THREADS_PER_CASE threads per case..."
./.venv/bin/python experiments/jqsrt_application/run_batch.py \
  --cases experiments/jqsrt_application/cases.json \
  --tier A \
  --workers $JQSRT_CASE_WORKERS \
  --threads-per-case $JQSRT_THREADS_PER_CASE
