#!/usr/bin/env bash
set -euo pipefail

MSTM_SRC_DIR="${1:-/Users/kaer/Downloads/MSTM-main/code}"
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_mstm_parallel"

mkdir -p "${BUILD_DIR}"
cp "${MSTM_SRC_DIR}/mstm-intrinsics.f90" "${BUILD_DIR}/"
cp "${MSTM_SRC_DIR}/mpidefs-parallel.f90" "${BUILD_DIR}/"
cp "${MSTM_SRC_DIR}/mstm-v4.0.f90" "${BUILD_DIR}/mstm.f90"

cd "${BUILD_DIR}"
mpifort -O2 -fallow-argument-mismatch -c -o mstm-intrinsics.o mstm-intrinsics.f90
mpifort -O2 -fallow-argument-mismatch -c -o mpidefs-parallel.o mpidefs-parallel.f90
mpifort -O2 -fallow-argument-mismatch -c -o mstm.o mstm.f90
mpifort -O2 -fallow-argument-mismatch -o mstm_parallel \
  mstm-intrinsics.o mpidefs-parallel.o mstm.o

echo "${BUILD_DIR}/mstm_parallel"
