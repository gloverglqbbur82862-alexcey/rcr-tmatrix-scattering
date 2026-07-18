#!/usr/bin/env python3
"""Run optional MSTM anchor validation for bpca_n16_w550_seed42 and compare output matrices."""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Optional MSTM anchor validation for BPCA cluster")
    parser.add_argument("--case-id", default="bpca_n16_w550_seed42", help="Case to validate")
    parser.add_argument("--mstm-bin", default="/tmp/mstm_src/MSTM-main/code/mstm", help="Path to MSTM executable")
    args = parser.parse_args()

    print(f"Checking for MSTM binary at {args.mstm_bin}...")
    if not os.path.exists(args.mstm_bin):
        print(f"\n[SKIP] MSTM binary not found at {args.mstm_bin}.")
        print("Optional BPCA MSTM anchor validation was skipped. The existing BCCA MSTM validation remains the core validation.")
        sys.exit(0)

    print("MSTM binary found, but BPCA anchor execution is not implemented in this helper yet.", file=sys.stderr)
    print("Refusing to report a passing BPCA anchor without actually running MSTM.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
