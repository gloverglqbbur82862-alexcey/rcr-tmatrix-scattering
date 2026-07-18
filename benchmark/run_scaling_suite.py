#!/usr/bin/env python3
"""EPYC Thread Scaling and NUMA Benchmarking Suite.

Iterates over particle counts and thread configurations, running each in an isolated
process. Compiles scaling statistics, parallel speedups, and exports dashboards.
"""

import os
import sys
import json
import subprocess
import argparse
import shutil
import numpy as np

def run_suite(n_values: list[int], thread_counts: list[int], num_mvm: int, use_numa: bool):
    # Ensure experiments folder exists
    os.makedirs("experiments", exist_ok=True)
    
    # Path to the single benchmark script
    bench_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_numba_scaling.py")
    python_exe = sys.executable
    
    # Verify numactl if requested
    has_numactl = shutil.which("numactl") is not None
    if use_numa and not has_numactl:
        print("WARNING: numactl requested but not found in PATH. Disabling NUMA wrapping.")
        use_numa = False
        
    results = {}
    
    print("\n" + "=" * 80)
    print("🚀 Starting RCR Parallel Scaling Suite")
    print(f"   Particle Counts (N): {n_values}")
    print(f"   Thread Configurations: {thread_counts}")
    print(f"   NUMA Tuning enabled: {use_numa} (via numactl --interleave=all)")
    print("=" * 80 + "\n")
    
    for n in n_values:
        results[n] = {}
        print(f"--- Running scaling tests for N = {n} ---")
        
        # We will use T = 1 as the serial baseline for speedup calculations
        t1_avg = None
        
        for t in thread_counts:
            # Set thread count environment variable
            env = os.environ.copy()
            env["NUMBA_NUM_THREADS"] = str(t)
            env["OMP_NUM_THREADS"] = str(t)
            env["MKL_NUM_THREADS"] = str(t)
            
            # Prepare execution command
            cmd = [python_exe, bench_script, "-n", str(n), "--mvm", str(num_mvm)]
            
            if use_numa:
                cmd = ["numactl", "--interleave=all"] + cmd
                
            print(f"  [N={n:3d}] Running with T = {t:3d} threads... ", end="", flush=True)
            
            try:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=600  # 10 minute timeout per configuration
                )
                
                if proc.returncode != 0:
                    print(f"FAILED (Exit Code {proc.returncode})")
                    print(f"STDOUT:\n{proc.stdout}")
                    print(f"STDERR:\n{proc.stderr}")
                    continue
                    
                # Parse stdout for RESULT line
                res_line = None
                for line in proc.stdout.split("\n"):
                    if line.startswith("RESULT:"):
                        res_line = line
                        break
                        
                if res_line is None:
                    print("FAILED (No result line found)")
                    continue
                    
                # Parse values
                parts = res_line.replace("RESULT:", "").strip().split(",")
                metrics = {}
                for p in parts:
                    k, v = p.split("=")
                    metrics[k] = float(v)
                    
                avg_mvm = metrics["avg_mvm"]
                std_mvm = metrics["std_mvm"]
                init_time = metrics["init_time"]
                warmup_time = metrics["warmup_time"]
                
                # Baseline for T=1
                if t == 1 or t1_avg is None:
                    t1_avg = avg_mvm
                    
                speedup = t1_avg / avg_mvm
                efficiency = speedup / t
                
                results[n][t] = {
                    "avg_mvm": avg_mvm,
                    "std_mvm": std_mvm,
                    "init_time": init_time,
                    "warmup_time": warmup_time,
                    "speedup": speedup,
                    "efficiency": efficiency
                }
                
                print(f"Success. MVM avg: {avg_mvm * 1000:.2f} ms | Speedup: {speedup:.2f}x | Efficiency: {efficiency * 100:.1f}%")
                
            except subprocess.TimeoutExpired:
                print("TIMEOUT")
            except Exception as e:
                print(f"ERROR: {e}")
                
    # Save results to JSON
    out_path = "experiments/epyc_scaling_results.json"
    # Convert keys to strings for JSON serialization
    serialized_results = {}
    for n_key, n_data in results.items():
        serialized_results[str(n_key)] = {str(t_key): t_data for t_key, t_data in n_data.items()}
        
    with open(out_path, "w") as f:
        json.dump(serialized_results, f, indent=4)
        
    # Print Final Summary Dashboard
    print("\n" + "=" * 80)
    print("📊 SCALING BENCHMARK SUMMARY DASHBOARD")
    print("=" * 80)
    
    for n in n_values:
        print(f"\nParticle Count N = {n}:")
        print(f"{'Threads':<8} | {'MVM Avg (ms)':<15} | {'Speedup':<10} | {'Efficiency (%)':<15}")
        print("-" * 58)
        
        for t in thread_counts:
            if t in results[n]:
                res = results[n][t]
                print(f"{t:<8d} | {res['avg_mvm'] * 1000:<15.2f} | {res['speedup']:<9.2f}x | {res['efficiency'] * 100:<14.1f}%")
            else:
                print(f"{t:<8d} | {'N/A':<15} | {'N/A':<10} | {'N/A':<15}")
                
    print("\n" + "=" * 80)
    print(f"💾 Results saved successfully to {out_path}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, default="16,32,64", help="Comma-separated N values")
    parser.add_argument("-t", type=str, default="1,4,8,16,24,48,64,128,256,512", help="Comma-separated thread values")
    parser.add_argument("--mvm", type=int, default=10, help="Number of Matvecs per test")
    parser.add_argument("--numa", action="store_true", help="Wrap runs with numactl --interleave=all")
    args = parser.parse_args()
    
    n_values = [int(x) for x in args.n.split(",")]
    thread_counts = [int(x) for x in args.t.split(",")]
    
    run_suite(n_values, thread_counts, args.mvm, args.numa)
