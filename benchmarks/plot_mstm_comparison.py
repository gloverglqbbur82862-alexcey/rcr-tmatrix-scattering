#!/usr/bin/env python3
"""Plot full physical solve comparison: MSTM Fortran vs. Our JIT Solver."""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_comparison_plot():
    results_path = "experiments/full_solve_comparison.json"
    plot_out = "experiments/full_solve_comparison.png"
    
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found.")
        return
        
    with open(results_path) as f:
        results = json.load(f)
        
    n_list = sorted([int(x) for x in results.keys()])
    t_mstm = []
    t_rcr_single = []
    t_rcr_multi = []
    t_fmm = []
    best_threads = []
    
    for n in n_list:
        n_str = str(n)
        data = results[n_str]
        
        t_mstm.append(data["mstm"]["time"])
        t_rcr_single.append(data["rcr_single"]["time"])
        t_fmm.append(data.get("fmm_serial", {"time": 0.0})["time"])
        
        # Find best parallel time
        best_t = None
        best_time = float("inf")
        for k, v in data.items():
            if k.startswith("rcr_multi_t"):
                if v["time"] < best_time:
                    best_time = v["time"]
                    best_t = int(k.replace("rcr_multi_t", ""))
        t_rcr_multi.append(best_time)
        best_threads.append(best_t)
        
    plt.figure(figsize=(10.5, 6.8))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot MSTM Fortran Serial
    plt.plot(
        n_list, t_mstm, 
        marker='^', markersize=9, color='#ff7675', linewidth=2.5, linestyle='--',
        label="MSTM Native Fortran (Serial)"
    )
    
    # Plot Our Solver Single-Core (T=1)
    plt.plot(
        n_list, t_rcr_single, 
        marker='o', markersize=9, color='#e056fd', linewidth=2.5,
        label="Our JIT Solver: Single-Core (T=1)"
    )
    
    # Plot Our Solver Multi-Core (T=Best)
    plt.plot(
        n_list, t_rcr_multi, 
        marker='s', markersize=9, color='#0984e3', linewidth=2.5,
        label="Our JIT Solver: Multi-Core (T=Best Parallel)"
    )
    
    # Plot Our FMM Cascade Solver (Serial)
    valid_fmm_n = [n for idx, n in enumerate(n_list) if t_fmm[idx] > 0]
    valid_fmm_t = [t for t in t_fmm if t > 0]
    if valid_fmm_n:
        plt.plot(
            valid_fmm_n, valid_fmm_t, 
            marker='D', markersize=9, color='#00b894', linewidth=2.5,
            label="Our FMM Cascade (Serial Python)"
        )
    
    # Annotate speedup next to parallel data points compared to MSTM
    for idx, n in enumerate(n_list):
        t_m = t_mstm[idx]
        t_p = t_rcr_multi[idx]
        t_opt = best_threads[idx]
        speedup = t_m / t_p if t_p > 0 else 0.0
        
        plt.annotate(
            f"T={t_opt}\n{t_p:.2f} s\n({speedup:.1f}x vs MSTM)",
            xy=(n, t_p),
            xytext=(15, -15 if idx % 2 == 0 else 15),
            textcoords='offset points',
            fontsize=9.5,
            fontweight='bold',
            color='#0984e3',
            arrowprops=dict(arrowstyle="->", color='#0984e3', lw=0.8)
        )
        
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xticks(n_list, [str(n) for n in n_list], fontsize=11)
    plt.yticks(fontsize=11)
    
    plt.xlabel("Particle Count (N)", fontsize=13, fontweight='bold', labelpad=10)
    plt.ylabel("Complete Physical Solve Time (seconds)", fontsize=13, fontweight='bold', labelpad=10)
    
    plt.title(
        "RCR Light Scattering JIT Solver vs. MSTM Native Fortran\n"
        "Complete Solve to 10^-12 Precision on AMD EPYC 9754\n"
        "(Soot cluster sizes N = 16 to 128, Size parameter x ≈ 0.2856, Truncation order p = 7)",
        fontsize=13.5, fontweight='bold', pad=15
    )
    
    plt.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='#dfe6e9')
    plt.tight_layout()
    
    plt.savefig(plot_out, dpi=300)
    print(f"📊 Full solve comparison plot generated and saved successfully to {plot_out}")

if __name__ == "__main__":
    generate_comparison_plot()
