#!/usr/bin/env python3
"""Plot parallel scaling curves from benchmark results JSON."""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_plot():
    results_path = "experiments/epyc_scaling_results.json"
    plot_out = "experiments/scaling_curve.png"
    
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found.")
        return
        
    with open(results_path) as f:
        results = json.load(f)
        
    # We want to extract N values and their corresponding T=1 and T=best times
    n_list = sorted([int(x) for x in results.keys()])
    t1_times = []
    tbest_times = []
    best_threads = []
    
    for n in n_list:
        n_str = str(n)
        t_data = results[n_str]
        
        # T=1 time
        t1_time = t_data["1"]["avg_mvm"] * 1000  # in ms
        t1_times.append(t1_time)
        
        # Find best parallel time
        best_t = None
        best_time = float("inf")
        for t_str, metrics in t_data.items():
            t_val = int(t_str)
            if t_val > 1 and metrics["avg_mvm"] < best_time:
                best_time = metrics["avg_mvm"]
                best_t = t_val
                
        tbest_times.append(best_time * 1000)  # in ms
        best_threads.append(best_t)
        
    # Create matplotlib plot
    plt.figure(figsize=(10, 6.5))
    
    # Use modern sleek styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot Single-Core (T=1)
    plt.plot(
        n_list, t1_times, 
        marker='o', markersize=8, color='#e056fd', linewidth=2.5, 
        label="Our JIT Solver: Single-Core (T=1)"
    )
    
    # Plot Multi-Core (T=best)
    plt.plot(
        n_list, tbest_times, 
        marker='s', markersize=8, color='#0984e3', linewidth=2.5,
        label="Our JIT Solver: Multi-Core (T=Best Parallel)"
    )
    
    # Annotate the best thread count and time next to each parallel data point
    for idx, n in enumerate(n_list):
        t_opt = best_threads[idx]
        t_ms = tbest_times[idx]
        speedup = t1_times[idx] / t_ms
        plt.annotate(
            f"T={t_opt}\n{t_ms:.1f} ms\n({speedup:.1f}x)",
            xy=(n, t_ms),
            xytext=(10, -10 if idx % 2 == 0 else 10),
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
    plt.ylabel("Matrix-Vector Multiplication Time (ms)", fontsize=13, fontweight='bold', labelpad=10)
    
    plt.title(
        "RCR Light Scattering JIT Solver Parallel Scaling on AMD EPYC 9754\n"
        "(Size parameter x ≈ 0.2856, Truncation order p = 7)",
        fontsize=14, fontweight='bold', pad=15
    )
    
    plt.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='#dfe6e9')
    plt.tight_layout()
    
    plt.savefig(plot_out, dpi=300)
    print(f"📊 Scaling plot generated and saved successfully to {plot_out}")

if __name__ == "__main__":
    generate_plot()
