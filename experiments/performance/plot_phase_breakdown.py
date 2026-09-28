import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load benchmark results
json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_solve_benchmark.json")
with open(json_path, "r") as f:
    results = json.load(f)

# Ensure figures directory exists
figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "figures")
os.makedirs(figures_dir, exist_ok=True)

# Set scientific plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18,
    'legend.fontsize': 11,
    'font.family': 'sans-serif'
})

sizes = ["64", "128", "256"]
backends = ["direct_gpu_dense"]

# Extract phase times for stacked bar chart
assembly = []
h2d = []
rhs = []
solve = []
merge = []
orientation = []
overhead = []

for N in sizes:
    data = results[N]["direct_gpu_dense"]
    total = data["time"]
    asm = data["assembly_time"]
    h_2_d = data["h2d_time"]
    r_h_s = data["rhs_time"]
    sol = data["solve_time"]
    merg = data["merge_time"]
    ori = data["orientation_time"]
    
    sum_phases = asm + h_2_d + r_h_s + sol + merg + ori
    over = max(0.0, total - sum_phases)
    
    assembly.append(asm)
    h2d.append(h_2_d)
    rhs.append(r_h_s)
    solve.append(sol)
    merge.append(merg)
    orientation.append(ori)
    overhead.append(over)

# Plot 1: Stacked Bar Chart of Phase Times
fig, ax = plt.subplots(figsize=(8, 6))
width = 0.5
x = np.arange(len(sizes))

colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

p1 = ax.bar(x, assembly, width, label='Assembly', color=colors[0])
p2 = ax.bar(x, h2d, width, bottom=assembly, label='H2D Transfer', color=colors[1])
p3 = ax.bar(x, rhs, width, bottom=np.array(assembly)+np.array(h2d), label='RHS Build', color=colors[2])
p4 = ax.bar(x, solve, width, bottom=np.array(assembly)+np.array(h2d)+np.array(rhs), label='Krylov Solve', color=colors[3])
p5 = ax.bar(x, merge, width, bottom=np.array(assembly)+np.array(h2d)+np.array(rhs)+np.array(solve), label='Merge Total', color=colors[4])
p6 = ax.bar(x, orientation, width, bottom=np.array(assembly)+np.array(h2d)+np.array(rhs)+np.array(solve)+np.array(merge), label='Orientation Averaging', color=colors[5])
p7 = ax.bar(x, overhead, width, bottom=np.array(assembly)+np.array(h2d)+np.array(rhs)+np.array(solve)+np.array(merge)+np.array(orientation), label='Overhead/Other', color=colors[6])

ax.set_ylabel('Execution Time (seconds)', fontweight='bold')
ax.set_xlabel('Cluster Size (N)', fontweight='bold')
ax.set_title('Phase Breakdown for GPU Dense Backend', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.legend(loc='upper left', frameon=True)

# Add value labels on top of the bars
for i, N in enumerate(sizes):
    total_val = results[N]["direct_gpu_dense"]["time"]
    ax.text(i, total_val + 1.0, f"{total_val:.2f}s", ha='center', fontweight='bold')

plt.tight_layout()
fig_path_1 = os.path.join(figures_dir, "gpu_dense_phase_breakdown.png")
plt.savefig(fig_path_1, dpi=300)
print(f"Saved phase breakdown plot to: {fig_path_1}")
plt.close()

# Plot 2: Speedup vs CPU Parallel Baseline
sweep_sizes = ["16", "32", "64", "128", "256"]
x_sweep = np.arange(len(sweep_sizes))

speedup_gpu = []
speedup_gpu_dense = []

for N in sweep_sizes:
    cpu_time = results[N]["direct_cpu_parallel"]["time"]
    gpu_time = results[N]["direct_gpu"]["time"]
    gpu_dense_time = results[N]["direct_gpu_dense"]["time"]
    
    speedup_gpu.append(cpu_time / gpu_time)
    speedup_gpu_dense.append(cpu_time / gpu_dense_time)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x_sweep, speedup_gpu, marker='o', linewidth=2.5, markersize=8, label='GPUDirect (Dynamic Operator)', color='#e377c2')
ax.plot(x_sweep, speedup_gpu_dense, marker='s', linewidth=2.5, markersize=8, label='GPUDense (Pre-assembled Operator)', color='#17becf')

# Reference baseline line at y=1
ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='64-Thread CPU Parallel Baseline')

ax.set_ylabel('Speedup (x-fold)', fontweight='bold')
ax.set_xlabel('Cluster Size (N)', fontweight='bold')
ax.set_title('End-to-End Speedup over 64-Thread CPU Baseline', fontweight='bold', pad=15)
ax.set_xticks(x_sweep)
ax.set_xticklabels(sweep_sizes)
ax.legend(loc='upper left', frameon=True)

# Add labels for the final speedup values at N=256
ax.text(len(sweep_sizes)-1, speedup_gpu[-1] + 0.3, f"{speedup_gpu[-1]:.2f}x", ha='center', color='#e377c2', fontweight='bold')
ax.text(len(sweep_sizes)-1, speedup_gpu_dense[-1] + 0.3, f"{speedup_gpu_dense[-1]:.2f}x", ha='center', color='#17becf', fontweight='bold')

plt.tight_layout()
fig_path_2 = os.path.join(figures_dir, "gpu_backends_speedup_vs_cpu.png")
plt.savefig(fig_path_2, dpi=300)
print(f"Saved speedup plot to: {fig_path_2}")
plt.close()
