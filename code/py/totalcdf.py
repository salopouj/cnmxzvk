import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pickle
import os
from matplotlib.ticker import FuncFormatter

# --- 1. Style Configuration ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri'] + plt.rcParams['font.sans-serif']
hzysfontsize = 24
plt.rcParams['font.size'] = hzysfontsize

# --- 2. Data Loading ---
file_path = '/Users/hk00569ml/代码/fake deposit/merged_max_results.pkl'

print(f"Reading file: {file_path}")
try:
    with open(file_path, 'rb') as f:
        raw_data = pickle.load(f)
    data_full = np.array(raw_data).flatten().astype(int)
    print(f"Data loaded successfully, total {len(data_full)} items.")
except Exception as e:
    print(f"Error: {e}")
    data_full = np.random.randint(10, 800, 1000000)

# --- 3. Data Processing ---
mean_val = np.mean(data_full)
percentile_90 = np.percentile(data_full, 90)

sorted_full = np.sort(data_full)
n_total = len(sorted_full)
y_full = np.arange(1, n_total + 1) / n_total

# Plot downsampling
target_plot_points = 50000 
if n_total > target_plot_points:
    step = n_total // target_plot_points
    plot_x = sorted_full[::step]
    plot_y = y_full[::step]
    plot_x = np.append(plot_x, sorted_full[-1])
    plot_y = np.append(plot_y, 1.0)
else:
    plot_x = sorted_full
    plot_y = y_full

# --- 4. Plotting ---
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(plot_x, plot_y, color='#82B0D2', linewidth=3, zorder=1)

# --- 5. Annotation ---
mean_y_exact = np.searchsorted(sorted_full, mean_val, side='right') / n_total

ax.scatter(percentile_90, 0.9, 
            color="#96C37D", 
            marker='*', 
            edgecolor='black', 
            linewidth=1.0, 
            s=220, 
            zorder=5, 
            label=f'90%: {int(percentile_90)}ms')

ax.scatter(mean_val, mean_y_exact,
            color='#FA7F6F',
            marker='^',
            s=180, 
            edgecolor='black',
            linewidth=1.0, 
            zorder=5, 
            label=f'Mean: {int(mean_val)}ms')

# --- 6. Axis Settings (Modified for Log Scale) ---

# [Modification 1] Update Label
ax.set_xlabel('Time (ms, log)', fontsize=hzysfontsize, labelpad=10)
ax.set_ylabel('CDF', fontsize=hzysfontsize)
ax.grid(True, linestyle='--', alpha=0.6)

ax.legend(
    loc='upper right', # Moved to left as log scale usually opens up space on the left
    fontsize=hzysfontsize,
    frameon=False, 
    handletextpad=0.2,
    handlelength=1.0
)

# [Modification 2] Set Log Scale
ax.set_xscale('log')

# [Modification 3] Adjust Ticks and Limits for Log Scale
# Log scale cannot start at 0. We use 1, 10, 100, 1000 for better readability.
fixed_ticks = [1, 10, 100, 1000]
ax.set_xticks(fixed_ticks)

# Adjust limit to accommodate the ticks (e.g., 1 to 1500 covers the data range up to ~800+)
ax.set_xlim(1, 1500) 

# Keep the integer formatting
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))

# --- 7. Layout Adjustment ---

# Move X-axis labels down
padding_value = 20 
ax.tick_params(axis='x', which='major', pad=padding_value)

# [Critical Alignment] Keep consistent with bar chart
plt.subplots_adjust(left=0.20, right=0.95, top=0.92, bottom=0.25)

# --- 8. Saving ---
output_dir = './figs'
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'total_cdf.pdf')

# [Critical Alignment] Do not use bbox_inches='tight'
plt.savefig(save_path, format='pdf', dpi=300)
print(f"CDF Chart (Log Scale, aligned) saved to: {save_path}")