import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import FuncFormatter
import os

# --- 1. Style Configuration ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri'] + plt.rcParams['font.sans-serif']
hzysfontsize = 22
plt.rcParams['font.size'] = hzysfontsize

# --- 2. Data Preparation ---
# 18 data points provided
raw_values = [3969.933, 949.84, 1805.74, 1542.375, 1294.482, 2672.669, 1605.515, 3844, 940, 1073.52, 765.51, 434.42, 412.25, 825.47, 1105.17, 2339.85, 2051.63, 1942.65]
costs = [int(v) for v in raw_values]

# [Modification 1] Generate labels for 18 months ending in 25-Sep
# Start date shifts to 24-Apr to match length of 18
months_24 = ['24-Apr', '24-May', '24-Jun', '24-Jul', '24-Aug', '24-Sep', '24-Oct', '24-Nov', '24-Dec']
months_25 = ['25-Jan', '25-Feb', '25-Mar', '25-Apr', '25-May', '25-Jun', '25-Jul', '25-Aug', '25-Sep']
months = months_24 + months_25

# Validate lengths
assert len(costs) == len(months), f"Data length {len(costs)} != Labels length {len(months)}"

mean_val = 1642
line_color = '#82B0D2' # Blue
mean_color = '#FA7F6F' # Red

# --- 3. Plotting ---
fig, ax = plt.subplots(figsize=(8, 4))

# Use index for x-axis plotting to control ticks easily
x_indices = np.arange(len(months))

# Mean line
ax.axhline(y=mean_val, color=mean_color, linewidth=3, linestyle='--', label='Average', zorder=1)
# Line plot
ax.plot(x_indices, costs, color=line_color, linewidth=3, marker='o', markersize=10, label='Monthly Cost', zorder=2)

# Label only extreme values (Max and Min)
max_val = max(costs)
min_val = min(costs)
offset_val = max_val * 0.03 # Adjusted offset for visibility

for i, y in enumerate(costs):
    if y == max_val:
        ax.text(i+1, y + offset_val/2, str(y), ha='center', va='bottom', fontsize=hzysfontsize, color='black')
    elif y == min_val:
        ax.text(i+1.5, y , str(y), ha='center', va='top', fontsize=hzysfontsize, color='black')

# --- 4. Axes and Legend ---
ax.legend(fontsize=18, frameon=False, loc='upper right')

# [Modification 2 & 3] Sparse X-axis labeling (5 ticks, no rotation)
tick_indices = np.linspace(0, len(months) - 1, 5, dtype=int)
tick_labels = [months[i] for i in tick_indices]

ax.set_xticks(tick_indices)
ax.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=hzysfontsize)

# Move X-axis labels down
padding_value = 10 
ax.tick_params(axis='x', which='major', pad=padding_value)

ax.set_xlabel('Month', fontsize=hzysfontsize, labelpad=10)
ax.set_ylabel('Cost (USD)', fontsize=hzysfontsize)
ax.grid(True, which='major', linestyle='--', alpha=0.6)

# Range settings
ax.set_ylim(-200, max(costs) * 1.3)
ax.set_xlim(-0.5, len(months) - 0.5)

# Y-axis formatting
def k_formatter(x, pos):
    if abs(x) >= 1000: return f'{int(x/1000)}k'
    return f'{int(x)}' if x == 0 else str(int(x))
ax.yaxis.set_major_formatter(FuncFormatter(k_formatter))

# Mean annotation (right side + above line)
ax.text(len(months)-1, mean_val - 600, f'{mean_val}', 
        color=mean_color, 
        ha='right',    
        va='bottom',   
        fontsize=18, 
        fontweight='bold')

# --- [Critical] Manually lock margins (Consistent with Bar Chart) ---
plt.subplots_adjust(left=0.16, right=0.95,
                    top=0.92, bottom=0.25)

# --- 5. Saving ---
output_dir = './figs'
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'cost_line.pdf')

# [Critical] Do not use bbox_inches='tight'
plt.savefig(save_path, format='pdf', dpi=300)
print(f"Expanded Line chart (18 months) saved: {save_path}")
