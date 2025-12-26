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
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
raw_values = [1073.52, 765.51, 434.42, 412.25, 825.47, 1105.17, 2339.85, 2051.63, 1942.65]
costs = [int(v) for v in raw_values]
mean_val = 1216

line_color = '#82B0D2' # Blue
mean_color = '#FA7F6F' # Red

# --- 3. Plotting ---
fig, ax = plt.subplots(figsize=(8, 5))

# Mean line
ax.axhline(y=mean_val, color=mean_color, linewidth=3, linestyle='--', label='Average', zorder=1)
# Line plot
ax.plot(months, costs, color=line_color, linewidth=3, marker='o', markersize=10, label='Monthly Cost', zorder=2)

# Label only extreme values
max_val = max(costs)
min_val = min(costs)
offset_val = max_val * 0.05 
for x, y in zip(months, costs):
    if y == max_val:
        ax.text(x, y + offset_val/2, str(y), ha='center', va='bottom', fontsize=hzysfontsize, color='black')
    elif y == min_val:
        ax.text(x, y - offset_val, str(y), ha='center', va='top', fontsize=hzysfontsize, color='black')

# --- 4. Axes and Legend ---
ax.legend(fontsize=18, frameon=False, loc='upper left')

# Move X-axis labels down (Consistent with Fig 1)
padding_value = 35 
ax.tick_params(axis='x', which='major', pad=padding_value)
ax.set_xticklabels(months, rotation=0, ha='center')

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
ax.text(1.0, mean_val, f'{mean_val}', 
        transform=ax.get_yaxis_transform(), 
        color=mean_color, 
        ha='right',    # Close to the right border inside
        va='bottom',   # Located above the line
        fontsize=18, 
        fontweight='bold')

# --- [Critical] Manually lock margins (completely consistent with Fig 1) ---
plt.subplots_adjust(left=0.16, right=0.95, top=0.92, bottom=0.25)

# --- 5. Saving ---
output_dir = './figs'
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'cost_line.pdf')

# [Critical] Do not use bbox_inches='tight'
plt.savefig(save_path, format='pdf', dpi=300)
print(f"Line chart saved: {save_path}")