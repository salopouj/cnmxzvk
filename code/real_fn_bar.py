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
# Raw data (Sep -> Jan)
raw_values = [4317, 3871, 4105, 2167, 2231, 1649, 1498, 1343, 1512]
# Reversed to (Jan -> Sep)
data_values = raw_values[::-1] 
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

raw_values = [4317, 3871, 4105, 2167, 2231, 1649, 1498, 1343, 1512]
hzysmean_val = np.mean(raw_values)
print(f"Mean: {hzysmean_val:.2f}")

# Style
main_color = '#82B0D2'  
main_hatch = '/'

# --- 3. Plotting ---
fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(months, data_values, 
       color=main_color, 
       hatch=main_hatch,
       edgecolor='black', 
       linewidth=1.0)

# Top labels (x.xk) -- [Modification: changed to 1 decimal place]
for i, v in enumerate(data_values):
    offset = max(data_values) * 0.01
    # Changed to .1f
    label_text = f"{v/1000:.1f}k"
    ax.text(i, v + offset, label_text, ha='center', va='bottom', fontsize=hzysfontsize)

# --- 4. Axis Settings ---
ax.set_xlabel('Month', fontsize=hzysfontsize, labelpad=10)
ax.set_ylabel('Consolidates', fontsize=hzysfontsize)

# Y-axis formatting (1k, 2k)
def k_formatter(x, pos):
    if x == 0: return '0'
    return f'{int(x/1000)}k'
ax.yaxis.set_major_formatter(FuncFormatter(k_formatter))

ax.grid(True, which='major', linestyle='--', alpha=0.6, axis='y')
ax.set_ylim(0, max(data_values) * 1.15)

# Move X-axis labels down
padding_value = 35 
ax.tick_params(axis='x', which='major', pad=padding_value)
# Explicitly set labels to prevent overlap
ax.set_xticklabels(months, rotation=0, ha='center') 

# --- [Critical] Manually lock margins (Consistent between both plots) ---
plt.subplots_adjust(left=0.16, right=0.95, top=0.92, bottom=0.25)

# --- 5. Saving ---
output_dir = './figs'
os.makedirs(output_dir, exist_ok=True)
# [Keep filename unchanged]
save_path = os.path.join(output_dir, 'consolidates_monthly.pdf')

# [Critical] Do not use bbox_inches='tight', rely on subplots_adjust for alignment
plt.savefig(save_path, format='pdf', dpi=300)
print(f"Bar chart saved: {save_path}")