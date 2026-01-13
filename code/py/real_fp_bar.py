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

raw_values = [2039, 1532, 1921, 2285, 2499, 4007, 3035, 3852, 1355, 1512, 1343, 1498, 1649, 2231, 2167, 4105, 3871, 4317]

data_values = raw_values
# print(f"Data values: ", np.mean(raw_values))

# Generate labels for 18 months ending in 25-Sep
months_24 = ['24-Apr', '24-May', '24-Jun', '24-Jul', '24-Aug', '24-Sep', '24-Oct', '24-Nov', '24-Dec']
months_25 = ['25-Jan', '25-Feb', '25-Mar', '25-Apr', '25-May', '25-Jun', '25-Jul', '25-Aug', '25-Sep']
months = months_24 + months_25

assert len(data_values) == len(months), f"Data length {len(data_values)} != Labels length {len(months)}"

hzysmean_val = np.mean(data_values)
print(f"Mean: {hzysmean_val:.2f}")

# Style
main_color = '#82B0D2'  
main_hatch = '/'
mean_color = '#FA7F6F' # Red color for mean line

# --- 3. Plotting ---
fig, ax = plt.subplots(figsize=(8, 4))

x_indices = np.arange(len(months))

# Draw Bars
ax.bar(x_indices, data_values, 
       width=0.6,
       color=main_color, 
       hatch=main_hatch,
       edgecolor='black', 
       linewidth=1.0,
       label='_nolegend_') # Exclude bars from legend if preferred, or remove label

ax.axhline(y=hzysmean_val, color=mean_color, linestyle='--', linewidth=2.5, label=f'Mean ({hzysmean_val/1000:.2f}ko0)', zorder=5)

# Only annotate Max and Min values
max_val = max(data_values)
min_val = min(data_values)

for i, v in enumerate(data_values):
    if v == max_val or v == min_val:
        offset = max(data_values) * 0.02 
        label_text = f"{v/1000:.1f}k"
        ax.text(i, v + offset, label_text, ha='center', va='bottom', fontsize=14, color='black')

# --- 4. Axis Settings ---
ax.set_xlabel('Month', fontsize=hzysfontsize, labelpad=10)
ax.set_ylabel('Consolidation times', fontsize=hzysfontsize)

# [新增] 添加图例
ax.legend(loc='upper left', fontsize=18, frameon=False)

# Y-axis formatting (1k, 2k)
def k_formatter(x, pos):
    if x == 0: return '0'
    return f'{int(x/1000)}k'
ax.yaxis.set_major_formatter(FuncFormatter(k_formatter))

ax.grid(True, which='major', linestyle='--', alpha=0.6, axis='y')
ax.set_ylim(0, max(data_values) * 1.15)

# --- X-axis sparse labeling ---
tick_indices = np.linspace(0, len(months) - 1, 5, dtype=int)
tick_labels = [months[i] for i in tick_indices]

ax.set_xticks(tick_indices)
ax.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=hzysfontsize)

# Move X-axis labels down
padding_value = 10 
ax.tick_params(axis='x', which='major', pad=padding_value)

# --- [Critical] Manually lock margins ---
plt.subplots_adjust(left=0.16, right=0.95, top=0.92, bottom=0.25)

# --- 5. Saving ---
output_dir = './figs'
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'consolidates_monthly.pdf')

plt.savefig(save_path, format='pdf', dpi=300)
print(f"Bar chart with Mean line saved: {save_path}")