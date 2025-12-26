import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from matplotlib.ticker import SymmetricalLogLocator

# --- 1. Style Configuration ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri'] + plt.rcParams['font.sans-serif']

hzysfontsize = 22
plt.rcParams['font.size'] = hzysfontsize

# --- 2. Data Preparation ---
labels = [f'#{i}' for i in range(1, 11)]

data = {
    'MOT':        [60, 6, 19, 97, 4, 1, 103, 2, 14, 2206],
    'Baseline_1': [9, 1, 3, 12, 0, 1, 23, 0, 5, 322],
    'Baseline_2': [0, 0, 19, 0, 0, 1, 0, 2, 14, 0]
}

# 1. Calculate sums
sum_mot = sum(data['MOT'])
sum_b1 = sum(data['Baseline_1'])
sum_b2 = sum(data['Baseline_2'])

print(f"MOT Sum (Baseline): {sum_mot}")
print(f"Baseline_1 Sum: {sum_b1}")
print(f"Baseline_2 Sum: {sum_b2}")

# 2. Calculate detection rates
rate_b1 = sum_b1 / sum_mot
rate_b2 = sum_b2 / sum_mot

print("-" * 30)
print(f"Baseline_1 Detection Rate: {rate_b1:.2%} (i.e., {rate_b1:.4f})")
print(f"Baseline_2 Detection Rate: {rate_b2:.2%} (i.e., {rate_b2:.4f})")

colors = {'MOT': '#FA7F6F', 'Baseline_1': '#82B0D2', 'Baseline_2': '#FFBE7A'}
markers = {'MOT': 'o', 'Baseline_1': 's', 'Baseline_2': '^'}

# --- 3. Plotting ---
fig, ax = plt.subplots(figsize=(10, 4))

for name, values in data.items():
    
    # --- Label processing logic ---
    if '_' in name:
        base, suffix = name.split('_')
        # Generate LaTeX format string, e.g., Baseline^{1}
        label_display = f'{base}$^{{{suffix}}}$'
    else:
        label_display = name
    
    ax.plot(labels, values, 
            color=colors[name], 
            linewidth=3,
            marker=markers[name],
            markersize=10,
            linestyle='--', 
            label=label_display)  # [Correction]: Use label_display here

# --- 4. Axis Settings (Symlog) ---
ax.set_yscale('symlog', linthresh=1)
ax.set_yticks([0, 1, 10, 100, 500, 2500])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# --- 5. Style Details Adjustment ---

ax.legend(fontsize=18, 
          frameon=False, 
          loc='lower center', 
          bbox_to_anchor=(0.5, 0.7), 
          ncol=3, 
          columnspacing=1.5)

# Move X-axis labels down
padding_value = 35 
ax.tick_params(axis='x', which='major', pad=padding_value)

ax.set_xlabel('Exploitation type', fontsize=hzysfontsize, labelpad=10)
ax.set_ylabel('# of detect (log)', fontsize=hzysfontsize)

ax.grid(True, which='major', linestyle='--', alpha=0.6)

ax.set_ylim(-0.5, 4000)
ax.set_xlim(-0.5, len(labels) - 0.5)

# --- 6. Saving ---
plt.tight_layout()
output_dir = './figs'
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, 'exp_baselines.pdf')
plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')

print(f"Corrected line chart saved to {save_path}")