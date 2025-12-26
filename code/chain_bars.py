import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

# --- 1. Style Configuration ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri'] + plt.rcParams['font.sans-serif']
hzysfontsize = 22
plt.rcParams['font.size'] = hzysfontsize

# --- 2. Data Preparation ---
categories = ['BTC-like', 'ETH-like', 'COSMOS-like', 
              'DOT-like', 'TON-like', 'OTHERS']
base_values = [13, 68, 21, 6, 2, 20]
top_values = [0, 0, 0, 0, 0, 8]
total_values = [b + t for b, t in zip(base_values, top_values)]

main_color = '#82B0D2'  
main_hatch = '/'
special_color = '#FA7F6F' 
special_hatch = 'x'

# --- 3. Plotting ---
fig, ax = plt.subplots(figsize=(8, 5))

# Draw bars
ax.bar(categories, base_values, color=main_color, hatch=main_hatch, label='_nolegend_')
ax.bar(categories, top_values, bottom=base_values, color=special_color, hatch=special_hatch, label='Explicitly state update')

# Add labels
for i, total in enumerate(total_values):
    ax.text(i, total + 0.5, str(total), ha='center', va='bottom', fontsize=hzysfontsize)
ax.text(len(categories) - 1, 24, '8', ha='center', va='center', fontsize=hzysfontsize, color='black')

# --- 4. Axes and Details ---

# [Key Modification 1] Move X-axis title up (Chain Type)
# labelpad is positive by default (approx 4.0).
# Setting it to 0 or negative forces "Chain Type" upwards, closer to tick labels.
# Adjust this value based on actual results; -10 to -15 usually resolves clipping issues.
ax.set_xlabel('Chain Type', fontsize=hzysfontsize, labelpad=-12)

ax.set_ylabel('Chains Count', fontsize=hzysfontsize)
ax.grid(True, which='major', linestyle='--', alpha=0.6, axis='y')

ax.legend(loc='upper right', fontsize=hzysfontsize, ncol=1, columnspacing=0.6, 
          handletextpad=0.5, handlelength=1.0, frameon=False)

ax.set_ylim(0, max(total_values) * 1.15)

# --- [Key Modification 2] Move tick labels up (BTC-like...) ---
plt.xticks(rotation=20, ha='right')
# pad=-2 lifts tick text slightly to make room for "Chain Type" below
ax.tick_params(axis='x', which='major', pad=-2)

# --- 5. Saving ---
# Keep alignment parameters unchanged
plt.subplots_adjust(left=0.16, right=0.95, top=0.92, bottom=0.25)

output_dir = './figs'
os.makedirs(output_dir, exist_ok=True)

# Remove bbox_inches='tight'
plt.savefig(os.path.join(output_dir, 'chain_comparison_bar.pdf'), format='pdf', dpi=300)

print(f"Bar chart (Label Moved Up Corrected Version) saved")