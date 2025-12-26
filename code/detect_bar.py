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
categories = ['BTC', 'ETH', 'TON', 'DOT', 'COSMOS', 'SOL']
avg_values = [24.9141, 31.6763, 27.6296, 16.0191, 23.5416, 17.0276]

# Define colors
main_color = '#82B0D2'  
main_hatch = '/'

# --- 3. Create Chart ---
fig, ax = plt.subplots(figsize=(8, 5))

# Draw bar chart
ax.bar(categories, avg_values, 
       color=main_color, 
       hatch=main_hatch, 
       edgecolor='black', 
       linewidth=1.0)

# --- 4. Add Data Labels ---
for i, v in enumerate(avg_values):
    offset = max(avg_values) * 0.02
    label_text = f"{v:.1f}" 
    ax.text(i, v + offset, label_text, ha='center', va='bottom', fontsize=hzysfontsize)

# --- 5. Axis and Detail Settings ---
ax.set_xlabel('Chain Type', fontsize=hzysfontsize, labelpad=10)
ax.set_ylabel('Detection Time (ms)', fontsize=hzysfontsize)

ax.grid(True, which='major', linestyle='--', alpha=0.6, axis='y')
ax.set_ylim(0, max(avg_values) * 1.2)

# Rotate X-axis labels
plt.xticks(rotation=0, ha='center') 

# --- [Critical Alignment Modification 1] Move X-axis labels down (Consistent with CDF plot) ---
padding_value = 35 
ax.tick_params(axis='x', which='major', pad=padding_value)

# --- [Critical Alignment Modification 2] Manually lock margins (Consistent with CDF plot) ---
# left=0.20 to accommodate large font Y-axis label
# bottom=0.25 to accommodate sunken X-axis labels
plt.subplots_adjust(left=0.20, right=0.95, top=0.92, bottom=0.25)

# --- 6. Saving ---
output_dir = './figs'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'chain_detection_time.pdf')

# --- [Critical Alignment Modification 3] Remove bbox_inches='tight' ---
# This ensures output PDF is strictly 8x5 inch, and internal box position is fixed
plt.savefig(output_path, format='pdf', dpi=300) 

print(f"Detection time bar chart (aligned) saved to {output_path}")