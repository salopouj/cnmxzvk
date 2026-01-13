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
months = ['25-Jan', '25-Feb', '25-Mar', '25-Apr', '25-May', '25-Jun', '25-Jul', '25-Aug', '25-Sep']
updates = [16, 29, 47, 63, 55, 58, 55, 46, 40]

# --- 3. Plotting ---
fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(months, updates, color='#FA7F6F', linewidth=3, marker='o', markersize=10)

for x, y in zip(months, updates):
    ax.text(x, y + 3, str(y), ha='center', va='bottom', fontsize=hzysfontsize)

# --- 4. Axes and Details ---
# To make the line chart X-axis text visually close to the bar chart (since bar chart text is rotated and takes space)
# We keep some padding, but rely on subplots_adjust to ensure axis alignment
padding_value = 10
plt.xticks(rotation=30, ha='right')
ax.tick_params(axis='x', which='major', pad=padding_value)

# ax.set_xlabel('Month', fontsize=hzysfontsize)
ax.set_ylabel('# of Updates', fontsize=hzysfontsize)
ax.grid(True, which='major', linestyle='--', alpha=0.6)

ax.set_ylim(0, max(updates) * 1.2)
ax.set_xlim(-0.5, len(months) - 0.5)

# --- [Critical Modification] Manually lock margins ---
# Must be exactly consistent with the bar chart!
plt.subplots_adjust(left=0.16, right=0.95, top=0.92, bottom=0.25)

# --- 5. Saving ---
output_dir = './figs'
os.makedirs(output_dir, exist_ok=True)

# [Critical Modification] Remove bbox_inches='tight'
plt.savefig(os.path.join(output_dir, 'updates_trend.pdf'), format='pdf', dpi=300)

print(f"Line chart (aligned) saved")