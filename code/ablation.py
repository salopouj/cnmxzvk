import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from matplotlib.ticker import SymmetricalLogLocator

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri'] + plt.rcParams['font.sans-serif']

hzysfontsize = 22
plt.rcParams['font.size'] = hzysfontsize

# --- 2. Data Preparation ---
labels = [f'#{i}' for i in range(1, 11)]

# Updated data
data = {
    'MOT':       [60, 6, 19, 97, 4, 1, 103, 2, 14, 2206],
    'MOT_dl':    [60, 6, 19, 97, 4, 1, 103, 2, 14, 0],
    'MOT_svm':   [53, 4, 13, 37, 4, 0, 84, 1, 5, 2206]
}

# Mappings for colors, markers, and linestyles
colors = {'MOT': '#FA7F6F', 'MOT_dl': '#82B0D2', 'MOT_svm': '#FFBE7A'}
markers = {'MOT': 'o', 'MOT_dl': 's', 'MOT_svm': '^'}
# [Key Modification 1] Set different linestyle for each line
# '--': dashed, '-.': dash-dot, ':': dotted
linestyles = {'MOT': ':', 'MOT_dl': '-.', 'MOT_svm': '--'}

# [Key Modification 2] Set plotting order (zorder); larger values are drawn on top
# MOT_dl overlaps mostly with MOT, so set its zorder lower than MOT to let MOT show on top
zorders = {'MOT': 3, 'MOT_dl': 2, 'MOT_svm': 1}

# --- 3. Plotting ---
fig, ax = plt.subplots(figsize=(10, 4))

for name, values in data.items():
    # --- Logic for label superscript ---
    if '_' in name:
        base, suffix = name.split('_')
        # Use LaTeX syntax to convert suffix to superscript, e.g., MOT^{dl}
        label_display = f'{base}$^{{{suffix}}}$'
    else:
        label_display = name

    ax.plot(labels, values, 
            color=colors[name], 
            linewidth=3,
            marker=markers[name],
            markersize=10,
            linestyle=linestyles[name],  # Use custom linestyle
            zorder=zorders[name],        # Set layer order (zorder)
            label=label_display)

# --- 4. Axis Settings (Symlog) ---
ax.set_yscale('symlog', linthresh=1)
ax.set_yticks([0, 1, 10, 100, 500, 2500])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# --- 5. Style Details Adjustment ---

# Legend horizontally at the top
ax.legend(fontsize=18, 
          frameon=False, 
          loc='lower center', 
          bbox_to_anchor=(0.5, 0.7), 
          ncol=3, 
          columnspacing=1.5)

# Move X-axis labels down
padding_value = 35 
ax.tick_params(axis='x', which='major', pad=padding_value)

ax.set_xlabel('Exploitation types', fontsize=hzysfontsize, labelpad=10)
ax.set_ylabel('# of detect (log)', fontsize=hzysfontsize)

# Grid lines
ax.grid(True, which='major', linestyle='--', alpha=0.6)

# Set ranges
ax.set_ylim(-0.5, 4000)
ax.set_xlim(-0.5, len(labels) - 0.5)

# --- 6. Saving ---
plt.tight_layout()
output_dir = './figs'
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, 'exp_ablation.pdf')
plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')

print(f"Line chart with legend on top saved to {save_path}")