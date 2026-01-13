import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from matplotlib.ticker import FuncFormatter

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri'] + plt.rcParams['font.sans-serif']

hzysfontsize = 21
plt.rcParams['font.size'] = hzysfontsize

labels = [f'#{i}' for i in range(1, 11)]


raw_data = {
    'MOT':        [60, 1,  2206, 19, 2, 14, 6, 97, 4, 103],
    'Baseline_1': [6,  0,  201,  2, 0,  1, 0, 8, 0,  23], 
    'Baseline_2': [0,  0,    0,  19, 2, 14, 0,  0, 0,   0]
}

print("recall rates:",sum(raw_data['Baseline_1'])/sum(raw_data['MOT']),
      sum(raw_data['Baseline_2'])/sum(raw_data['MOT']))

print(" rates:",1-sum(raw_data['Baseline_1'])/sum(raw_data['MOT']),
      1-sum(raw_data['Baseline_2'])/sum(raw_data['MOT']))


mot_values = np.array(raw_data['MOT'])
data = {}
for key, values in raw_data.items():
    safe_mot = np.where(mot_values == 0, 1, mot_values)
    data[key] = np.array(values) / safe_mot

colors = {'MOT': '#FA7F6F', 'Baseline_1': '#82B0D2', 'Baseline_2': '#FFBE7A'}
hatches = {'MOT': '///', 'Baseline_1': '\\\\\\', 'Baseline_2': '...'}

x = np.arange(len(labels))
width = 0.25 
bar_keys = ['MOT', 'Baseline_1', 'Baseline_2']

fig, ax = plt.subplots(figsize=(10, 3.3))

for i, name in enumerate(bar_keys):
    offset = (i - 1) * width

    if '_' in name:
        base, suffix = name.split('_')
        label_display = f'{base}$^{{{suffix}}}$'
    else:
        label_display = name

    ax.bar(x + offset, data[name], width, 
           label=label_display, 
           color=colors[name], 
           hatch=hatches[name], 
           edgecolor='black', 
           linewidth=1.0, 
           zorder=3)


ax.set_yscale('linear')
ax.set_yticks([0, 0.5, 1.0])

def to_percent(y, position):
    return f"{int(y * 100)}%"
ax.yaxis.set_major_formatter(FuncFormatter(to_percent))

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.legend(fontsize=18, 
          frameon=False, 
          loc='upper center', 
          bbox_to_anchor=(0.5, 1.07), 
          ncol=3, 
          columnspacing=1.5)

padding_value = 10 
ax.tick_params(axis='x', which='major', pad=padding_value)

ax.set_xlabel('Exploitation types', fontsize=hzysfontsize, labelpad=10)
ax.set_ylabel('Recall Rate', fontsize=hzysfontsize)

ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.6, zorder=0)


ax.set_ylim(0, 1.3)
ax.set_xlim(-0.6, len(labels) - 0.4)

plt.tight_layout()
output_dir = './figs'
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, 'exp_baselines.pdf')
plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')

print(f"Verified Baseline Bar Chart saved to {save_path}")
