import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri'] + plt.rcParams['font.sans-serif']

hzysfontsize = 21
plt.rcParams['font.size'] = hzysfontsize

labels = [f'#{i}' for i in range(1, 11)]

data = {
    'MOT':       [60, 1, 2206, 19, 2, 14, 6, 97, 4, 103],
    'MOT_dl':    [60, 1,    0, 19, 2, 14, 6, 97, 4, 103],
    'MOT_svm':   [53, 0, 2206, 13, 1,  5, 4, 37, 4,  84]
}

colors = {'MOT': '#FA7F6F', 'MOT_dl': '#82B0D2', 'MOT_svm': '#FFBE7A'}
hatches = {'MOT': '///', 'MOT_dl': '\\\\\\', 'MOT_svm': '...'}

x = np.arange(len(labels))
width = 0.25 
bar_keys = list(data.keys()) 

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

def add_text(pos_x, value):
    pos_y = value if value > 0 else 0.1
    
    if int(value) == 2206:
        display_text = '2.2k'
    else:
        display_text = f'{int(value)}'
        
    ax.text(pos_x, pos_y, display_text, 
            ha='center', va='bottom', fontsize=12, color='black')

for i in range(len(labels)):
    v1 = data['MOT'][i]
    v2 = data['MOT_dl'][i]
    v3 = data['MOT_svm'][i]
    
    x1 = x[i] - width
    x2 = x[i]
    x3 = x[i] + width
    
    if v1 == v2 == v3:
        add_text(x2, v2)
        
    elif v1 == v2:
        add_text((x1 + x2) / 2, v1)
        add_text(x3, v3)
        
    elif v2 == v3:
        add_text(x1, v1)
        add_text((x2 + x3) / 2, v2)
        
    else:
        add_text(x1, v1)
        add_text(x2, v2)
        add_text(x3, v3)

ax.set_yscale('symlog', linthresh=1)
ax.set_yticks([0, 1, 10, 100, 2500])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.legend(fontsize=18, 
          frameon=False, 
          loc='upper center', 
          bbox_to_anchor=(0.65, 1.07), 
          ncol=3, 
          columnspacing=1.5)

padding_value = 10 
ax.tick_params(axis='x', which='major', pad=padding_value)

ax.set_xlabel('Exploitation types', fontsize=hzysfontsize, labelpad=10)
ax.set_ylabel('# of detections', fontsize=hzysfontsize)

ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.6, zorder=0)

ax.set_ylim(0, 20000)
ax.set_xlim(-0.6, len(labels) - 0.4)

plt.tight_layout()
output_dir = './figs'
os.makedirs(output_dir, exist_ok=True)

save_path = os.
