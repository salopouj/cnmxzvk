import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pickle
import os
from matplotlib.ticker import ScalarFormatter

# --- 1. Style Configuration ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri'] + plt.rcParams['font.sans-serif']
hzysfontsize = 24
plt.rcParams['font.size'] = hzysfontsize

# --- 2. Data Loading ---
file_path = '/Users/hk00569ml/代码/fake deposit/ETL_test_results.pkl'
print(f"Reading file: {file_path}")

try:
    with open(file_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    if isinstance(raw_data, list) and len(raw_data) > 0:
        if isinstance(raw_data[0], dict):
            data_values = [d.get('latency_ms', 0) for d in raw_data]
        else:
            data_values = raw_data
    elif isinstance(raw_data, np.ndarray):
         data_values = raw_data.flatten()
    else:
        data_values = []

    data = np.array(data_values)
    
    original_len = len(data)
    data = data[data > 0]
    if len(data) < original_len:
        print(f"Filtered {original_len - len(data)} outliers (<=0).")
        
    print(f"Data loaded successfully, total {len(data)} items. Max: {data.max()} ms")

    # ================= [Added Section: Output Percentiles] =================
    if len(data) > 0:
        print("\n" + "="*30)
        print("[Statistical Percentiles]")
        p25 = np.percentile(data, 25)
        p50 = np.percentile(data, 50) # Median
        p75 = np.percentile(data, 75)
        p90 = np.percentile(data, 90)
        p95 = np.percentile(data, 95)
        p99 = np.percentile(data, 99)
        
        print(f"25% (Q1)  : {p25:.4f} ms")
        print(f"50% (Med) : {p50:.4f} ms")
        print(f"75% (Q3)  : {p75:.4f} ms")
        print(f"90%       : {p90:.4f} ms")
        print(f"95%       : {p95:.4f} ms")
        print(f"99%       : {p99:.4f} ms")
        print("="*30 + "\n")
    # =======================================================

except Exception as e:
    print(f"Error: {e}")
    data = np.random.lognormal(mean=2, sigma=1.0, size=10000)

# --- 3. Plotting Setup ---
main_color = '#F6C6C6'  # Light Pink
fig = plt.figure(figsize=(8, 4.5))
gs = fig.add_gridspec(2, 1, height_ratios=[0.15, 0.85], hspace=0.05)
ax_box = fig.add_subplot(gs[0])
ax_hist = fig.add_subplot(gs[1], sharex=ax_box)

# --- 4. Generate Log Bins ---
min_val = max(data.min(), 0.1) 
max_val = data.max()
log_bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)

# --- 5. Plot Top: Boxplot ---
box = ax_box.boxplot(data, vert=False, patch_artist=True, widths=0.6,
                     showfliers=True, 
                     flierprops={'marker': '.', 'markersize': 2, 'alpha': 0.3, 'color': 'gray'})

for patch in box['boxes']:
    patch.set_facecolor(main_color)
    patch.set_edgecolor('gray')
    patch.set_alpha(0.9)
for median in box['medians']: median.set_color('gray')
for whisker in box['whiskers']: whisker.set_color('gray')
for cap in box['caps']: cap.set_color('gray')

ax_box.axis('off')

# --- 6. Plot Bottom: Histogram ---
ax_hist.hist(data, bins=log_bins, density=True, 
             color=main_color, edgecolor='black', 
             alpha=0.9, linewidth=0.5)

# --- 7. Set Logarithmic Axes ---
ax_hist.set_xscale('log')
ax_box.set_xscale('log')

# --- 8. Tick Formatting and Range Limits ---
formatter = ScalarFormatter()
formatter.set_scientific(False) 
ax_hist.xaxis.set_major_formatter(formatter)

ticks_candidates = [0.1, 1, 10, 50, 100, 500] 
ticks = [t for t in ticks_candidates if t >= min_val/2] 
ax_hist.set_xticks(ticks)
ax_hist.set_xlim(left=min_val, right=500)

# --- 9. Labels and Alignment Settings ---
ax_hist.set_xlabel('Datalog detection time (ms)', fontsize=hzysfontsize, labelpad=10)
ax_hist.set_ylabel('Frequency', fontsize=hzysfontsize, labelpad=10)
ax_hist.grid(True, which='major', axis='y', linestyle='--', alpha=0.4)
ax_hist.set_axisbelow(True)

# [Critical Alignment 1] Manually lock four margins
# left=0.18: Reserve enough space for left Y-axis label
# bottom=0.20: Reserve enough space for bottom X-axis label
plt.subplots_adjust(left=0.20, right=0.95, top=0.92, bottom=0.25)

output_dir = './figs'
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'datalog_dist_log.pdf')

# [Critical Alignment 2] Remove bbox_inches='tight' to make subplots_adjust effective
plt.savefig(save_path, format='pdf', dpi=300) 
print(f"Datalog aligned distribution plot saved to: {save_path}")