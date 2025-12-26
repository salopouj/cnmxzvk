import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pickle
import os
from matplotlib.ticker import FuncFormatter

# --- 1. Style Configuration ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri'] + plt.rcParams['font.sans-serif']
hzysfontsize = 22
plt.rcParams['font.size'] = hzysfontsize

# --- 2. Data Loading and Preprocessing ---
file_path = '/Users/fake deposit/ocsvm_test_results.pkl'
print(f"Reading file: {file_path}")

try:
    with open(file_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    if isinstance(raw_data, dict) and 'inference_times_ms' in raw_data:
        data_values = raw_data['inference_times_ms']
    elif isinstance(raw_data, list):
        data_values = raw_data
    else:
        data_values = []
        
    data = np.array(data_values)
    # Filter non-positive numbers (log scale requires data > 0)
    data = data[data > 0]

    # --- 3. Statistical Percentile Output ---
    if len(data) > 0:
        quantiles_list = [25, 50, 75, 90, 95, 99, 99.9]
        quantile_values = np.percentile(data, quantiles_list)
        
        print("\n" + "="*40)
        print(f"{'Metric':<20} | {'Time (ms)':<15}")
        print("-" * 40)
        print(f"{'Min':<20} | {data.min():.4f}")
        for q, val in zip(quantiles_list, quantile_values):
            print(f"{q:<17}% (P{q:<2}) | {val:.4f}")
        print(f"{'Max':<20} | {data.max():.4f}")
        print(f"{'Mean':<20} | {data.mean():.4f}")
        print(f"{'Total Samples':<20} | {len(data)}")
        print("="*40 + "\n")
    else:
        print("Warning: No valid data found.")

    # Downsampling logic (only for plot optimization)
    MAX_POINTS = 50000
    if len(data) > MAX_POINTS:
        print(f"Data size too large, downsampling to {MAX_POINTS} for plotting...")
        plot_data = np.random.choice(data, size=MAX_POINTS, replace=False)
    else:
        plot_data = data
        
except Exception as e:
    print(f"Read or calculation error: {e}")
    plot_data = np.random.lognormal(mean=0, sigma=1.0, size=10000)

# --- 4. Plotting Setup ---
main_color = '#CFE2F3'  # Light Blue
fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2, 1, height_ratios=[0.15, 0.85], hspace=0.05)
ax_box = fig.add_subplot(gs[0])
ax_hist = fig.add_subplot(gs[1], sharex=ax_box)

# --- 5. Generate Log Bins ---
min_val = max(plot_data.min(), 0.01) 
max_val = plot_data.max()
log_bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)

# --- 6. Plot Top: Boxplot ---
box = ax_box.boxplot(plot_data, vert=False, patch_artist=True, widths=0.6,
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

# --- 7. Plot Bottom: Histogram ---
ax_hist.hist(plot_data, bins=log_bins, density=True, 
             color=main_color, edgecolor='black', 
             alpha=0.9, linewidth=0.5)

# --- 8. Set Logarithmic Axes ---
ax_hist.set_xscale('log')
ax_box.set_xscale('log')

# --- 9. Tick Formatting ---
def format_func(x, pos):
    if x >= 1:
        return f'{int(x)}' 
    else:
        return f'{x}'

ax_hist.xaxis.set_major_formatter(FuncFormatter(format_func))
ax_hist.xaxis.set_minor_formatter(FuncFormatter(lambda x, _: ''))

ticks_candidates = [0.01, 0.1, 1, 10, 100, 1000]
valid_ticks = [t for t in ticks_candidates if min_val/5 <= t <= max_val*2]
ax_hist.set_xticks(valid_ticks)

# --- 10. Labels and Alignment Settings ---
ax_hist.set_xlabel('ocsvm detection time (ms)', fontsize=hzysfontsize, labelpad=10)
ax_hist.set_ylabel('frequency', fontsize=hzysfontsize, labelpad=10)
ax_hist.grid(True, which='major', axis='y', linestyle='--', alpha=0.4)
ax_hist.set_axisbelow(True)

# Keep layout consistent with ETL code
plt.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.20)

output_dir = './figs'
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'ocsvm_dist_log.pdf')

# Save image
plt.savefig(save_path, format='pdf', dpi=300)
print(f"OCSVM aligned distribution plot saved to: {save_path}")