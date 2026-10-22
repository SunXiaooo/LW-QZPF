from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, lfilter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_FILE = Path("./dataset/BCGIPG/250408_0919_CYZ.xlsx")
OUTPUT_FILE = Path("./result/Figure2.png")
SAMPLING_FREQUENCY = 200


def normalize_signal(signal, new_range=(0, 100)):
    minimum = np.min(signal)
    data_range = np.max(signal) - minimum
    if data_range == 0:
        return np.zeros_like(signal)
    target_range = new_range[1] - new_range[0]
    return (signal - minimum) * target_range / data_range + new_range[0]


def causal_butterworth_filter(signal, lowcut, highcut, order=3):
    nyquist = 0.5 * SAMPLING_FREQUENCY
    coefficients = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return lfilter(*coefficients, signal)


def differentiate(signal):
    return np.gradient(signal, 1 / SAMPLING_FREQUENCY)


def select_cycle_valleys(signal, peaks):
    peak_values = np.asarray(signal[peaks])
    candidates = peak_values[peak_values < np.percentile(signal, 9)]
    if candidates.size < 2:
        candidates = peak_values
        first_value = candidates.min()
        first_index = np.where(peak_values == first_value)[0][0]
        left = peak_values[first_index - 1] if first_index > 0 else None
        right = peak_values[first_index + 1] if first_index + 1 < len(peak_values) else None
    else:
        first_value = candidates.min()
        first_index = np.where(peak_values == first_value)[0][0]
        candidate_index = np.where(candidates == first_value)[0][0]
        left = candidates[candidate_index - 1] if candidate_index > 0 else None
        right = candidates[candidate_index + 1] if candidate_index + 1 < len(candidates) else None

    if left is None:
        second_value = right
    elif right is None:
        second_value = left
    else:
        second_value = min(left, right)

    second_index = np.where(peak_values == second_value)[0][0]
    return sorted([peaks[first_index], peaks[second_index]])


data = pd.read_excel(INPUT_FILE)
ipg_raw = -(data["IPG"] - data["IPG"].iloc[0])
window_length = SAMPLING_FREQUENCY * 3
window_ipg_raw = ipg_raw.iloc[:window_length]
window_ipg_filtered = causal_butterworth_filter(window_ipg_raw, 0.5, 5)
window_ipg_second_derivative = differentiate(differentiate(window_ipg_filtered))

valleys, _ = find_peaks(
    -window_ipg_second_derivative,
    distance=int(SAMPLING_FREQUENCY * 0.4),
    prominence=0.05,
)
cycle_valleys = select_cycle_valleys(window_ipg_second_derivative, valleys)
ninth_percentile = np.percentile(window_ipg_second_derivative, 9)
accepted_valleys = valleys[window_ipg_second_derivative[valleys] < ninth_percentile]
rejected_valleys = np.setdiff1d(valleys, accepted_valleys)

raw_plot = normalize_signal(window_ipg_raw)
filtered_plot = normalize_signal(window_ipg_filtered)
second_derivative_plot = normalize_signal(window_ipg_second_derivative)
percentile_plot = normalize_signal(
    np.append(window_ipg_second_derivative, ninth_percentile)
)[-1]

warm_colors = ["#FF8811", "#FF7700", "#EE6600", "#CC5500", "#AA4400", "#883300"]
font_size_title = 12
font_size_label = 11
font_size_tick = 10
font_size_legend = 10

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
figure, axes = plt.subplots(
    5,
    1,
    figsize=(10, 10),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1, 1, 1.5, 1.5]},
)

axes[0].plot(raw_plot, linewidth=2, color=warm_colors[3], label="Raw IPG")
axes[0].set_title("(a) IPG Raw Signal (3s)", loc="left", fontsize=font_size_title, fontweight="bold")
axes[0].set_ylabel("Amplitude", fontsize=font_size_label)
axes[0].tick_params(axis="both", labelsize=font_size_tick)
axes[0].legend(loc="upper right", frameon=False, fontsize=font_size_legend)

axes[1].plot(filtered_plot, linewidth=2, color=warm_colors[3], label="Filtered IPG")
axes[1].set_title("(b) IPG Butterworth-Filtered", loc="left", fontsize=font_size_title, fontweight="bold")
axes[1].set_ylabel("Amplitude", fontsize=font_size_label)
axes[1].tick_params(axis="both", labelsize=font_size_tick)
axes[1].legend(loc="upper right", frameon=False, fontsize=font_size_legend)

axes[2].plot(second_derivative_plot, linewidth=2, color=warm_colors[3], label="IPG second derivative")
axes[2].set_title("(c) IPG Second Derivative", loc="left", fontsize=font_size_title, fontweight="bold")
axes[2].set_ylabel("Amplitude", fontsize=font_size_label)
axes[2].tick_params(axis="both", labelsize=font_size_tick)
axes[2].legend(loc="upper right", frameon=False, fontsize=font_size_legend)

axes[3].plot(second_derivative_plot, linewidth=2, color=warm_colors[3])
axes[3].axhline(percentile_plot, linewidth=1.5, linestyle="--", color="#7A3E9D")
axes[3].plot(
    rejected_valleys,
    second_derivative_plot[rejected_valleys],
    ".",
    markersize=12,
    markeredgecolor="black",
    markeredgewidth=0.6,
    color="green",
    label="Rejected candidates",
)
axes[3].plot(
    accepted_valleys,
    second_derivative_plot[accepted_valleys],
    ".",
    markersize=12,
    markeredgecolor="black",
    markeredgewidth=0.6,
    color="red",
    label="Accepted candidates",
)
axes[3].set_ylim(second_derivative_plot.min() - 10, second_derivative_plot.max() + 40)
axes[3].set_title("(d) Valley detection with three constraints", loc="left", fontsize=font_size_title, fontweight="bold")
axes[3].set_ylabel("Amplitude", fontsize=font_size_label)
axes[3].tick_params(axis="both", labelsize=font_size_tick)
axes[3].legend(loc="upper right", frameon=False, fontsize=font_size_legend)

axes[4].plot(second_derivative_plot, linewidth=2, color=warm_colors[3])
first_valley = min(cycle_valleys, key=lambda index: window_ipg_second_derivative[index])
second_valley = cycle_valleys[0] if cycle_valleys[1] == first_valley else cycle_valleys[1]
axes[4].plot(
    [first_valley, second_valley],
    second_derivative_plot[[first_valley, second_valley]],
    ".",
    markersize=12,
    markeredgecolor="black",
    markeredgewidth=0.6,
    color="red",
    label="Accepted candidates",
)
axes[4].set_ylim(second_derivative_plot.min() - 40, second_derivative_plot.max() + 10)
axes[4].set_title("(e) Determination of Cardiac-Cycle Length from Selected Valleys", loc="left", fontsize=font_size_title, fontweight="bold")
axes[4].set_xlabel("Sample Index", fontsize=font_size_label)
axes[4].set_ylabel("Amplitude", fontsize=font_size_label)
axes[4].tick_params(axis="both", labelsize=font_size_tick)
axes[4].legend(loc="upper right", frameon=False, fontsize=font_size_legend)

figure.suptitle(INPUT_FILE.stem, fontsize=13, fontweight="bold")
figure.subplots_adjust(top=0.92, bottom=0.08, left=0.12, right=0.97, hspace=0.35)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.close(figure)
