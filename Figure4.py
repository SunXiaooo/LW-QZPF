import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, lfilter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_FILE = Path("./dataset/BCGIPG/250618_0942_JXY.xlsx")
IPG_OUTPUT_FILE = Path("./result/Figure4_IPG.jpg")
BCG_OUTPUT_FILE = Path("./result/Figure4_BCG.jpg")
SAMPLING_FREQUENCY = 200


def differentiate(signal):
    return np.gradient(np.asarray(signal, dtype=float), 1 / SAMPLING_FREQUENCY)


def butterworth_filter(signal, lowcut, highcut, order=3, zero_phase=True):
    nyquist = 0.5 * SAMPLING_FREQUENCY
    coefficients = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    signal = np.asarray(signal, dtype=float)
    if zero_phase:
        return filtfilt(*coefficients, signal)
    return lfilter(*coefficients, signal)


def get_single_cardiac_cycle_distance(ipg_raw):
    window_length = SAMPLING_FREQUENCY * 3
    ipg_window = np.asarray(ipg_raw[:window_length], dtype=float)
    filtered = butterworth_filter(ipg_window, 0.5, 5, zero_phase=False)
    second_derivative = differentiate(differentiate(filtered))
    valleys, _ = find_peaks(
        -second_derivative,
        distance=int(SAMPLING_FREQUENCY * 0.4),
        prominence=0.05,
    )

    valley_values = np.asarray(second_derivative[valleys])
    candidates = valley_values[
        valley_values < np.percentile(second_derivative, 9)
    ]
    if candidates.size < 2:
        candidates = valley_values
        first_value = candidates.min()
        first_index = np.where(valley_values == first_value)[0][0]
        left = valley_values[first_index - 1] if first_index > 0 else None
        right = (
            valley_values[first_index + 1]
            if first_index + 1 < len(valley_values)
            else None
        )
    else:
        first_value = candidates.min()
        first_index = np.where(valley_values == first_value)[0][0]
        candidate_index = np.where(candidates == first_value)[0][0]
        left = candidates[candidate_index - 1] if candidate_index > 0 else None
        right = (
            candidates[candidate_index + 1]
            if candidate_index + 1 < len(candidates)
            else None
        )

    if left is None:
        second_value = right
    elif right is None:
        second_value = left
    else:
        second_value = min(left, right)

    second_index = np.where(valley_values == second_value)[0][0]
    return abs(int(valleys[second_index] - valleys[first_index]))


def hann_blend_components(previous_overlap, current_overlap):
    previous_overlap = np.asarray(previous_overlap, dtype=float)
    current_overlap = np.asarray(current_overlap, dtype=float)
    overlap_size = len(previous_overlap)
    if overlap_size == 1:
        weight_previous = np.array([0.5])
        weight_current = np.array([0.5])
    else:
        hann_window = np.hanning(2 * overlap_size)
        weight_current = hann_window[:overlap_size]
        weight_previous = hann_window[overlap_size:]

    weighted_previous = previous_overlap * weight_previous
    weighted_current = current_overlap * weight_current
    blended = (weighted_previous + weighted_current) / (
        weight_previous + weight_current
    )
    return {
        "weight_previous": weight_previous,
        "weight_current": weight_current,
        "weighted_previous": weighted_previous,
        "weighted_current": weighted_current,
        "blended": blended,
    }


def get_ipg_two_window_data(ipg_raw, cycle_length, overlap_size):
    initial_length = SAMPLING_FREQUENCY * 3
    previous_raw = ipg_raw[
        initial_length - overlap_size : initial_length
    ].astype(float)
    previous_derivative = differentiate(previous_raw)
    simulation_data = ipg_raw[initial_length:].astype(float)

    first_buffer = differentiate(simulation_data[:cycle_length])
    first_window = butterworth_filter(
        np.concatenate((previous_derivative, first_buffer)), 0.5, 5
    )
    first_output = first_window[overlap_size:].copy()

    second_previous = first_buffer[-overlap_size:].copy()
    second_buffer = differentiate(
        simulation_data[cycle_length : 2 * cycle_length].copy()
    )
    second_window = butterworth_filter(
        np.concatenate((second_previous, second_buffer)), 0.5, 5
    )

    previous_overlap = first_output[-overlap_size:].copy()
    current_overlap = second_window[:overlap_size].copy()
    blend = hann_blend_components(previous_overlap, current_overlap)
    fused_output = np.concatenate(
        (
            first_output[:-overlap_size],
            blend["blended"],
            second_window[overlap_size:],
        )
    )
    return {
        "previous_segment": first_output,
        "current_segment": second_window,
        "previous_overlap": previous_overlap,
        "current_overlap": current_overlap,
        "fused_output": fused_output,
        **blend,
    }


def get_bcg_two_window_data(bcg_raw, cycle_length, overlap_size):
    initial_length = SAMPLING_FREQUENCY * 3
    previous_raw = bcg_raw[
        initial_length - overlap_size : initial_length
    ].astype(float)
    simulation_data = bcg_raw[initial_length:].astype(float)

    first_buffer = simulation_data[:cycle_length].copy()
    first_window = butterworth_filter(
        np.concatenate((previous_raw, first_buffer)), 0.9, 9
    )
    first_output = first_window[overlap_size:].copy()

    second_previous = first_buffer[-overlap_size:].copy()
    second_buffer = simulation_data[cycle_length : 2 * cycle_length].copy()
    second_window = butterworth_filter(
        np.concatenate((second_previous, second_buffer)), 0.9, 9
    )

    previous_overlap = first_output[-overlap_size:].copy()
    current_overlap = second_window[:overlap_size].copy()
    blend = hann_blend_components(previous_overlap, current_overlap)
    fused_output = np.concatenate(
        (
            first_output[:-overlap_size],
            blend["blended"],
            second_window[overlap_size:],
        )
    )
    return {
        "previous_segment": first_output,
        "current_segment": second_window,
        "previous_overlap": previous_overlap,
        "current_overlap": current_overlap,
        "fused_output": fused_output,
        **blend,
    }


def plot_hann_stitch_figure(data, signal_label, output_file):
    font_size_tick = 10
    font_size_label = 12
    font_size_legend = 11
    color_previous = "#004488"
    color_current = "#CC5500"
    color_fused = "#CC0000"
    color_overlap = "#D9EAF7"
    color_boundary = "#666666"

    previous_segment = data["previous_segment"]
    current_segment = data["current_segment"]
    previous_overlap = data["previous_overlap"]
    current_overlap = data["current_overlap"]
    overlap_size = len(previous_overlap)

    previous_x = np.arange(len(previous_segment)) / SAMPLING_FREQUENCY
    current_start = (len(previous_segment) - overlap_size) / SAMPLING_FREQUENCY
    current_x = current_start + np.arange(len(current_segment)) / SAMPLING_FREQUENCY
    overlap_x = current_start + np.arange(overlap_size) / SAMPLING_FREQUENCY
    fused_x = np.arange(len(data["fused_output"])) / SAMPLING_FREQUENCY
    normalized_x = np.linspace(0, 1, overlap_size)
    overlap_start, overlap_end = overlap_x[[0, -1]]

    figure = plt.figure(figsize=(12, 10), dpi=300)
    grid = figure.add_gridspec(
        4,
        2,
        height_ratios=[1.35, 1.0, 0.9, 1.35],
        hspace=0.55,
        wspace=0.32,
    )

    axis = figure.add_subplot(grid[0, :])
    axis.plot(previous_x, previous_segment, linewidth=1.8, color=color_previous, label="Previous filtered segment")
    axis.plot(current_x, current_segment, linewidth=1.8, color=color_current, label="Current filtered segment")
    axis.axvspan(overlap_start, overlap_end, color=color_overlap, alpha=0.12)
    axis.axvline(overlap_start, linestyle="--", color=color_boundary, linewidth=1.0)
    axis.axvline(overlap_end, linestyle="--", color=color_boundary, linewidth=1.0)
    axis.set_title(f"(a) Filtered segments with overlap ({signal_label})", loc="left", fontweight="bold")
    axis.set_xlabel("Time (s)", fontsize=font_size_label)
    axis.set_ylabel("Amplitude", fontsize=font_size_label)
    axis.tick_params(axis="both", labelsize=font_size_tick)
    combined = np.concatenate((previous_segment, current_segment))
    y_range = combined.max() - combined.min()
    axis.set_ylim(combined.min() - 0.08 * y_range, combined.max() + 0.30 * y_range)
    axis.legend(frameon=False, ncol=2, fontsize=font_size_legend, loc="upper right")

    axis = figure.add_subplot(grid[1, :])
    axis.plot(normalized_x, data["weight_previous"], linewidth=1.9, color=color_previous, label=r"$w_{prev}$")
    axis.plot(normalized_x, data["weight_current"], linewidth=1.9, color=color_current, label=r"$w_{curr}$")
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.02, 1.05)
    axis.set_title("(b) Hann cross-fade weights used by the algorithm", loc="left", fontweight="bold")
    axis.set_xlabel("Normalized overlap position", fontsize=font_size_label)
    axis.set_ylabel("Weight", fontsize=font_size_label)
    axis.legend(frameon=False, ncol=2, fontsize=font_size_legend, loc="upper center")
    axis.tick_params(axis="both", labelsize=font_size_tick)

    axis = figure.add_subplot(grid[2, 0])
    axis.plot(overlap_x, data["weighted_previous"], color=color_previous, linewidth=1.8)
    axis.set_title("(c) Previous overlap weighted by Hann window", loc="left", fontweight="bold")
    axis.set_xlabel("Time (s)", fontsize=font_size_label)
    axis.set_ylabel("Amplitude", fontsize=font_size_label)
    axis.tick_params(axis="both", labelsize=font_size_tick)

    axis = figure.add_subplot(grid[2, 1])
    axis.plot(overlap_x, data["weighted_current"], color=color_current, linewidth=1.8)
    axis.set_title("(d) Current overlap weighted by Hann window", loc="left", fontweight="bold")
    axis.set_xlabel("Time (s)", fontsize=font_size_label)
    axis.set_ylabel("Amplitude", fontsize=font_size_label)
    axis.tick_params(axis="both", labelsize=font_size_tick)

    axis = figure.add_subplot(grid[3, :])
    axis.plot(fused_x, data["fused_output"], color=color_fused, linewidth=2.0, label="Fused output")
    axis.plot(overlap_x, previous_overlap, color=color_previous, linewidth=1.0, alpha=0.45)
    axis.plot(overlap_x, current_overlap, color=color_current, linewidth=1.0, alpha=0.45)
    axis.axvspan(overlap_start, overlap_end, alpha=0.12)
    axis.axvline(overlap_start, linestyle="--", linewidth=1.0)
    axis.axvline(overlap_end, linestyle="--", linewidth=1.0)
    axis.set_title("(e) Fused output after normalized Hann blending", loc="left", fontweight="bold")
    axis.set_xlabel("Time (s)", fontsize=font_size_label)
    axis.set_ylabel("Amplitude", fontsize=font_size_label)
    axis.legend(frameon=False, fontsize=font_size_legend, loc="upper right")
    axis.tick_params(axis="both", labelsize=font_size_tick)

    figure.subplots_adjust(top=0.94, bottom=0.07, left=0.08, right=0.98)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(figure)


data = pd.read_excel(INPUT_FILE)
ipg_raw = -(data["IPG"].to_numpy(float) - data["IPG"].iloc[0])
bcg_raw = -(data["BCG"].to_numpy(float) - data["BCG"].iloc[0])
cycle_length = get_single_cardiac_cycle_distance(ipg_raw)
overlap_size = math.ceil(cycle_length / 2)

plot_hann_stitch_figure(
    get_ipg_two_window_data(ipg_raw, cycle_length, overlap_size),
    "DIPG",
    IPG_OUTPUT_FILE,
)
plot_hann_stitch_figure(
    get_bcg_two_window_data(bcg_raw, cycle_length, overlap_size),
    "BCG",
    BCG_OUTPUT_FILE,
)
