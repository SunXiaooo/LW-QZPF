import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, lfilter


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

SAMPLING_FREQUENCY = 200
SOURCE_INDEX = 196
INPUT_FILE = Path("./dataset/BCGIPG/250618_0942_JXY.xlsx")


def normalize_signal(signal, new_range=(0, 100)):
    minimum = np.min(signal)
    maximum = np.max(signal)
    old_range = maximum - minimum
    if old_range == 0:
        return np.zeros_like(signal)
    return (
        (signal - minimum) * (new_range[1] - new_range[0]) / old_range
        + new_range[0]
    )


def differentiate(signal, sampling_frequency=SAMPLING_FREQUENCY):
    return np.gradient(signal, 1 / sampling_frequency)


def butterworth_filter(
    signal,
    lowcut,
    highcut,
    order=3,
    sampling_frequency=SAMPLING_FREQUENCY,
    zero_phase=True,
):
    nyquist = 0.5 * sampling_frequency
    coefficients = butter(
        order,
        [lowcut / nyquist, highcut / nyquist],
        btype="band",
    )
    if zero_phase:
        return filtfilt(*coefficients, signal)
    return lfilter(*coefficients, signal)


def get_single_cardiac_cycle_distance(ipg_raw):
    initial_window = ipg_raw.iloc[: 3 * SAMPLING_FREQUENCY]
    filtered = butterworth_filter(
        initial_window,
        0.5,
        5,
        zero_phase=False,
    )
    second_derivative = differentiate(differentiate(filtered))
    peaks, _ = find_peaks(
        -second_derivative,
        distance=int(SAMPLING_FREQUENCY * 0.4),
        prominence=0.05,
    )

    peak_values = np.asarray(second_derivative[peaks])
    candidates = peak_values[
        peak_values < np.percentile(second_derivative, 9)
    ]
    if candidates.size < 2:
        candidates = peak_values
        first_value = candidates.min()
        first_index = np.where(peak_values == first_value)[0][0]
        left = peak_values[first_index - 1] if first_index > 0 else None
        right = (
            peak_values[first_index + 1]
            if first_index + 1 < len(peak_values)
            else None
        )
    else:
        first_value = candidates.min()
        first_index = np.where(peak_values == first_value)[0][0]
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
        second_value = left if left < right else right

    second_index = np.where(peak_values == second_value)[0][0]
    return abs(peaks[second_index] - peaks[first_index])


def hann_stitch(output, current_window, overlap_size):
    if not output:
        output.extend(current_window[overlap_size:].tolist())
        return output

    previous_overlap = np.asarray(output[-overlap_size:], dtype=np.float64)
    current_overlap = current_window[:overlap_size]
    if overlap_size == 1:
        blended = (previous_overlap + current_overlap) / 2.0
    else:
        hann = np.hanning(2 * overlap_size)
        current_weight = hann[:overlap_size]
        previous_weight = hann[overlap_size:]
        blended = (
            previous_overlap * previous_weight
            + current_overlap * current_weight
        ) / (previous_weight + current_weight)

    output[-overlap_size:] = blended.tolist()
    output.extend(current_window[overlap_size:].tolist())
    return output


def run_lw_qzpf(
    signal,
    window_size,
    overlap_size,
    lowcut,
    highcut,
    use_derivative=False,
):
    initial_length = 3 * SAMPLING_FREQUENCY
    previous = signal.iloc[
        initial_length - overlap_size : initial_length
    ].to_numpy(dtype=np.float64)
    if use_derivative:
        previous = differentiate(previous)

    output = []
    buffer = np.empty(window_size, dtype=np.float64)
    count = 0
    for sample in signal.iloc[initial_length:]:
        buffer[count] = sample
        count += 1
        if count != window_size:
            continue

        if use_derivative:
            buffer = differentiate(buffer)
        current_window = np.concatenate((previous, buffer))
        current_window = butterworth_filter(
            current_window,
            lowcut,
            highcut,
            zero_phase=True,
        )
        hann_stitch(output, current_window, overlap_size)
        previous = buffer[-overlap_size:].copy()
        count = 0

    return np.asarray(output)


def plot_signal(raw_signal, processed_signal, label_prefix, colors):
    initial_length = 3 * SAMPLING_FREQUENCY
    plt.figure(figsize=(15, 6))
    plt.plot(raw_signal, label=f"{label_prefix} reference", color=colors[0])
    plt.plot(
        np.arange(initial_length, initial_length + len(processed_signal)),
        processed_signal,
        label=f"LW-QZPF {label_prefix}",
        color=colors[1],
    )
    plt.title(f"{INPUT_FILE.stem} (index: {SOURCE_INDEX})")
    plt.subplots_adjust(right=0.80)
    plt.legend(bbox_to_anchor=(1.02, 1), fontsize=12, loc="upper left")
    plt.grid()
    plt.show()

    plt.ioff()
    plt.show(block=True)


frame = pd.read_excel(INPUT_FILE)
ipg_raw = -(frame["IPG"] - frame["IPG"].iloc[0])
bcg_raw = -(frame["BCG"] - frame["BCG"].iloc[0])

# LW-QZPF estimates its adaptive window from the first three seconds of IPG.
window_size = get_single_cardiac_cycle_distance(ipg_raw)
overlap_size = math.ceil(window_size / 2)

# Completed local windows are filtered bidirectionally and joined by a normalized Hann cross-fade.
ipg_realtime = run_lw_qzpf(
    ipg_raw,
    window_size,
    overlap_size,
    0.5,
    5,
    use_derivative=True,
)
bcg_realtime = run_lw_qzpf(
    bcg_raw,
    window_size,
    overlap_size,
    0.9,
    9,
)

plot_signal(
    normalize_signal(differentiate(ipg_raw)),
    normalize_signal(ipg_realtime) - 100,
    "IPG first derivative",
    ("#FF7700", "#EE6600"),
)
plot_signal(
    normalize_signal(bcg_raw),
    normalize_signal(bcg_realtime) - 100,
    "BCG",
    ("#0066CC", "#0055AA"),
)
