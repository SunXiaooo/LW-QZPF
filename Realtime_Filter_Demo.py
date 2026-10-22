# Real-time BCG-IPG comparison of IIR, FIR, LW-QZPF, and offline filtfilt.

import os
import math
import time
import numpy as np
import pandas as pd
import matplotlib

# Select the GUI backend before importing pyplot.
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.signal import butter, bessel, filtfilt, lfilter, firwin, find_peaks
from scipy.stats import pearsonr


# Configuration
excel_path = "./dataset/BCGIPG/250618_0942_JXY.xlsx"
sampling_freq = 200
point_delay = 1.0 / sampling_freq
draw_refresh_interval = 0.025
initial_seconds = 3
initial_length = int(sampling_freq * initial_seconds)
IIR_buffer_size = 1
Bessel_buffer_size = 1
FIR_buffer_size = 1
online_detection_hop = 5
fir_numtaps = 201

# Use None to process the complete record.
processing_seconds = None
show_performance_table = True
performance_table_delay = 3.0
startup_delay = 3.0


# Plot configuration
plt.rcParams['axes.unicode_minus'] = False
color_cool = ["#0077EE", "#0066CC", "#0055AA", "#004488", "#003366", "#002244"]
color_warm = ["#FF8811", "#FF7700", "#EE6600", "#CC5500", "#AA4400", "#883300"]


def maximize_matplotlib_window(fig):
    """Maximize a Matplotlib figure window when supported by the active backend."""
    manager = fig.canvas.manager
    window = getattr(manager, "window", None)

    if window is None:
        return

    try:
        window.update_idletasks()

        try:
            window.state("zoomed")
            return
        except Exception:
            pass

        try:
            window.attributes("-zoomed", True)
            return
        except Exception:
            pass

        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        window.geometry(f"{screen_width}x{screen_height}+0+0")
    except Exception:
        pass


def wait_with_gui_updates(duration):
    """Keep Matplotlib responsive while imposing a presentation delay."""
    if duration is None or duration <= 0:
        return

    end_time = time.perf_counter() + duration
    while time.perf_counter() < end_time:
        remaining = end_time - time.perf_counter()
        plt.pause(min(0.05, max(remaining, 0.001)))


def normalize_signal(signal, new_range=(0, 100)):
    """Normalize a signal to the requested range."""
    min_value = np.min(signal)
    max_value = np.max(signal)
    old_range = max_value - min_value
    if old_range == 0:
        return np.zeros_like(signal)
    return ((signal - min_value) * (new_range[1] - new_range[0])) / old_range + new_range[0]

def differentiate(signal,sampling_freq=200):
    """Return the first derivative."""
    dt = 1/sampling_freq
    first_derivative = np.gradient(signal, dt)
    return  first_derivative


def init_streaming_derivative_state(ipg_prev=None, bcg_prev=None):
    """Initialize the state used by the streaming three-point central difference."""
    state = {"ipg": [], "bcg": []}
    if ipg_prev is not None and bcg_prev is not None:
        state["ipg"].append(float(ipg_prev))
        state["bcg"].append(float(bcg_prev))
    return state

def streaming_center_gradient(ipg_filtered_value, bcg_filtered_value, state, sampling_freq=200):
    """Apply a three-point central difference with one-sample confirmation delay."""
    state["ipg"].append(float(ipg_filtered_value))
    state["bcg"].append(float(bcg_filtered_value))

    if len(state["ipg"]) < 3:
        return None, None

    derivative_value = (state["ipg"][-1] - state["ipg"][-3]) * sampling_freq / 2.0
    bcg_center_value = state["bcg"][-2]

    state["ipg"].pop(0)
    state["bcg"].pop(0)

    return derivative_value, bcg_center_value

def butterworth_filter(signal, lowcut, highcut,order=3, sampling_freq=200, mode=0):
    """Apply a Butterworth band-pass filter."""
    fs = sampling_freq
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    if mode==0:
        reconstructed_signal = filtfilt(b, a, signal)
    else:
        reconstructed_signal = lfilter(b, a, signal)
    return reconstructed_signal

def get_SingleCardiacCycleDistance(ipg_raw, sampling_freq=200):
    # Estimate the cardiac-cycle length from the first 3 s of IPG.
    window_size = int(sampling_freq * 3)
    window_IPG_raw = ipg_raw[0:window_size]

    window_IPG_processed = butterworth_filter(window_IPG_raw, 0.5, 5, 3, 200, 1)
    window_IPG_derivative_two = differentiate(differentiate(window_IPG_processed))

    peaks, _ = find_peaks(
        -window_IPG_derivative_two,
        distance=int(sampling_freq * 0.4),
        prominence=0.05
    )

    arr = np.array(window_IPG_derivative_two[peaks])
    candidates = arr[arr < np.percentile(window_IPG_derivative_two, 9)]

    if candidates.size < 2:
        candidates = arr
        first_value = candidates.min()
        first_index = np.where(arr == first_value)[0][0]
        left = arr[first_index - 1] if first_index - 1 >= 0 else None
        right = arr[first_index + 1] if first_index + 1 < len(arr) else None

    else:
        first_value = candidates.min()
        first_index = np.where(arr == first_value)[0][0]
        index = np.where(candidates == first_value)[0][0]
        left = candidates[index - 1] if index - 1 >= 0 else None
        right = candidates[index + 1] if index + 1 < len(candidates) else None

    if left is None:
        second_value = right
    elif right is None:
        second_value = left
    else:
        second_value = left if left < right else right

    second_index = np.where(arr == second_value)[0][0]

    if peaks[first_index] < peaks[second_index]:
        r_first_index = peaks[first_index]
        r_second_index = peaks[second_index]
    else:
        r_first_index = peaks[second_index]
        r_second_index = peaks[first_index]

    return r_second_index - r_first_index

def Realtime_Processed(signal_fragment, lowcut=0.5, highcut=5):
    window_filtered = butterworth_filter(signal_fragment, lowcut, highcut, 3, 200, 0)
    return pd.Series(window_filtered)

def Hann_Stitch(IPG_realtime, current_data, overlap_size):
    if len(IPG_realtime) == 0:
        IPG_realtime.extend(current_data[overlap_size:].tolist())
        return IPG_realtime

    previous_overlap = np.asarray(IPG_realtime[-overlap_size:],dtype=np.float64)
    current_overlap = current_data[:overlap_size]
    if overlap_size == 1:
        blended = (previous_overlap + current_overlap) / 2.0
    else:
        hann = np.hanning(2 * overlap_size)
        weight_current = hann[:overlap_size]
        weight_previous = hann[overlap_size:]
        weight_sum = weight_previous + weight_current
        blended = (previous_overlap * weight_previous + current_overlap * weight_current) / weight_sum
    IPG_realtime[-overlap_size:] = blended.tolist()
    IPG_realtime.extend(
        current_data[overlap_size:].tolist()
    )

    return IPG_realtime

def robust_sigma(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def normalize01(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    xmin = np.min(x)
    xmax = np.max(x)
    if xmax - xmin < 1e-12:
        return np.ones_like(x) * 0.5
    return (x - xmin) / (xmax - xmin)

def global_to_local(index, global_start = 600):
    return int(index - global_start)

def get_candidates(signal, global_left, global_right, prominence_sigma_ratio = 0.08, global_start = 600):
    """Score positive peaks by prominence and amplitude above the local baseline."""
    signal = np.asarray(signal, dtype=np.float64)

    local_left = max(1, global_to_local(global_left))
    local_right = min(len(signal) - 2, global_to_local(global_right))

    if local_right - local_left < 3:
        return (
            np.array([], dtype=int),
            np.array([], dtype=np.float64)
        )

    # Extend the slice so boundary peaks remain detectable.
    segment = signal[local_left - 1: local_right + 2]

    sigma = robust_sigma(segment)
    prominence_threshold = max(
        prominence_sigma_ratio * sigma,
        0.005 * np.ptp(segment),
        1e-12
    )

    local_peaks, properties = find_peaks(
        segment,
        prominence=prominence_threshold
    )

    global_peaks = (
        local_peaks
        + local_left - 1
        + global_start
    )

    valid = (
        (global_peaks >= global_left)
        & (global_peaks <= global_right)
    )

    global_peaks = global_peaks[valid]
    prominences = properties["prominences"][valid]

    if len(global_peaks) == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=np.float64)
        )

    local_indices = (global_peaks - global_start).astype(int)
    local_baseline = np.median(segment)

    quality = (
        prominences
        + np.maximum(signal[local_indices] - local_baseline, 0)
    )

    return global_peaks.astype(int), quality.astype(np.float64)

def locate_online_DIPG_BCG_points(dipg_signal, bcg_signal, stable_end_global, state, heart_cycle_samples,
        sampling_freq = 200, global_start = 600, initial_search_delay_samples = 0, bcg_lookahead_ratio = 0.10):

    dipg_signal = np.asarray(dipg_signal, dtype=np.float64)
    bcg_signal = np.asarray(bcg_signal, dtype=np.float64)

    n = min(len(dipg_signal), len(bcg_signal))
    dipg_signal = dipg_signal[:n]
    bcg_signal = bcg_signal[:n]

    T = float(heart_cycle_samples)
    fs = int(sampling_freq)
    global_start = int(global_start)
    stable_end_global = int(stable_end_global)

    if T <= 0:
        raise ValueError("heart_cycle_samples must > 0.")

    if state is None: state = {}

    if not state.get("_initialized", False):
        state.clear()
        state["_initialized"] = True
        state["dipg_initialized"] = False
        state["dipg_all_peaks"] = []
        state["rr_history"] = []
        state["SPK"] = None
        state["NPK"] = None

        state["pending_dipg"] = []
        state["DIPG_peaks"] = []
        state["BCG_J_peaks"] = []
        state["DIPG_OK"] = []
        state["BCG_OK"] = []

        state["BCG_SPK"] = None
        state["BCG_NPK"] = None
        state["lag_history"] = []
        state["last_BCG_J"] = None

        state["initial_search_delay_samples"] = int(initial_search_delay_samples)
    new_detected_dipg = []


    # Detect DIPG peaks with adaptive thresholds and RR tracking.
    if not state["dipg_initialized"]:
        first_left = global_start + state["initial_search_delay_samples"]
        first_right = int(round(first_left + 1.20 * T))

        if stable_end_global >= first_right:
            candidates, quality = get_candidates(dipg_signal, first_left, first_right, prominence_sigma_ratio=0.08)
            if len(candidates) > 0:
                best = int(np.argmax(quality))
                first_peak = int(candidates[best])
                first_quality = float(quality[best])
                noise_quality = np.delete(quality, best)
                state["SPK"] = first_quality
                if len(noise_quality) > 0:
                    state["NPK"] = float(np.median(noise_quality))
                else:
                    state["NPK"] = 0.20 * first_quality
                state["dipg_all_peaks"].append(first_peak)
                state["dipg_initialized"] = True
                new_detected_dipg.append(first_peak)

    while state["dipg_initialized"]:
        last_peak = state["dipg_all_peaks"][-1]
        if len(state["rr_history"]) > 0:
            rr_expected = float(np.median(state["rr_history"][-8:]))
        else:
            rr_expected = T
        search_left = int(round(last_peak + 0.65 * rr_expected))
        search_right = int(round(last_peak + 1.35 * rr_expected))

        if stable_end_global < search_right:
            break
        candidates, quality = get_candidates(dipg_signal, search_left, search_right, prominence_sigma_ratio=0.08)
        threshold = (state["NPK"] + 0.25 * (state["SPK"] - state["NPK"]))

        selected_peak = None
        selected_quality = None
        if len(candidates) > 0:
            good = quality >= threshold
            if not np.any(good):
                good = quality >= 0.50 * threshold
            if np.any(good):
                good_peaks = candidates[good]
                good_quality = quality[good]
                expected_position = last_peak + rr_expected
                if len(good_quality) > 1:
                    quality_score = normalize01(good_quality)
                else:
                    quality_score = np.array([1.0])
                timing_score = np.exp(-0.5 * ((good_peaks - expected_position) / (0.18 * rr_expected)) ** 2)
                score = 0.70 * quality_score + 0.30 * timing_score
                best = int(np.argmax(score))
                selected_peak = int(good_peaks[best])
                selected_quality = float(good_quality[best])
                rejected_quality = quality[~good]

                for q in rejected_quality:
                    state["NPK"] = 0.125 * float(q) + 0.875 * state["NPK"]

        # Search back when the expected RR window has no reliable peak.
        if selected_peak is None:
            searchback_right = int(round(last_peak + 2.35 * rr_expected))
            if stable_end_global < searchback_right:
                break
            sb_candidates, sb_quality = get_candidates(dipg_signal, search_right + 1, searchback_right, prominence_sigma_ratio=0.08)
            if len(sb_candidates) > 0:
                lower_threshold = 0.50 * threshold
                good = sb_quality >= lower_threshold
                if np.any(good):
                    good_peaks = sb_candidates[good]
                    good_quality = sb_quality[good]
                    expected_position = last_peak + 2.0 * rr_expected
                    if len(good_quality) > 1:
                        quality_score = normalize01(good_quality)
                    else:
                        quality_score = np.array([1.0])

                    timing_score = np.exp(-0.5 * ((good_peaks - expected_position) / (0.25 * rr_expected)) ** 2)
                    score = 0.70 * quality_score + 0.30 * timing_score
                    best = int(np.argmax(score))
                    selected_peak = int(good_peaks[best])
                    selected_quality = float(good_quality[best])

            # Fall back to the strongest extended-window candidate.
            if selected_peak is None:
                ext_candidates, ext_quality = get_candidates(
                    dipg_signal,
                    search_left,
                    searchback_right,
                    prominence_sigma_ratio=0.08
                )
                if len(ext_candidates) == 0:
                    break
                best = int(np.argmax(ext_quality))
                selected_peak = int(ext_candidates[best])
                selected_quality = float(ext_quality[best])

        rr = selected_peak - last_peak
        state["rr_history"].append(rr)
        state["dipg_all_peaks"].append(selected_peak)
        state["SPK"] = 0.125 * selected_quality + 0.875 * state["SPK"]
        new_detected_dipg.append(selected_peak)

    state["pending_dipg"].extend(new_detected_dipg)

    # Detect BCG J peaks with DIPG-guided temporal gating.
    new_pairs = []
    still_pending = []

    for dipg_peak in state["pending_dipg"]:
        required_end = int(round(dipg_peak + bcg_lookahead_ratio * T))

        if stable_end_global < required_end:
            still_pending.append(dipg_peak)
            continue

        if len(state["lag_history"]) == 0:
            bcg_search_left = int(round(dipg_peak - 0.55 * T))
            bcg_search_right = int(round(dipg_peak + 0.08 * T))
        else:
            lag_expected = float(np.median(state["lag_history"][-8:]))

            predicted_j = dipg_peak - lag_expected
            bcg_search_left = int(round(predicted_j - 0.18 * T))
            bcg_search_right = int(round(predicted_j + 0.18 * T))

        bcg_candidates, bcg_quality = get_candidates(bcg_signal, bcg_search_left, bcg_search_right,
                                                     prominence_sigma_ratio=0.08)
        selected_bcg = None
        selected_bcg_quality = None
        if len(bcg_candidates) > 0:
            if state["BCG_SPK"] is None:
                best = int(np.argmax(bcg_quality))
                selected_bcg = int(bcg_candidates[best])
                selected_bcg_quality = float(bcg_quality[best])
                noise_quality = np.delete(bcg_quality, best)
                state["BCG_SPK"] = selected_bcg_quality
                if len(noise_quality) > 0:
                    state["BCG_NPK"] = float(np.median(noise_quality))
                else:
                    state["BCG_NPK"] = (0.20 * selected_bcg_quality)
            else:
                bcg_threshold = state["BCG_NPK"] + 0.25 * (state["BCG_SPK"] - state["BCG_NPK"])
                good = bcg_quality >= 0.50 * bcg_threshold
                if not np.any(good):
                    good = np.ones(len(bcg_quality),dtype=bool)
                good_peaks = bcg_candidates[good]
                good_quality = bcg_quality[good]
                if len(state["lag_history"]) > 0:
                    lag_expected = float(np.median(state["lag_history"][-8:]))
                else:
                    lag_expected = 0.15 * T
                predicted_j = dipg_peak - lag_expected
                if len(good_quality) > 1:
                    quality_score = normalize01(good_quality)
                else:
                    quality_score = np.array([1.0])

                timing_score = np.exp(-0.5 * ((good_peaks - predicted_j) / (0.12 * T)) ** 2)

                if state["last_BCG_J"] is None:
                    rr_score = np.ones(len(good_peaks)) * 0.5
                else:
                    jj = good_peaks - state["last_BCG_J"]
                    rr_error = np.abs(jj - T) / T
                    rr_score = np.exp(-0.5 * (rr_error / 0.20) ** 2)
                score = (
                        0.60 * quality_score
                        + 0.25 * timing_score
                        + 0.15 * rr_score
                )
                best = int(np.argmax(score))
                selected_bcg = int(good_peaks[best])
                selected_bcg_quality = float(good_quality[best])
                rejected_quality = bcg_quality[~good]

                for q in rejected_quality:
                    state["BCG_NPK"] = (
                            0.125 * float(q)
                            + 0.875 * state["BCG_NPK"]
                    )
        # Reject implausible BCG-DIPG and J-J intervals.
        if selected_bcg is not None:
            lag = dipg_peak - selected_bcg
            if lag < -0.08 * T or lag > 0.55 * T:
                selected_bcg = None
        if selected_bcg is not None and state["last_BCG_J"] is not None:
            jj = selected_bcg - state["last_BCG_J"]
            if jj < 0.55 * T or jj > 1.45 * T:
                rr_valid = ((bcg_candidates - state["last_BCG_J"] >= 0.55 * T)
                        & (bcg_candidates - state["last_BCG_J"] <= 1.45 * T))
                backup = bcg_candidates[rr_valid]
                if len(backup) == 0:
                    selected_bcg = None
                else:
                    if len(state["lag_history"]) > 0:
                        lag_expected = float(np.median(state["lag_history"][-8:]))
                    else:
                        lag_expected = 0.15 * T
                    predicted_j = dipg_peak - lag_expected
                    selected_bcg = int(
                        backup[
                            np.argmin(
                                np.abs(backup - predicted_j)
                            )
                        ]
                    )
        state["DIPG_peaks"].append(int(dipg_peak))
        state["DIPG_OK"].append(1)
        if selected_bcg is None:
            state["BCG_J_peaks"].append(np.nan)
            state["BCG_OK"].append(0)
            new_pairs.append((int(dipg_peak), None))
        else:
            if selected_bcg_quality is not None and state["BCG_SPK"] is not None:
                state["BCG_SPK"] = 0.125 * selected_bcg_quality + 0.875 * state["BCG_SPK"]
            lag = int(dipg_peak - selected_bcg)
            state["last_BCG_J"] = int(selected_bcg)
            state["lag_history"].append(lag)
            state["BCG_J_peaks"].append(int(selected_bcg))
            state["BCG_OK"].append(1)
            new_pairs.append((int(dipg_peak), int(selected_bcg)))
    state["pending_dipg"] = still_pending

    return state, new_pairs

def _print_online_detection_result(name, state):
    if state is None:
        print(f"{name}: no confirmable cardiac cycle has been detected yet")
        return

    dipg_ok = np.asarray(state["DIPG_OK"], dtype=np.int8)
    bcg_ok = np.asarray(state["BCG_OK"], dtype=np.int8)

    paired = int(np.sum((dipg_ok == 1) & (bcg_ok == 1)))

    print(
        f"{name}: "
        f"DIPG={int(np.sum(dipg_ok))}, "
        f"BCG J={int(np.sum(bcg_ok))}, "
        f"paired detections={paired}"
    )

def crop_and_shift_peak_pairs(dipg_peaks, bcg_peaks, dipg_ok, bcg_ok, crop_global_start, crop_length):
        """Crop paired fiducials and convert them to local indices."""
        dipg_peaks = np.asarray(dipg_peaks, dtype=np.float64)
        bcg_peaks = np.asarray(bcg_peaks, dtype=np.float64)
        dipg_ok = np.asarray(dipg_ok, dtype=np.int8)
        bcg_ok = np.asarray(bcg_ok, dtype=np.int8)

        event_len = min(len(dipg_peaks), len(bcg_peaks), len(dipg_ok), len(bcg_ok))
        if len({len(dipg_peaks), len(bcg_peaks), len(dipg_ok), len(bcg_ok)}) != 1:
            print("Warning: DIPG/BCG/validity arrays within the same method have unequal lengths and were truncated to the shortest length.")

        dipg_peaks = dipg_peaks[:event_len]
        bcg_peaks = bcg_peaks[:event_len]
        dipg_ok = dipg_ok[:event_len]
        bcg_ok = bcg_ok[:event_len]

        crop_end = crop_global_start + crop_length
        keep = ((dipg_ok == 1) & np.isfinite(dipg_peaks) & (dipg_peaks >= crop_global_start) & (dipg_peaks < crop_end))

        dipg_peaks = dipg_peaks[keep]
        bcg_peaks = bcg_peaks[keep]
        dipg_ok = dipg_ok[keep]
        bcg_ok = bcg_ok[keep]

        dipg_peaks = (dipg_peaks - crop_global_start).astype(np.int64)

        bcg_valid = ((bcg_ok == 1) & np.isfinite(bcg_peaks) & (bcg_peaks >= crop_global_start) & (bcg_peaks < crop_end))
        bcg_peaks_local = np.full(len(bcg_peaks), np.nan, dtype=np.float64)
        bcg_peaks_local[bcg_valid] = bcg_peaks[bcg_valid] - crop_global_start
        bcg_ok = np.where(bcg_valid, 1, 0).astype(np.int8)

        return dipg_peaks, bcg_peaks_local, dipg_ok.astype(np.int8), bcg_ok

def align_peak_pairs_by_reference(reference_peaks, dipg_peaks, bcg_peaks, dipg_ok, bcg_ok,
                                      singledistance, expected_delay=0, match_ratio=0.45):
        """Align method-specific peak pairs to filtfilt DIPG reference beats."""
        reference_peaks = np.asarray(reference_peaks, dtype=np.float64)
        dipg_peaks = np.asarray(dipg_peaks, dtype=np.float64)
        bcg_peaks = np.asarray(bcg_peaks, dtype=np.float64)
        dipg_ok = np.asarray(dipg_ok, dtype=np.int8)
        bcg_ok = np.asarray(bcg_ok, dtype=np.int8)

        aligned_DIPG_peaks = np.full(len(reference_peaks), np.nan, dtype=np.float64)
        aligned_BCG_peaks = np.full(len(reference_peaks), np.nan, dtype=np.float64)
        aligned_DIPG_OK = np.zeros(len(reference_peaks), dtype=np.int8)
        aligned_BCG_OK = np.zeros(len(reference_peaks), dtype=np.int8)

        used = set()
        match_distance = singledistance * match_ratio

        for j in range(0,len(reference_peaks)):
            expected_peak = reference_peaks[j] + expected_delay
            candidate_index = []

            for k in range(0,len(dipg_peaks)):
                if k in used:
                    continue
                if dipg_ok[k] != 1 or not np.isfinite(dipg_peaks[k]):
                    continue
                distance = abs(dipg_peaks[k] - expected_peak)
                if distance <= match_distance:
                    candidate_index.append((distance, k))

            if len(candidate_index) == 0:
                continue

            _, best_index = min(candidate_index, key=lambda x:x[0])
            used.add(best_index)

            aligned_DIPG_peaks[j] = dipg_peaks[best_index]
            aligned_DIPG_OK[j] = 1

            if bcg_ok[best_index] == 1 and np.isfinite(bcg_peaks[best_index]):
                aligned_BCG_peaks[j] = bcg_peaks[best_index]
                aligned_BCG_OK[j] = 1

        return aligned_DIPG_peaks, aligned_BCG_peaks, aligned_DIPG_OK, aligned_BCG_OK


def safe_pearsonr(signal1, signal2):
    signal1 = np.asarray(signal1, dtype=np.float64)
    signal2 = np.asarray(signal2, dtype=np.float64)

    n = min(len(signal1), len(signal2))
    signal1 = signal1[:n]
    signal2 = signal2[:n]

    valid = np.isfinite(signal1) & np.isfinite(signal2)
    signal1 = signal1[valid]
    signal2 = signal2[valid]

    if len(signal1) < 3:
        return np.nan, np.nan

    if np.ptp(signal1) < 1e-12 or np.ptp(signal2) < 1e-12:
        return np.nan, np.nan

    return pearsonr(signal1, signal2)


def calculate_nrmse(reference_signal, test_signal):
    reference_signal = np.asarray(reference_signal, dtype=np.float64)
    test_signal = np.asarray(test_signal, dtype=np.float64)

    n = min(len(reference_signal), len(test_signal))
    reference_signal = reference_signal[:n]
    test_signal = test_signal[:n]

    valid = np.isfinite(reference_signal) & np.isfinite(test_signal)
    reference_signal = reference_signal[valid]
    test_signal = test_signal[valid]

    if len(reference_signal) == 0:
        return np.nan

    signal_range = np.max(reference_signal) - np.min(reference_signal)
    if signal_range < 1e-12:
        return np.nan

    rmse = np.sqrt(np.mean((reference_signal - test_signal) ** 2))
    return rmse / signal_range


def calculate_bias_mae(test_peaks, reference_peaks, valid_mask):
    test_peaks = np.asarray(test_peaks, dtype=np.float64)
    reference_peaks = np.asarray(reference_peaks, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    valid_mask = (
        valid_mask
        & np.isfinite(test_peaks)
        & np.isfinite(reference_peaks)
    )

    if np.sum(valid_mask) == 0:
        return np.nan, np.nan

    diff = test_peaks[valid_mask] - reference_peaks[valid_mask]
    return np.mean(diff), np.mean(np.abs(diff))


def calculate_bland_altman(test, reference):
    test = np.asarray(test, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    valid = np.isfinite(test) & np.isfinite(reference)
    diff = test[valid] - reference[valid]

    if len(diff) == 0:
        return np.nan, np.nan, np.nan, np.nan

    bias = np.mean(diff)

    if len(diff) >= 2:
        sd = np.std(diff, ddof=1)
    else:
        sd = 0.0

    lower = bias - 1.96 * sd
    upper = bias + 1.96 * sd
    width = upper - lower

    return bias, lower, upper, width


def _set_line_and_autoscale(axis, line, x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    line.set_data(x, y)

    if len(y) > 1:
        valid = np.isfinite(y)
        if np.any(valid):
            ymin = np.min(y[valid])
            ymax = np.max(y[valid])
            yrange = ymax - ymin
            if yrange < 1e-12:
                yrange = max(abs(ymax), 1.0) * 0.1
            margin = 0.10 * yrange
            axis.set_ylim(ymin - margin, ymax + margin)


def _set_peak_artist(artist, signal, peaks_global, ok, global_start, sampling_freq):
    signal = np.asarray(signal, dtype=np.float64)

    if peaks_global is None or ok is None or len(signal) == 0:
        artist.set_data([], [])
        return

    peaks_global = np.asarray(peaks_global, dtype=np.float64)
    ok = np.asarray(ok, dtype=np.int8)

    n = min(len(peaks_global), len(ok))
    peaks_global = peaks_global[:n]
    ok = ok[:n]

    valid = (ok == 1) & np.isfinite(peaks_global)
    peaks_global = peaks_global[valid]

    if len(peaks_global) == 0:
        artist.set_data([], [])
        return

    local = (peaks_global - global_start).astype(np.int64)
    valid_local = (local >= 0) & (local < len(signal))

    peaks_global = peaks_global[valid_local]
    local = local[valid_local]

    artist.set_data(
        peaks_global / sampling_freq,
        signal[local]
    )



def create_realtime_figure(total_seconds):
    fig, axes = plt.subplots(6, 2, figsize=(16, 12), sharex=True)
    fig.canvas.manager.set_window_title("Real-Time BCG-IPG Signal Processing")

    row_names = ["Raw", "Butterworth IIR", "Bessel IIR", "FIR", "LW-QZPF", "filtfilt"]

    for row in range(6):
        axes[row, 0].set_ylabel(row_names[row])
        axes[row, 0].grid(alpha=0.15)
        axes[row, 1].grid(alpha=0.15)
        axes[row, 0].set_xlim(0, total_seconds)
        axes[row, 1].set_xlim(0, total_seconds)

    axes[0, 0].set_title("IPG / DIPG")
    axes[0, 1].set_title("BCG")
    axes[5, 0].set_xlabel("Time (s)")
    axes[5, 1].set_xlabel("Time (s)")

    line_IPG_raw, = axes[0, 0].plot([], [], color=color_warm[0], linewidth=1.0, label="IPG Raw")
    line_BCG_raw, = axes[0, 1].plot([], [], color=color_cool[0], linewidth=1.0, label="BCG Raw")

    line_DIPG_IIR, = axes[1, 0].plot([], [], color=color_warm[1], linewidth=1.0, label="DIPG Butterworth IIR")
    line_BCG_IIR, = axes[1, 1].plot([], [], color=color_cool[1], linewidth=1.0, label="BCG Butterworth IIR")
    peak_DIPG_IIR, = axes[1, 0].plot([], [], '^', color='red', linestyle='None', markersize=5)
    peak_BCG_IIR, = axes[1, 1].plot([], [], '^', color='red', linestyle='None', markersize=5)

    line_DIPG_Bessel, = axes[2, 0].plot([], [], color=color_warm[2], linewidth=1.0, label="DIPG Bessel IIR")
    line_BCG_Bessel, = axes[2, 1].plot([], [], color=color_cool[2], linewidth=1.0, label="BCG Bessel IIR")
    peak_DIPG_Bessel, = axes[2, 0].plot([], [], '^', color='red', linestyle='None', markersize=5)
    peak_BCG_Bessel, = axes[2, 1].plot([], [], '^', color='red', linestyle='None', markersize=5)

    line_DIPG_FIR, = axes[3, 0].plot([], [], color=color_warm[3], linewidth=1.0, label="DIPG FIR")
    line_BCG_FIR, = axes[3, 1].plot([], [], color=color_cool[3], linewidth=1.0, label="BCG FIR")
    peak_DIPG_FIR, = axes[3, 0].plot([], [], '^', color='red', linestyle='None', markersize=5)
    peak_BCG_FIR, = axes[3, 1].plot([], [], '^', color='red', linestyle='None', markersize=5)

    line_DIPG_LWQZPF, = axes[4, 0].plot([], [], color=color_warm[5], linewidth=1.0, label="DIPG LW-QZPF")
    line_BCG_LWQZPF, = axes[4, 1].plot([], [], color=color_cool[5], linewidth=1.0, label="BCG LW-QZPF")
    peak_DIPG_LWQZPF, = axes[4, 0].plot([], [], '^', color='red', linestyle='None', markersize=5)
    peak_BCG_LWQZPF, = axes[4, 1].plot([], [], '^', color='red', linestyle='None', markersize=5)

    # filtfilt is available only after complete acquisition.
    line_DIPG_filtfilt, = axes[5, 0].plot([], [], color=color_warm[4], linewidth=1.0, label="DIPG filtfilt")
    line_BCG_filtfilt, = axes[5, 1].plot([], [], color=color_cool[4], linewidth=1.0, label="BCG filtfilt")
    peak_DIPG_filtfilt, = axes[5, 0].plot([], [], '^', color='red', linestyle='None', markersize=5)
    peak_BCG_filtfilt, = axes[5, 1].plot([], [], '^', color='red', linestyle='None', markersize=5)

    for row in range(6):
        axes[row, 0].legend(loc='upper right', fontsize=8)
        axes[row, 1].legend(loc='upper right', fontsize=8)

    text_filtfilt_left = axes[5, 0].text(
        0.5, 0.5, "Waiting for acquisition of the complete raw record...",
        transform=axes[5, 0].transAxes, ha='center', va='center', fontsize=11
    )
    text_filtfilt_right = axes[5, 1].text(
        0.5, 0.5, "Waiting for acquisition of the complete raw record...",
        transform=axes[5, 1].transAxes, ha='center', va='center', fontsize=11
    )

    status_text = fig.text(
        0.5, 0.985, "Real-Time BCG-IPG Signal Processing",
        ha='center', va='top', fontsize=11
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    artists = {
        "IPG_raw": line_IPG_raw, "BCG_raw": line_BCG_raw,
        "DIPG_IIR": line_DIPG_IIR, "BCG_IIR": line_BCG_IIR,
        "peak_DIPG_IIR": peak_DIPG_IIR, "peak_BCG_IIR": peak_BCG_IIR,
        "DIPG_Bessel": line_DIPG_Bessel, "BCG_Bessel": line_BCG_Bessel,
        "peak_DIPG_Bessel": peak_DIPG_Bessel, "peak_BCG_Bessel": peak_BCG_Bessel,
        "DIPG_FIR": line_DIPG_FIR, "BCG_FIR": line_BCG_FIR,
        "peak_DIPG_FIR": peak_DIPG_FIR, "peak_BCG_FIR": peak_BCG_FIR,
        "DIPG_LWQZPF": line_DIPG_LWQZPF, "BCG_LWQZPF": line_BCG_LWQZPF,
        "peak_DIPG_LWQZPF": peak_DIPG_LWQZPF, "peak_BCG_LWQZPF": peak_BCG_LWQZPF,
        "DIPG_filtfilt": line_DIPG_filtfilt, "BCG_filtfilt": line_BCG_filtfilt,
        "peak_DIPG_filtfilt": peak_DIPG_filtfilt, "peak_BCG_filtfilt": peak_BCG_filtfilt,
        "text_filtfilt_left": text_filtfilt_left, "text_filtfilt_right": text_filtfilt_right,
        "status_text": status_text
    }

    return fig, axes, artists


def format_table_value(value, digits=3):
    if isinstance(value, str):
        return value
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def show_result_table(waveform_table, temporal_table, resource_table):
    if not show_performance_table:
        return

    fig_table, axes_table = plt.subplots(3, 1, figsize=(19, 11))
    maximize_matplotlib_window(fig_table)
    fig_table.canvas.manager.set_window_title("Performance Evaluation of Filtering Methods")

    tables = [
        (
            axes_table[0],
            waveform_table,
            "A. Waveform Agreement with Offline Zero-Phase filtfilt",
            10.5
        ),
        (
            axes_table[1],
            temporal_table,
            "B. Temporal-Feature Preservation Relative to Offline filtfilt",
            10.5
        ),
        (
            axes_table[2],
            resource_table,
            "C. Continuous-Processing and Computational-Resource Requirements",
            10.5
        )
    ]

    for axis, df_table, title, fontsize in tables:
        axis.axis('off')
        table = axis.table(
            cellText=df_table.values,
            colLabels=df_table.columns,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.scale(1.0, 1.55)

        for (row, col), cell in table.get_celld().items():
            cell.set_text_props(ha='center', va='center')
            if row == 0:
                cell.set_text_props(weight='bold', ha='center', va='center')

        axis.set_title(title, fontsize=15, pad=14)

    plt.tight_layout()
    fig_table.canvas.draw_idle()
    fig_table.canvas.flush_events()
    maximize_matplotlib_window(fig_table)


def run_realtime_processing():
    # Load one record.
    if not os.path.exists(excel_path):
        raise FileNotFoundError(
            "Excel file not found. Please update excel_path at the top of the script:\n" + excel_path
        )

    df = pd.read_excel(excel_path)

    if "IPG" not in df.columns or "BCG" not in df.columns:
        raise ValueError("The Excel file must contain at least the 'IPG' and 'BCG' columns.")

    IPG_raw = np.asarray(df["IPG"], dtype=np.float64)
    BCG_raw = np.asarray(df["BCG"], dtype=np.float64)

    IPG_raw = -(IPG_raw - IPG_raw[0])
    BCG_raw = -(BCG_raw - BCG_raw[0])

    signal_length = min(len(IPG_raw), len(BCG_raw))
    IPG_raw = IPG_raw[:signal_length]
    BCG_raw = BCG_raw[:signal_length]

    if processing_seconds is not None:
        signal_length = min(signal_length, int(processing_seconds * sampling_freq))
        IPG_raw = IPG_raw[:signal_length]
        BCG_raw = BCG_raw[:signal_length]

    if signal_length <= initial_length + 3 * sampling_freq:
        raise ValueError("The input record is too short; at least 6 s of data is recommended.")

    total_seconds = signal_length / sampling_freq

    # Create the real-time display.
    plt.ion()
    fig, axes, artists = create_realtime_figure(total_seconds)
    maximize_matplotlib_window(fig)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.05)
    wait_with_gui_updates(startup_delay)

    acquired_IPG = []
    acquired_BCG = []

    online_ready = False
    singledistance = None
    overlap_size = None

    DIPG_butterworth_IIR = []
    BCG_butterworth_IIR = []
    state_IIR = None
    IIR_Tproc_list = []

    DIPG_Bessel = []
    BCG_Bessel = []
    state_Bessel = None
    Bessel_Tproc_list = []

    DIPG_FIR = []
    BCG_FIR = []
    state_FIR = None
    FIR_Tproc_list = []

    count_LWQZPF = 0
    DIPG_LWQZPF = []
    BCG_LWQZPF = []
    state_LWQZPF = None
    LWQZPF_Tproc_list = []

    # Acquire and process at 200 Hz.
    print("\n================ Real-Time Acquisition ================")
    print(f"Sampling frequency = {sampling_freq} Hz")
    print(f"Sample interval = {point_delay:.6f} s")
    print(f"Total samples = {signal_length}, duration = {total_seconds:.2f} s")

    acquire_index = 0
    start_wall_time = time.perf_counter()

    while acquire_index < signal_length and plt.fignum_exists(fig.number):
        if point_delay > 0:
            elapsed = time.perf_counter() - start_wall_time
            target_index = min(signal_length, int(elapsed / point_delay) + 1)
        else:
            target_index = signal_length

        while acquire_index < target_index:
            sample_IPG = IPG_raw[acquire_index]
            sample_BCG = BCG_raw[acquire_index]

            acquired_IPG.append(sample_IPG)
            acquired_BCG.append(sample_BCG)

            # Initialize online methods after the first 3 s.
            if (not online_ready) and (acquire_index + 1 >= initial_length):
                initial_IPG = np.asarray(acquired_IPG[:initial_length], dtype=np.float64)
                initial_BCG = np.asarray(acquired_BCG[:initial_length], dtype=np.float64)

                # LW-QZPF initialization
                singledistance = get_SingleCardiacCycleDistance(initial_IPG, sampling_freq=sampling_freq)
                CalculatedAverageCardiacCycle = round(singledistance / sampling_freq, 4)
                overlap_size = math.ceil(singledistance / 2)

                IPG_LWQZPF_buffer = np.empty(singledistance, dtype=np.float64)
                BCG_LWQZPF_buffer = np.empty(singledistance, dtype=np.float64)
                DIPG_pre = initial_IPG[initial_length - overlap_size:initial_length].copy()
                BCG_pre = initial_BCG[initial_length - overlap_size:initial_length].copy()

                nyquist = 0.5 * sampling_freq

                # Butterworth IIR initialization
                IPG_IIR_buffer = np.empty(IIR_buffer_size, dtype=np.float64)
                BCG_IIR_buffer = np.empty(IIR_buffer_size, dtype=np.float64)

                b_IPG_IIR, a_IPG_IIR = butter(3, [0.5 / nyquist, 5 / nyquist], btype='band')
                b_BCG_IIR, a_BCG_IIR = butter(3, [0.9 / nyquist, 9 / nyquist], btype='band')

                zi_IPG_IIR = np.zeros(max(len(a_IPG_IIR), len(b_IPG_IIR)) - 1, dtype=np.float64)
                zi_BCG_IIR = np.zeros(max(len(a_BCG_IIR), len(b_BCG_IIR)) - 1, dtype=np.float64)

                IPG_IIR_initial_filtered, zi_IPG_IIR = lfilter(
                    b_IPG_IIR, a_IPG_IIR, initial_IPG, zi=zi_IPG_IIR
                )
                BCG_IIR_initial_filtered, zi_BCG_IIR = lfilter(
                    b_BCG_IIR, a_BCG_IIR, initial_BCG, zi=zi_BCG_IIR
                )

                IIR_stream_state = init_streaming_derivative_state(
                    IPG_IIR_initial_filtered[-1], BCG_IIR_initial_filtered[-1]
                )

                # Bessel IIR initialization
                IPG_Bessel_buffer = np.empty(Bessel_buffer_size, dtype=np.float64)
                BCG_Bessel_buffer = np.empty(Bessel_buffer_size, dtype=np.float64)

                b_IPG_Bessel, a_IPG_Bessel = bessel(
                    3, [0.5, 5], btype='band', norm='mag', fs=sampling_freq
                )
                b_BCG_Bessel, a_BCG_Bessel = bessel(
                    3, [0.9, 9], btype='band', norm='mag', fs=sampling_freq
                )

                zi_IPG_Bessel = np.zeros(
                    max(len(a_IPG_Bessel), len(b_IPG_Bessel)) - 1, dtype=np.float64
                )
                zi_BCG_Bessel = np.zeros(
                    max(len(a_BCG_Bessel), len(b_BCG_Bessel)) - 1, dtype=np.float64
                )

                IPG_Bessel_initial_filtered, zi_IPG_Bessel = lfilter(
                    b_IPG_Bessel, a_IPG_Bessel, initial_IPG, zi=zi_IPG_Bessel
                )
                BCG_Bessel_initial_filtered, zi_BCG_Bessel = lfilter(
                    b_BCG_Bessel, a_BCG_Bessel, initial_BCG, zi=zi_BCG_Bessel
                )

                Bessel_stream_state = init_streaming_derivative_state(
                    IPG_Bessel_initial_filtered[-1], BCG_Bessel_initial_filtered[-1]
                )

                # FIR initialization
                fir_group_delay_samples = (fir_numtaps - 1) // 2
                fir_group_delay_ms = fir_group_delay_samples / sampling_freq * 1000
                IPG_FIR_buffer = np.empty(FIR_buffer_size, dtype=np.float64)
                BCG_FIR_buffer = np.empty(FIR_buffer_size, dtype=np.float64)

                b_IPG_FIR = firwin(
                    numtaps=fir_numtaps, cutoff=[0.5, 5], pass_zero=False,
                    fs=sampling_freq, window='hamming'
                )
                b_BCG_FIR = firwin(
                    numtaps=fir_numtaps, cutoff=[0.9, 9], pass_zero=False,
                    fs=sampling_freq, window='hamming'
                )
                a_IPG_FIR = np.array([1.0], dtype=np.float64)
                a_BCG_FIR = np.array([1.0], dtype=np.float64)

                zi_IPG_FIR = np.zeros(len(b_IPG_FIR) - 1, dtype=np.float64)
                zi_BCG_FIR = np.zeros(len(b_BCG_FIR) - 1, dtype=np.float64)

                _, zi_IPG_FIR = lfilter(
                    b_IPG_FIR, a_IPG_FIR, initial_IPG, zi=zi_IPG_FIR
                )
                _, zi_BCG_FIR = lfilter(
                    b_BCG_FIR, a_BCG_FIR, initial_BCG, zi=zi_BCG_FIR
                )

                FIR_stream_state = init_streaming_derivative_state()

                print(
                    f"Initial 3-s acquisition completed: cardiac-cycle estimate = "
                    f"{singledistance} samples ({CalculatedAverageCardiacCycle:.3f} s)"
                )
                print(
                    f"FIR numtaps = {fir_numtaps}, theoretical group delay = "
                    f"{fir_group_delay_samples} samples ({fir_group_delay_ms:.1f} ms)"
                )

                online_ready = True

            elif online_ready and acquire_index >= initial_length:
                online_index = acquire_index - initial_length

                # LW-QZPF
                IPG_LWQZPF_buffer[count_LWQZPF] = sample_IPG
                BCG_LWQZPF_buffer[count_LWQZPF] = sample_BCG
                count_LWQZPF += 1

                if count_LWQZPF == singledistance:
                    count_LWQZPF = 0
                    LWQZPF_Tproc_start = time.perf_counter()

                    currentDIPG_LWQZPF = np.concatenate((DIPG_pre, IPG_LWQZPF_buffer))
                    currentDIPG_LWQZPF = Realtime_Processed(currentDIPG_LWQZPF, 0.5, 5)
                    currentDIPG_LWQZPF = differentiate(currentDIPG_LWQZPF, sampling_freq=sampling_freq)
                    DIPG_LWQZPF = Hann_Stitch(DIPG_LWQZPF, currentDIPG_LWQZPF, overlap_size)
                    DIPG_pre = IPG_LWQZPF_buffer[-overlap_size:].copy()

                    currentBCG_LWQZPF = np.concatenate((BCG_pre, BCG_LWQZPF_buffer))
                    currentBCG_LWQZPF = Realtime_Processed(currentBCG_LWQZPF, 0.9, 9)
                    BCG_LWQZPF = Hann_Stitch(BCG_LWQZPF, currentBCG_LWQZPF, overlap_size)
                    BCG_pre = BCG_LWQZPF_buffer[-overlap_size:].copy()

                    stable_end_LWQZPF = initial_length + len(DIPG_LWQZPF) - overlap_size - 1
                    state_LWQZPF, _ = locate_online_DIPG_BCG_points(
                        DIPG_LWQZPF, BCG_LWQZPF, stable_end_LWQZPF, state_LWQZPF,
                        heart_cycle_samples=singledistance, sampling_freq=sampling_freq,
                        global_start=initial_length, initial_search_delay_samples=0
                    )

                    LWQZPF_Tproc_list.append(time.perf_counter() - LWQZPF_Tproc_start)

                # Butterworth IIR
                IIR_Tproc_start = time.perf_counter()
                IPG_IIR_buffer[0] = sample_IPG
                BCG_IIR_buffer[0] = sample_BCG

                currentIPG_IIR, zi_IPG_IIR = lfilter(
                    b_IPG_IIR, a_IPG_IIR, IPG_IIR_buffer, zi=zi_IPG_IIR
                )
                currentBCG_IIR, zi_BCG_IIR = lfilter(
                    b_BCG_IIR, a_BCG_IIR, BCG_IIR_buffer, zi=zi_BCG_IIR
                )
                currentDIPG_IIR, currentBCG_IIR_center = streaming_center_gradient(
                    currentIPG_IIR[0], currentBCG_IIR[0], IIR_stream_state, sampling_freq
                )

                if currentDIPG_IIR is not None:
                    DIPG_butterworth_IIR.append(currentDIPG_IIR)
                    BCG_butterworth_IIR.append(currentBCG_IIR_center)

                    if len(DIPG_butterworth_IIR) % online_detection_hop == 0:
                        stable_end_IIR = initial_length + len(DIPG_butterworth_IIR) - 1
                        state_IIR, _ = locate_online_DIPG_BCG_points(
                            DIPG_butterworth_IIR, BCG_butterworth_IIR, stable_end_IIR, state_IIR,
                            heart_cycle_samples=singledistance, sampling_freq=sampling_freq,
                            global_start=initial_length, initial_search_delay_samples=0
                        )

                IIR_Tproc_list.append(time.perf_counter() - IIR_Tproc_start)

                # Bessel IIR
                Bessel_Tproc_start = time.perf_counter()
                IPG_Bessel_buffer[0] = sample_IPG
                BCG_Bessel_buffer[0] = sample_BCG

                currentIPG_Bessel, zi_IPG_Bessel = lfilter(
                    b_IPG_Bessel, a_IPG_Bessel, IPG_Bessel_buffer, zi=zi_IPG_Bessel
                )
                currentBCG_Bessel, zi_BCG_Bessel = lfilter(
                    b_BCG_Bessel, a_BCG_Bessel, BCG_Bessel_buffer, zi=zi_BCG_Bessel
                )
                currentDIPG_Bessel, currentBCG_Bessel_center = streaming_center_gradient(
                    currentIPG_Bessel[0], currentBCG_Bessel[0],
                    Bessel_stream_state, sampling_freq
                )

                if currentDIPG_Bessel is not None:
                    DIPG_Bessel.append(currentDIPG_Bessel)
                    BCG_Bessel.append(currentBCG_Bessel_center)

                    if len(DIPG_Bessel) % online_detection_hop == 0:
                        stable_end_Bessel = initial_length + len(DIPG_Bessel) - 1
                        state_Bessel, _ = locate_online_DIPG_BCG_points(
                            DIPG_Bessel, BCG_Bessel, stable_end_Bessel, state_Bessel,
                            heart_cycle_samples=singledistance, sampling_freq=sampling_freq,
                            global_start=initial_length, initial_search_delay_samples=0
                        )

                Bessel_Tproc_list.append(time.perf_counter() - Bessel_Tproc_start)

                # FIR
                FIR_Tproc_start = time.perf_counter()
                IPG_FIR_buffer[0] = sample_IPG
                BCG_FIR_buffer[0] = sample_BCG

                currentIPG_FIR, zi_IPG_FIR = lfilter(
                    b_IPG_FIR, a_IPG_FIR, IPG_FIR_buffer, zi=zi_IPG_FIR
                )
                currentBCG_FIR, zi_BCG_FIR = lfilter(
                    b_BCG_FIR, a_BCG_FIR, BCG_FIR_buffer, zi=zi_BCG_FIR
                )

                if online_index == fir_group_delay_samples - 1:
                    FIR_stream_state = init_streaming_derivative_state(
                        currentIPG_FIR[0], currentBCG_FIR[0]
                    )
                elif online_index >= fir_group_delay_samples:
                    currentDIPG_FIR, currentBCG_FIR_center = streaming_center_gradient(
                        currentIPG_FIR[0], currentBCG_FIR[0],
                        FIR_stream_state, sampling_freq
                    )

                    if currentDIPG_FIR is not None:
                        DIPG_FIR.append(currentDIPG_FIR)
                        BCG_FIR.append(currentBCG_FIR_center)

                        if len(DIPG_FIR) % online_detection_hop == 0:
                            stable_end_FIR = initial_length + len(DIPG_FIR) - 1
                            state_FIR, _ = locate_online_DIPG_BCG_points(
                                DIPG_FIR, BCG_FIR, stable_end_FIR, state_FIR,
                                heart_cycle_samples=singledistance, sampling_freq=sampling_freq,
                                global_start=initial_length, initial_search_delay_samples=0
                            )

                FIR_Tproc_list.append(time.perf_counter() - FIR_Tproc_start)

            acquire_index += 1

        # Refresh the display.
        current_time = acquire_index / sampling_freq
        time_raw = np.arange(acquire_index) / sampling_freq

        _set_line_and_autoscale(axes[0, 0], artists["IPG_raw"], time_raw, acquired_IPG)
        _set_line_and_autoscale(axes[0, 1], artists["BCG_raw"], time_raw, acquired_BCG)

        if online_ready:
            time_IIR = (np.arange(len(DIPG_butterworth_IIR)) + initial_length) / sampling_freq
            _set_line_and_autoscale(axes[1, 0], artists["DIPG_IIR"], time_IIR, DIPG_butterworth_IIR)
            _set_line_and_autoscale(axes[1, 1], artists["BCG_IIR"], time_IIR, BCG_butterworth_IIR)

            if state_IIR is not None:
                _set_peak_artist(
                    artists["peak_DIPG_IIR"], DIPG_butterworth_IIR,
                    state_IIR["DIPG_peaks"], state_IIR["DIPG_OK"], initial_length, sampling_freq
                )
                _set_peak_artist(
                    artists["peak_BCG_IIR"], BCG_butterworth_IIR,
                    state_IIR["BCG_J_peaks"], state_IIR["BCG_OK"], initial_length, sampling_freq
                )

            time_Bessel = (np.arange(len(DIPG_Bessel)) + initial_length) / sampling_freq
            _set_line_and_autoscale(
                axes[2, 0], artists["DIPG_Bessel"], time_Bessel, DIPG_Bessel
            )
            _set_line_and_autoscale(
                axes[2, 1], artists["BCG_Bessel"], time_Bessel, BCG_Bessel
            )

            if state_Bessel is not None:
                _set_peak_artist(
                    artists["peak_DIPG_Bessel"], DIPG_Bessel,
                    state_Bessel["DIPG_peaks"], state_Bessel["DIPG_OK"],
                    initial_length, sampling_freq
                )
                _set_peak_artist(
                    artists["peak_BCG_Bessel"], BCG_Bessel,
                    state_Bessel["BCG_J_peaks"], state_Bessel["BCG_OK"],
                    initial_length, sampling_freq
                )

            # Display FIR on its group-delay-compensated time axis.
            time_FIR = (np.arange(len(DIPG_FIR)) + initial_length) / sampling_freq
            _set_line_and_autoscale(
                axes[3, 0], artists["DIPG_FIR"], time_FIR, DIPG_FIR
            )
            _set_line_and_autoscale(
                axes[3, 1], artists["BCG_FIR"], time_FIR, BCG_FIR
            )

            if state_FIR is not None:
                _set_peak_artist(
                    artists["peak_DIPG_FIR"], DIPG_FIR,
                    state_FIR["DIPG_peaks"], state_FIR["DIPG_OK"],
                    initial_length, sampling_freq
                )
                _set_peak_artist(
                    artists["peak_BCG_FIR"], BCG_FIR,
                    state_FIR["BCG_J_peaks"], state_FIR["BCG_OK"],
                    initial_length, sampling_freq
                )

            time_LWQZPF = (np.arange(len(DIPG_LWQZPF)) + initial_length) / sampling_freq
            _set_line_and_autoscale(
                axes[4, 0], artists["DIPG_LWQZPF"], time_LWQZPF, DIPG_LWQZPF
            )
            _set_line_and_autoscale(
                axes[4, 1], artists["BCG_LWQZPF"], time_LWQZPF, BCG_LWQZPF
            )

            if state_LWQZPF is not None:
                _set_peak_artist(
                    artists["peak_DIPG_LWQZPF"], DIPG_LWQZPF,
                    state_LWQZPF["DIPG_peaks"], state_LWQZPF["DIPG_OK"],
                    initial_length, sampling_freq
                )
                _set_peak_artist(
                    artists["peak_BCG_LWQZPF"], BCG_LWQZPF,
                    state_LWQZPF["BCG_J_peaks"], state_LWQZPF["BCG_OK"],
                    initial_length, sampling_freq
                )

        if acquire_index < initial_length:
            artists["status_text"].set_text(
                f"Acquiring: {current_time:.2f}/{total_seconds:.2f} s | initial 3-s initialization period"
            )
        else:
            artists["status_text"].set_text(
                f"Acquiring: {current_time:.2f}/{total_seconds:.2f} s | "
                f"Butterworth IIR/Bessel IIR/FIR updated sample by sample; "
                f"FIR alignment delay = {fir_group_delay_ms:.0f} ms; "
                f"LW-QZPF update interval = {singledistance / sampling_freq:.3f} s; "
                f"filtfilt awaits the complete record"
            )

        fig.canvas.draw_idle()
        plt.pause(draw_refresh_interval)

    if not plt.fignum_exists(fig.number):
        print("The real-time processing window was closed by the user.")
        return

    # Check the remaining detection-hop samples.
    if len(DIPG_butterworth_IIR) > 0:
        stable_end_IIR = initial_length + len(DIPG_butterworth_IIR) - 1
        state_IIR, _ = locate_online_DIPG_BCG_points(
            DIPG_butterworth_IIR, BCG_butterworth_IIR, stable_end_IIR, state_IIR,
            heart_cycle_samples=singledistance, sampling_freq=sampling_freq,
            global_start=initial_length, initial_search_delay_samples=0
        )

    if len(DIPG_Bessel) > 0:
        stable_end_Bessel = initial_length + len(DIPG_Bessel) - 1
        state_Bessel, _ = locate_online_DIPG_BCG_points(
            DIPG_Bessel, BCG_Bessel, stable_end_Bessel, state_Bessel,
            heart_cycle_samples=singledistance, sampling_freq=sampling_freq,
            global_start=initial_length, initial_search_delay_samples=0
        )

    if len(DIPG_FIR) > 0:
        stable_end_FIR = initial_length + len(DIPG_FIR) - 1
        state_FIR, _ = locate_online_DIPG_BCG_points(
            DIPG_FIR, BCG_FIR, stable_end_FIR, state_FIR,
            heart_cycle_samples=singledistance, sampling_freq=sampling_freq,
            global_start=initial_length, initial_search_delay_samples=0
        )

    print("\n================ Online Fiducial-Point Detection Results ================")
    _print_online_detection_result("Butterworth IIR", state_IIR)
    _print_online_detection_result("Bessel IIR", state_Bessel)
    _print_online_detection_result("FIR", state_FIR)
    _print_online_detection_result("LW-QZPF", state_LWQZPF)

    # Apply offline filtfilt after acquisition.
    artists["status_text"].set_text(
        "Raw-data acquisition completed. Applying global filtfilt and fiducial-point detection..."
    )
    fig.canvas.draw_idle()
    plt.pause(0.01)

    Filtfilt_Tproc_start = time.perf_counter()

    IPG_butterworth_0 = butterworth_filter(IPG_raw, 0.5, 5, 3, sampling_freq, 0)
    DIPG_butterworth_0_full = differentiate(IPG_butterworth_0, sampling_freq=sampling_freq)
    BCG_butterworth_0_full = butterworth_filter(BCG_raw, 0.9, 9, 3, sampling_freq, 0)

    DIPG_filtfilt_input = np.asarray(DIPG_butterworth_0_full[initial_length:], dtype=np.float64)
    BCG_filtfilt_input = np.asarray(BCG_butterworth_0_full[initial_length:], dtype=np.float64)

    state_filtfilt = None
    stable_end_filtfilt = initial_length + len(DIPG_filtfilt_input) - 1

    state_filtfilt, _ = locate_online_DIPG_BCG_points(
        DIPG_filtfilt_input, BCG_filtfilt_input, stable_end_filtfilt, state_filtfilt,
        heart_cycle_samples=singledistance, sampling_freq=sampling_freq,
        global_start=initial_length, initial_search_delay_samples=0
    )

    Filtfilt_Tproc = time.perf_counter() - Filtfilt_Tproc_start

    print("\n================ Offline filtfilt Fiducial-Point Detection Results ================")
    _print_online_detection_result("filtfilt", state_filtfilt)

    time_filtfilt = (np.arange(len(DIPG_filtfilt_input)) + initial_length) / sampling_freq

    _set_line_and_autoscale(axes[5, 0], artists["DIPG_filtfilt"], time_filtfilt, DIPG_filtfilt_input)
    _set_line_and_autoscale(axes[5, 1], artists["BCG_filtfilt"], time_filtfilt, BCG_filtfilt_input)

    artists["text_filtfilt_left"].set_visible(False)
    artists["text_filtfilt_right"].set_visible(False)

    _set_peak_artist(
        artists["peak_DIPG_filtfilt"], DIPG_filtfilt_input,
        state_filtfilt["DIPG_peaks"], state_filtfilt["DIPG_OK"], initial_length, sampling_freq
    )
    _set_peak_artist(
        artists["peak_BCG_filtfilt"], BCG_filtfilt_input,
        state_filtfilt["BCG_J_peaks"], state_filtfilt["BCG_OK"], initial_length, sampling_freq
    )

    artists["status_text"].set_text(
        "Processing by all methods is complete. Computing performance metrics..."
    )
    fig.canvas.draw_idle()
    plt.pause(0.05)

    # Crop uniformly and align common beats.
    DIPG_butterworth_IIR = np.asarray(DIPG_butterworth_IIR, dtype=np.float64)
    BCG_butterworth_IIR = np.asarray(BCG_butterworth_IIR, dtype=np.float64)
    DIPG_Bessel = np.asarray(DIPG_Bessel, dtype=np.float64)
    BCG_Bessel = np.asarray(BCG_Bessel, dtype=np.float64)
    DIPG_FIR = np.asarray(DIPG_FIR, dtype=np.float64)
    BCG_FIR = np.asarray(BCG_FIR, dtype=np.float64)
    DIPG_LWQZPF = np.asarray(DIPG_LWQZPF, dtype=np.float64)
    BCG_LWQZPF = np.asarray(BCG_LWQZPF, dtype=np.float64)

    DIPG_IIR_peaks = np.asarray(state_IIR["DIPG_peaks"], dtype=np.float64)
    BCG_IIR_peaks = np.asarray(state_IIR["BCG_J_peaks"], dtype=np.float64)
    DIPG_IIR_OK = np.asarray(state_IIR["DIPG_OK"], dtype=np.int8)
    BCG_IIR_OK = np.asarray(state_IIR["BCG_OK"], dtype=np.int8)

    DIPG_Bessel_peaks = np.asarray(state_Bessel["DIPG_peaks"], dtype=np.float64)
    BCG_Bessel_peaks = np.asarray(state_Bessel["BCG_J_peaks"], dtype=np.float64)
    DIPG_Bessel_OK = np.asarray(state_Bessel["DIPG_OK"], dtype=np.int8)
    BCG_Bessel_OK = np.asarray(state_Bessel["BCG_OK"], dtype=np.int8)

    DIPG_FIR_peaks = np.asarray(state_FIR["DIPG_peaks"], dtype=np.float64)
    BCG_FIR_peaks = np.asarray(state_FIR["BCG_J_peaks"], dtype=np.float64)
    DIPG_FIR_OK = np.asarray(state_FIR["DIPG_OK"], dtype=np.int8)
    BCG_FIR_OK = np.asarray(state_FIR["BCG_OK"], dtype=np.int8)

    DIPG_LWQZPF_peaks = np.asarray(state_LWQZPF["DIPG_peaks"], dtype=np.float64)
    BCG_LWQZPF_peaks = np.asarray(state_LWQZPF["BCG_J_peaks"], dtype=np.float64)
    DIPG_LWQZPF_OK = np.asarray(state_LWQZPF["DIPG_OK"], dtype=np.int8)
    BCG_LWQZPF_OK = np.asarray(state_LWQZPF["BCG_OK"], dtype=np.int8)

    DIPG_filtfilt_peaks = np.asarray(state_filtfilt["DIPG_peaks"], dtype=np.float64)
    BCG_filtfilt_peaks = np.asarray(state_filtfilt["BCG_J_peaks"], dtype=np.float64)
    DIPG_filtfilt_OK = np.asarray(state_filtfilt["DIPG_OK"], dtype=np.int8)
    BCG_filtfilt_OK = np.asarray(state_filtfilt["BCG_OK"], dtype=np.int8)

    common_length = min(
        len(DIPG_LWQZPF), len(BCG_LWQZPF),
        len(DIPG_butterworth_IIR), len(BCG_butterworth_IIR),
        len(DIPG_Bessel), len(BCG_Bessel),
        len(DIPG_FIR), len(BCG_FIR),
        len(DIPG_butterworth_0_full) - initial_length,
        len(BCG_butterworth_0_full) - initial_length
    )

    if common_length <= 0:
        raise ValueError("No common signal interval is available for comparison.")

    DIPG_butterworth_0 = DIPG_butterworth_0_full[initial_length:initial_length + common_length]
    BCG_butterworth_0 = BCG_butterworth_0_full[initial_length:initial_length + common_length]

    DIPG_butterworth_IIR = DIPG_butterworth_IIR[:common_length]
    BCG_butterworth_IIR = BCG_butterworth_IIR[:common_length]
    DIPG_Bessel = DIPG_Bessel[:common_length]
    BCG_Bessel = BCG_Bessel[:common_length]
    DIPG_FIR = DIPG_FIR[:common_length]
    BCG_FIR = BCG_FIR[:common_length]
    DIPG_LWQZPF = DIPG_LWQZPF[:common_length]
    BCG_LWQZPF = BCG_LWQZPF[:common_length]

    DIPG_LWQZPF_peaks, BCG_LWQZPF_peaks, DIPG_LWQZPF_OK, BCG_LWQZPF_OK = crop_and_shift_peak_pairs(
        DIPG_LWQZPF_peaks, BCG_LWQZPF_peaks, DIPG_LWQZPF_OK, BCG_LWQZPF_OK,
        initial_length, common_length
    )
    DIPG_IIR_peaks, BCG_IIR_peaks, DIPG_IIR_OK, BCG_IIR_OK = crop_and_shift_peak_pairs(
        DIPG_IIR_peaks, BCG_IIR_peaks, DIPG_IIR_OK, BCG_IIR_OK,
        initial_length, common_length
    )
    DIPG_Bessel_peaks, BCG_Bessel_peaks, DIPG_Bessel_OK, BCG_Bessel_OK = crop_and_shift_peak_pairs(
        DIPG_Bessel_peaks, BCG_Bessel_peaks, DIPG_Bessel_OK, BCG_Bessel_OK,
        initial_length, common_length
    )
    DIPG_FIR_peaks, BCG_FIR_peaks, DIPG_FIR_OK, BCG_FIR_OK = crop_and_shift_peak_pairs(
        DIPG_FIR_peaks, BCG_FIR_peaks, DIPG_FIR_OK, BCG_FIR_OK,
        initial_length, common_length
    )
    DIPG_filtfilt_peaks, BCG_filtfilt_peaks, DIPG_filtfilt_OK, BCG_filtfilt_OK = crop_and_shift_peak_pairs(
        DIPG_filtfilt_peaks, BCG_filtfilt_peaks, DIPG_filtfilt_OK, BCG_filtfilt_OK,
        initial_length, common_length
    )

    reference_valid = (DIPG_filtfilt_OK == 1) & np.isfinite(DIPG_filtfilt_peaks)
    REFERENCE_DIPG_peaks = DIPG_filtfilt_peaks[reference_valid].copy()

    DIPG_LWQZPF_peaks, BCG_LWQZPF_peaks, DIPG_LWQZPF_OK, BCG_LWQZPF_OK = align_peak_pairs_by_reference(
        REFERENCE_DIPG_peaks, DIPG_LWQZPF_peaks, BCG_LWQZPF_peaks,
        DIPG_LWQZPF_OK, BCG_LWQZPF_OK, singledistance
    )
    DIPG_IIR_peaks, BCG_IIR_peaks, DIPG_IIR_OK, BCG_IIR_OK = align_peak_pairs_by_reference(
        REFERENCE_DIPG_peaks, DIPG_IIR_peaks, BCG_IIR_peaks,
        DIPG_IIR_OK, BCG_IIR_OK, singledistance
    )
    DIPG_Bessel_peaks, BCG_Bessel_peaks, DIPG_Bessel_OK, BCG_Bessel_OK = align_peak_pairs_by_reference(
        REFERENCE_DIPG_peaks, DIPG_Bessel_peaks, BCG_Bessel_peaks,
        DIPG_Bessel_OK, BCG_Bessel_OK, singledistance
    )
    DIPG_FIR_peaks, BCG_FIR_peaks, DIPG_FIR_OK, BCG_FIR_OK = align_peak_pairs_by_reference(
        REFERENCE_DIPG_peaks, DIPG_FIR_peaks, BCG_FIR_peaks,
        DIPG_FIR_OK, BCG_FIR_OK, singledistance
    )
    DIPG_filtfilt_peaks, BCG_filtfilt_peaks, DIPG_filtfilt_OK, BCG_filtfilt_OK = align_peak_pairs_by_reference(
        REFERENCE_DIPG_peaks, DIPG_filtfilt_peaks, BCG_filtfilt_peaks,
        DIPG_filtfilt_OK, BCG_filtfilt_OK, singledistance
    )

    ALL_OK = (
        (DIPG_LWQZPF_OK == 1) & (BCG_LWQZPF_OK == 1)
        & (DIPG_IIR_OK == 1) & (BCG_IIR_OK == 1)
        & (DIPG_Bessel_OK == 1) & (BCG_Bessel_OK == 1)
        & (DIPG_FIR_OK == 1) & (BCG_FIR_OK == 1)
        & (DIPG_filtfilt_OK == 1) & (BCG_filtfilt_OK == 1)
    )

    print("\n================ Common-Beat Validation ================")
    print("Number of common reference beats:", len(REFERENCE_DIPG_peaks))
    print("Number of beats with all ten fiducial detections successful:", int(np.sum(ALL_OK)))

    # Waveform agreement
    DIPG_IIR_r, _ = safe_pearsonr(DIPG_butterworth_IIR, DIPG_butterworth_0)
    DIPG_Bessel_r, _ = safe_pearsonr(DIPG_Bessel, DIPG_butterworth_0)
    DIPG_FIR_r, _ = safe_pearsonr(DIPG_FIR, DIPG_butterworth_0)
    DIPG_LWQZPF_r, _ = safe_pearsonr(DIPG_LWQZPF, DIPG_butterworth_0)

    BCG_IIR_r, _ = safe_pearsonr(BCG_butterworth_IIR, BCG_butterworth_0)
    BCG_Bessel_r, _ = safe_pearsonr(BCG_Bessel, BCG_butterworth_0)
    BCG_FIR_r, _ = safe_pearsonr(BCG_FIR, BCG_butterworth_0)
    BCG_LWQZPF_r, _ = safe_pearsonr(BCG_LWQZPF, BCG_butterworth_0)

    DIPG_IIR_nrmse = calculate_nrmse(DIPG_butterworth_0, DIPG_butterworth_IIR)
    DIPG_Bessel_nrmse = calculate_nrmse(DIPG_butterworth_0, DIPG_Bessel)
    DIPG_FIR_nrmse = calculate_nrmse(DIPG_butterworth_0, DIPG_FIR)
    DIPG_LWQZPF_nrmse = calculate_nrmse(DIPG_butterworth_0, DIPG_LWQZPF)

    BCG_IIR_nrmse = calculate_nrmse(BCG_butterworth_0, BCG_butterworth_IIR)
    BCG_Bessel_nrmse = calculate_nrmse(BCG_butterworth_0, BCG_Bessel)
    BCG_FIR_nrmse = calculate_nrmse(BCG_butterworth_0, BCG_FIR)
    BCG_LWQZPF_nrmse = calculate_nrmse(BCG_butterworth_0, BCG_LWQZPF)

    # Temporal bias and MAE in milliseconds
    DIPG_IIR_bias_sample, DIPG_IIR_mae_sample = calculate_bias_mae(
        DIPG_IIR_peaks, DIPG_filtfilt_peaks, ALL_OK
    )
    DIPG_Bessel_bias_sample, DIPG_Bessel_mae_sample = calculate_bias_mae(
        DIPG_Bessel_peaks, DIPG_filtfilt_peaks, ALL_OK
    )
    DIPG_FIR_bias_sample, DIPG_FIR_mae_sample = calculate_bias_mae(
        DIPG_FIR_peaks, DIPG_filtfilt_peaks, ALL_OK
    )
    DIPG_LWQZPF_bias_sample, DIPG_LWQZPF_mae_sample = calculate_bias_mae(
        DIPG_LWQZPF_peaks, DIPG_filtfilt_peaks, ALL_OK
    )

    BCG_IIR_bias_sample, BCG_IIR_mae_sample = calculate_bias_mae(
        BCG_IIR_peaks, BCG_filtfilt_peaks, ALL_OK
    )
    BCG_Bessel_bias_sample, BCG_Bessel_mae_sample = calculate_bias_mae(
        BCG_Bessel_peaks, BCG_filtfilt_peaks, ALL_OK
    )
    BCG_FIR_bias_sample, BCG_FIR_mae_sample = calculate_bias_mae(
        BCG_FIR_peaks, BCG_filtfilt_peaks, ALL_OK
    )
    BCG_LWQZPF_bias_sample, BCG_LWQZPF_mae_sample = calculate_bias_mae(
        BCG_LWQZPF_peaks, BCG_filtfilt_peaks, ALL_OK
    )

    sample_to_ms = 1000 / sampling_freq

    DIPG_IIR_bias_ms = DIPG_IIR_bias_sample * sample_to_ms
    DIPG_Bessel_bias_ms = DIPG_Bessel_bias_sample * sample_to_ms
    DIPG_FIR_bias_ms = DIPG_FIR_bias_sample * sample_to_ms
    DIPG_LWQZPF_bias_ms = DIPG_LWQZPF_bias_sample * sample_to_ms

    DIPG_IIR_mae_ms = DIPG_IIR_mae_sample * sample_to_ms
    DIPG_Bessel_mae_ms = DIPG_Bessel_mae_sample * sample_to_ms
    DIPG_FIR_mae_ms = DIPG_FIR_mae_sample * sample_to_ms
    DIPG_LWQZPF_mae_ms = DIPG_LWQZPF_mae_sample * sample_to_ms

    BCG_IIR_bias_ms = BCG_IIR_bias_sample * sample_to_ms
    BCG_Bessel_bias_ms = BCG_Bessel_bias_sample * sample_to_ms
    BCG_FIR_bias_ms = BCG_FIR_bias_sample * sample_to_ms
    BCG_LWQZPF_bias_ms = BCG_LWQZPF_bias_sample * sample_to_ms

    BCG_IIR_mae_ms = BCG_IIR_mae_sample * sample_to_ms
    BCG_Bessel_mae_ms = BCG_Bessel_mae_sample * sample_to_ms
    BCG_FIR_mae_ms = BCG_FIR_mae_sample * sample_to_ms
    BCG_LWQZPF_mae_ms = BCG_LWQZPF_mae_sample * sample_to_ms

    # PTT agreement
    if np.sum(ALL_OK) >= 2:
        PTT_IIR = np.abs(DIPG_IIR_peaks[ALL_OK] - BCG_IIR_peaks[ALL_OK]) / sampling_freq * 1000
        PTT_Bessel = np.abs(DIPG_Bessel_peaks[ALL_OK] - BCG_Bessel_peaks[ALL_OK]) / sampling_freq * 1000
        PTT_FIR = np.abs(DIPG_FIR_peaks[ALL_OK] - BCG_FIR_peaks[ALL_OK]) / sampling_freq * 1000
        PTT_LWQZPF = np.abs(DIPG_LWQZPF_peaks[ALL_OK] - BCG_LWQZPF_peaks[ALL_OK]) / sampling_freq * 1000
        PTT_filtfilt = np.abs(DIPG_filtfilt_peaks[ALL_OK] - BCG_filtfilt_peaks[ALL_OK]) / sampling_freq * 1000

        PTT_IIR_r, _ = safe_pearsonr(PTT_IIR, PTT_filtfilt)
        PTT_Bessel_r, _ = safe_pearsonr(PTT_Bessel, PTT_filtfilt)
        PTT_FIR_r, _ = safe_pearsonr(PTT_FIR, PTT_filtfilt)
        PTT_LWQZPF_r, _ = safe_pearsonr(PTT_LWQZPF, PTT_filtfilt)

        _, IIR_LoA_lower, IIR_LoA_upper, IIR_LoA_width = calculate_bland_altman(PTT_IIR, PTT_filtfilt)
        _, Bessel_LoA_lower, Bessel_LoA_upper, Bessel_LoA_width = calculate_bland_altman(
            PTT_Bessel, PTT_filtfilt
        )
        _, FIR_LoA_lower, FIR_LoA_upper, FIR_LoA_width = calculate_bland_altman(
            PTT_FIR, PTT_filtfilt
        )
        _, LWQZPF_LoA_lower, LWQZPF_LoA_upper, LWQZPF_LoA_width = calculate_bland_altman(
            PTT_LWQZPF, PTT_filtfilt
        )
    else:
        PTT_IIR_r = PTT_Bessel_r = PTT_FIR_r = PTT_LWQZPF_r = np.nan
        IIR_LoA_lower = IIR_LoA_upper = IIR_LoA_width = np.nan
        Bessel_LoA_lower = Bessel_LoA_upper = Bessel_LoA_width = np.nan
        FIR_LoA_lower = FIR_LoA_upper = FIR_LoA_width = np.nan
        LWQZPF_LoA_lower = LWQZPF_LoA_upper = LWQZPF_LoA_width = np.nan

    # Continuous-processing and resource estimates
    float_bytes = np.dtype(np.float64).itemsize
    signal_length_full = len(IPG_raw)

    # Butterworth IIR.
    IIR_Mbuf = IIR_buffer_size
    IIR_Twait = IIR_Mbuf / sampling_freq
    IIR_Tproc = np.mean(IIR_Tproc_list) if len(IIR_Tproc_list) > 0 else np.nan
    IIR_RTF = IIR_Tproc / IIR_Twait if np.isfinite(IIR_Tproc) else np.nan
    IIR_TperSample = IIR_Tproc / IIR_Mbuf * 1e6 if np.isfinite(IIR_Tproc) else np.nan
    IIR_Talign = np.nan
    IIR_Mstate_bytes = (
        zi_IPG_IIR.nbytes + zi_BCG_IIR.nbytes
        + b_IPG_IIR.nbytes + a_IPG_IIR.nbytes
        + b_BCG_IIR.nbytes + a_BCG_IIR.nbytes
    )
    IIR_Mmem_bytes = IPG_IIR_buffer.nbytes + BCG_IIR_buffer.nbytes + IIR_Mstate_bytes + 4 * float_bytes
    IIR_Mmem = IIR_Mmem_bytes / 1024
    IIR_MACsPerSample = (
        (len(b_IPG_IIR) + len(a_IPG_IIR) - 1)
        + (len(b_BCG_IIR) + len(a_BCG_IIR) - 1)
    )

    # Bessel IIR.
    Bessel_Mbuf = Bessel_buffer_size
    Bessel_Twait = Bessel_Mbuf / sampling_freq
    Bessel_Tproc = np.mean(Bessel_Tproc_list) if len(Bessel_Tproc_list) > 0 else np.nan
    Bessel_RTF = Bessel_Tproc / Bessel_Twait if np.isfinite(Bessel_Tproc) else np.nan
    Bessel_TperSample = Bessel_Tproc / Bessel_Mbuf * 1e6 if np.isfinite(Bessel_Tproc) else np.nan
    Bessel_Talign = np.nan
    Bessel_Mstate_bytes = (
        zi_IPG_Bessel.nbytes + zi_BCG_Bessel.nbytes
        + b_IPG_Bessel.nbytes + a_IPG_Bessel.nbytes
        + b_BCG_Bessel.nbytes + a_BCG_Bessel.nbytes
    )
    Bessel_Mmem_bytes = (
        IPG_Bessel_buffer.nbytes + BCG_Bessel_buffer.nbytes
        + Bessel_Mstate_bytes + 4 * float_bytes
    )
    Bessel_Mmem = Bessel_Mmem_bytes / 1024
    Bessel_MACsPerSample = (
        (len(b_IPG_Bessel) + len(a_IPG_Bessel) - 1)
        + (len(b_BCG_Bessel) + len(a_BCG_Bessel) - 1)
    )

    # FIR.
    FIR_Mbuf = FIR_buffer_size
    FIR_Twait = FIR_Mbuf / sampling_freq
    FIR_Tproc = np.mean(FIR_Tproc_list) if len(FIR_Tproc_list) > 0 else np.nan
    FIR_RTF = FIR_Tproc / FIR_Twait if np.isfinite(FIR_Tproc) else np.nan
    FIR_TperSample = FIR_Tproc / FIR_Mbuf * 1e6 if np.isfinite(FIR_Tproc) else np.nan
    FIR_Talign = fir_group_delay_samples / sampling_freq
    FIR_Mstate_bytes = (
        zi_IPG_FIR.nbytes + zi_BCG_FIR.nbytes
        + b_IPG_FIR.nbytes + a_IPG_FIR.nbytes
        + b_BCG_FIR.nbytes + a_BCG_FIR.nbytes
    )
    FIR_Mmem_bytes = (
        IPG_FIR_buffer.nbytes + BCG_FIR_buffer.nbytes
        + FIR_Mstate_bytes + 4 * float_bytes
    )
    FIR_Mmem = FIR_Mmem_bytes / 1024
    FIR_MACsPerSample = len(b_IPG_FIR) + len(b_BCG_FIR)

    # LW-QZPF.
    LWQZPF_Mbuf = singledistance + overlap_size
    LWQZPF_Twait = singledistance / sampling_freq
    LWQZPF_Tproc = np.mean(LWQZPF_Tproc_list) if len(LWQZPF_Tproc_list) > 0 else np.nan
    LWQZPF_RTF = LWQZPF_Tproc / LWQZPF_Twait if np.isfinite(LWQZPF_Tproc) else np.nan
    LWQZPF_TperSample = LWQZPF_Tproc / singledistance * 1e6 if np.isfinite(LWQZPF_Tproc) else np.nan
    LWQZPF_Talign = singledistance / sampling_freq
    LWQZPF_Mmem_bytes = (
        IPG_LWQZPF_buffer.nbytes + BCG_LWQZPF_buffer.nbytes
        + DIPG_pre.nbytes + BCG_pre.nbytes
        + 2 * LWQZPF_Mbuf * float_bytes
        + 4 * overlap_size * float_bytes
    )
    LWQZPF_Mmem = LWQZPF_Mmem_bytes / 1024
    LWQZPF_window_length = singledistance + overlap_size
    LWQZPF_MACsPerSample = (
        2 * LWQZPF_window_length / singledistance * IIR_MACsPerSample
    )

    # filtfilt.
    Filtfilt_Mbuf = signal_length_full
    Filtfilt_Twait = Filtfilt_Mbuf / sampling_freq
    Filtfilt_TperSample = Filtfilt_Tproc / Filtfilt_Mbuf * 1e6
    Filtfilt_Talign = Filtfilt_Twait
    Filtfilt_Mstate_bytes = (
        b_IPG_IIR.nbytes + a_IPG_IIR.nbytes
        + b_BCG_IIR.nbytes + a_BCG_IIR.nbytes
    )
    Filtfilt_Mmem_bytes = 5 * Filtfilt_Mbuf * float_bytes + Filtfilt_Mstate_bytes
    Filtfilt_Mmem = Filtfilt_Mmem_bytes / 1024
    Filtfilt_MACsPerSample = 2 * IIR_MACsPerSample

    IIR_Tbuf_ms = IIR_Twait * 1000
    Bessel_Tbuf_ms = Bessel_Twait * 1000
    FIR_Tbuf_ms = FIR_Twait * 1000
    LWQZPF_Tbuf_ms = LWQZPF_Twait * 1000
    Filtfilt_Tbuf_ms = Filtfilt_Twait * 1000

    IIR_Talign_ms = np.nan
    Bessel_Talign_ms = np.nan
    FIR_Talign_ms = FIR_Talign * 1000
    LWQZPF_Talign_ms = LWQZPF_Talign * 1000
    Filtfilt_Talign_ms = Filtfilt_Talign * 1000

    # Console output
    print("\n================ Performance Evaluation ================")
    print("Bias definition: comparison method - filtfilt; positive values indicate later fiducial locations.")
    print(f"Effective common beats = {int(np.sum(ALL_OK))}")
    print(f"Estimated cardiac cycle = {CalculatedAverageCardiacCycle:.4f} s")

    print("\n================ Continuous-Processing and Resource Requirements ================")
    print(
        "Butterworth IIR: Mbuf =", IIR_Mbuf, "samples, Mmem =", f"{IIR_Mmem:.3f}", "KB, "
        "Tsample =", f"{IIR_TperSample:.3f}", "us, RTF =", f"{IIR_RTF:.6f}",
        ", MACs/sample =", f"{IIR_MACsPerSample:.0f}", ", Tbuf =", f"{IIR_Tbuf_ms:.2f}", "ms, Talign = N/A"
    )
    print(
        "Bessel IIR: Mbuf =", Bessel_Mbuf, "samples, Mmem =", f"{Bessel_Mmem:.3f}", "KB, "
        "Tsample =", f"{Bessel_TperSample:.3f}", "us, RTF =", f"{Bessel_RTF:.6f}",
        ", MACs/sample =", f"{Bessel_MACsPerSample:.0f}", ", Tbuf =", f"{Bessel_Tbuf_ms:.2f}",
        "ms, Talign = N/A"
    )
    print(
        "FIR : Mbuf =", FIR_Mbuf, "samples, Mmem =", f"{FIR_Mmem:.3f}", "KB, "
        "Tsample =", f"{FIR_TperSample:.3f}", "us, RTF =", f"{FIR_RTF:.6f}",
        ", MACs/sample =", f"{FIR_MACsPerSample:.0f}", ", Tbuf =", f"{FIR_Tbuf_ms:.2f}",
        "ms, Talign =", f"{FIR_Talign_ms:.2f}", "ms"
    )
    print(
        "LW-QZPF  : Mbuf =", LWQZPF_Mbuf, "samples, Mmem =", f"{LWQZPF_Mmem:.3f}", "KB, "
        "Tsample =", f"{LWQZPF_TperSample:.3f}", "us, RTF =", f"{LWQZPF_RTF:.6f}",
        ", MACs/sample =", f"{LWQZPF_MACsPerSample:.0f}", ", Tbuf =", f"{LWQZPF_Tbuf_ms:.2f}",
        "ms, Talign =", f"{LWQZPF_Talign_ms:.2f}", "ms"
    )
    print(
        "filtfilt : Mbuf =", Filtfilt_Mbuf, "samples, Mmem =", f"{Filtfilt_Mmem:.3f}", "KB, "
        "Tsample =", f"{Filtfilt_TperSample:.3f}", "us, RTF = N/A",
        ", MACs/sample =", f"{Filtfilt_MACsPerSample:.0f}", ", Tbuf =", f"{Filtfilt_Tbuf_ms:.2f}",
        "ms, Talign =", f"{Filtfilt_Talign_ms:.2f}", "ms"
    )

    # Performance tables
    waveform_table = pd.DataFrame(
        [
            ["DIPG", "Pearson r",
             format_table_value(DIPG_IIR_r, 4), format_table_value(DIPG_Bessel_r, 4),
             format_table_value(DIPG_FIR_r, 4), format_table_value(DIPG_LWQZPF_r, 4)],
            ["", "NRMSE",
             format_table_value(DIPG_IIR_nrmse, 4), format_table_value(DIPG_Bessel_nrmse, 4),
             format_table_value(DIPG_FIR_nrmse, 4), format_table_value(DIPG_LWQZPF_nrmse, 4)],
            ["BCG", "Pearson r",
             format_table_value(BCG_IIR_r, 4), format_table_value(BCG_Bessel_r, 4),
             format_table_value(BCG_FIR_r, 4), format_table_value(BCG_LWQZPF_r, 4)],
            ["", "NRMSE",
             format_table_value(BCG_IIR_nrmse, 4), format_table_value(BCG_Bessel_nrmse, 4),
             format_table_value(BCG_FIR_nrmse, 4), format_table_value(BCG_LWQZPF_nrmse, 4)]
        ],
        columns=["Category", "Metric", "Butterworth IIR", "Bessel IIR", "FIR", "LW-QZPF"]
    )

    temporal_table = pd.DataFrame(
        [
            ["DIPG", "Bias (ms)",
             format_table_value(DIPG_IIR_bias_ms, 3), format_table_value(DIPG_Bessel_bias_ms, 3),
             format_table_value(DIPG_FIR_bias_ms, 3), format_table_value(DIPG_LWQZPF_bias_ms, 3)],
            ["", "MAE (ms)",
             format_table_value(DIPG_IIR_mae_ms, 3), format_table_value(DIPG_Bessel_mae_ms, 3),
             format_table_value(DIPG_FIR_mae_ms, 3), format_table_value(DIPG_LWQZPF_mae_ms, 3)],
            ["BCG J", "Bias (ms)",
             format_table_value(BCG_IIR_bias_ms, 3), format_table_value(BCG_Bessel_bias_ms, 3),
             format_table_value(BCG_FIR_bias_ms, 3), format_table_value(BCG_LWQZPF_bias_ms, 3)],
            ["", "MAE (ms)",
             format_table_value(BCG_IIR_mae_ms, 3), format_table_value(BCG_Bessel_mae_ms, 3),
             format_table_value(BCG_FIR_mae_ms, 3), format_table_value(BCG_LWQZPF_mae_ms, 3)],
            ["PTT", "Pearson r",
             format_table_value(PTT_IIR_r, 4), format_table_value(PTT_Bessel_r, 4),
             format_table_value(PTT_FIR_r, 4), format_table_value(PTT_LWQZPF_r, 4)],
            ["", "95% LoA (ms)",
             f"[{IIR_LoA_lower:.3f}, {IIR_LoA_upper:.3f}]" if np.isfinite(IIR_LoA_lower) else "N/A",
             f"[{Bessel_LoA_lower:.3f}, {Bessel_LoA_upper:.3f}]" if np.isfinite(Bessel_LoA_lower) else "N/A",
             f"[{FIR_LoA_lower:.3f}, {FIR_LoA_upper:.3f}]" if np.isfinite(FIR_LoA_lower) else "N/A",
             f"[{LWQZPF_LoA_lower:.3f}, {LWQZPF_LoA_upper:.3f}]" if np.isfinite(LWQZPF_LoA_lower) else "N/A"],
            ["", "95% LoA width (ms)",
             format_table_value(IIR_LoA_width, 3), format_table_value(Bessel_LoA_width, 3),
             format_table_value(FIR_LoA_width, 3), format_table_value(LWQZPF_LoA_width, 3)]
        ],
        columns=["Category", "Metric", "Butterworth IIR", "Bessel IIR", "FIR", "LW-QZPF"]
    )

    resource_table = pd.DataFrame(
        [
            ["Memory demand", "Mbuf (samples)",
             f"{IIR_Mbuf}", f"{Bessel_Mbuf}", f"{FIR_Mbuf}", f"{LWQZPF_Mbuf}", f"{Filtfilt_Mbuf}"],
            ["", "Mmem (KB)",
             format_table_value(IIR_Mmem, 3), format_table_value(Bessel_Mmem, 3),
             format_table_value(FIR_Mmem, 3), format_table_value(LWQZPF_Mmem, 3),
             format_table_value(Filtfilt_Mmem, 3)],
            ["Computation cost", "Tsample (us)",
             format_table_value(IIR_TperSample, 3), format_table_value(Bessel_TperSample, 3),
             format_table_value(FIR_TperSample, 3), format_table_value(LWQZPF_TperSample, 3),
             format_table_value(Filtfilt_TperSample, 3)],
            ["", "RTF",
             format_table_value(IIR_RTF, 4), format_table_value(Bessel_RTF, 4),
             format_table_value(FIR_RTF, 4), format_table_value(LWQZPF_RTF, 4), "N/A"],
            ["", "MACs/sample",
             f"{IIR_MACsPerSample:.0f}", f"{Bessel_MACsPerSample:.0f}",
             f"{FIR_MACsPerSample:.0f}", f"{LWQZPF_MACsPerSample:.0f}",
             f"{Filtfilt_MACsPerSample:.0f}"],
            ["Data latency", "Tbuf (ms)",
             format_table_value(IIR_Tbuf_ms, 1), format_table_value(Bessel_Tbuf_ms, 1),
             format_table_value(FIR_Tbuf_ms, 1), format_table_value(LWQZPF_Tbuf_ms, 1),
             format_table_value(Filtfilt_Tbuf_ms, 1)],
            ["", "Talign (ms)",
             "N/A", "N/A",
             format_table_value(FIR_Talign_ms, 1), format_table_value(LWQZPF_Talign_ms, 1),
             format_table_value(Filtfilt_Talign_ms, 1)]
        ],
        columns=["Category", "Metric", "Butterworth IIR", "Bessel IIR", "FIR", "LW-QZPF", "filtfilt"]
    )

    print("\n")
    print(waveform_table.to_string(index=False))
    print("\n")
    print(temporal_table.to_string(index=False))
    print("\n")
    print(resource_table.to_string(index=False))

    artists["status_text"].set_text(
        "Processing completed: filtfilt output and performance metrics are available."
    )

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    if show_performance_table:
        wait_with_gui_updates(performance_table_delay)
        show_result_table(waveform_table, temporal_table, resource_table)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_realtime_processing()
