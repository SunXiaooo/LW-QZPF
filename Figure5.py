import math
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import bessel, butter, filtfilt, find_peaks, firwin, lfilter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def normalize_signal(signal, new_range=(0, 100)):


    min_value = np.min(signal)
    max_value = np.max(signal)
    old_range = max_value - min_value
    if old_range == 0:
        return np.zeros_like(signal)
    return ((signal - min_value) * (new_range[1] - new_range[0])) / old_range + new_range[0]

def differentiate(signal,sampling_freq=200):


    dt = 1/sampling_freq

    first_derivative = np.gradient(signal, dt)


    return  first_derivative

def init_streaming_derivative_state(ipg_prev=None, bcg_prev=None):
    state = {"ipg": [], "bcg": []}
    if ipg_prev is not None and bcg_prev is not None:
        state["ipg"].append(float(ipg_prev))
        state["bcg"].append(float(bcg_prev))
    return state

def streaming_center_gradient(ipg_filtered_value, bcg_filtered_value, state, sampling_freq=200):


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


    signal = np.asarray(signal, dtype=np.float64)

    local_left = max(1, global_to_local(global_left))
    local_right = min(len(signal) - 2, global_to_local(global_right))

    if local_right - local_left < 3:
        return (
            np.array([], dtype=int),
            np.array([], dtype=np.float64)
        )


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

def align_peak_pairs_by_reference(reference_peaks, dipg_peaks, bcg_peaks, dipg_ok, bcg_ok,
                                  singledistance, expected_delay=0, match_ratio=0.45):


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


excel_path = "./dataset/SummaryTable_429samples.xlsx"
df_dataAll = pd.read_excel(excel_path)
samplesAll = df_dataAll['ID'].copy()

directory = ("./dataset/BCGIPG/")
filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
filenames = [file for file in filenames if file.endswith('.xlsx')]

sampling_freq = 200
Bessel_detection_hop = 5
output_file = Path("./result/Figure5.png")

color_cool = ["#0077EE", "#0066CC", "#0055AA", "#004488", "#003366", "#002244"]
color_warm = ["#FF8811", "#FF7700", "#EE6600", "#CC5500", "#AA4400", "#883300"]

ID_num = 0
for index_sample in range(ID_num, ID_num + 1):

    item_sample = samplesAll.iloc[index_sample]

    filename_index_deal = [i for i, x in enumerate(filenames) if item_sample in x][0]
    filename = filenames[filename_index_deal]
    file_path = directory + os.path.join(filename)+''
    df = pd.read_excel(file_path)


    IPG_raw = df['IPG']
    IPG_raw = -(IPG_raw - (IPG_raw[0]))
    BCG_raw = df['BCG']
    BCG_raw = -(BCG_raw - (BCG_raw[0]))


    initial_length = int(sampling_freq * 3)
    IPG_simulation = IPG_raw[int(sampling_freq * 3):]
    BCG_simulation = BCG_raw[int(sampling_freq * 3):]


    count_LWQZPF = 0
    singledistance = get_SingleCardiacCycleDistance(IPG_raw)
    CalculatedAverageCardiacCycle = round(singledistance/sampling_freq, 4)
    overlap_size = math.ceil(singledistance / 2)

    DIPG_LWQZPF = []
    IPG_LWQZPF_buffer = np.empty(singledistance, dtype=np.float64)
    DIPG_pre = IPG_raw.iloc[initial_length- overlap_size : initial_length].to_numpy(dtype=np.float64)
    DIPG_LWQZPF_peaks = []

    BCG_LWQZPF = []
    BCG_LWQZPF_buffer = np.empty(singledistance, dtype=np.float64)
    BCG_pre = BCG_raw.iloc[initial_length- overlap_size : initial_length].to_numpy(dtype=np.float64)
    BCG_LWQZPF_peaks = []

    DIPG_LWQZPF_OK = []
    BCG_LWQZPF_OK = []
    state_LWQZPF = None


    count_IIR = 0

    DIPG_butterworth_IIR = []
    IPG_IIR_buffer = np.empty(sampling_freq, dtype=np.float64)
    nyquist = 0.5 * sampling_freq

    b_IPG_IIR, a_IPG_IIR = butter(3, [0.5 / nyquist, 5 / nyquist], btype='band')
    zi_IPG_IIR = np.zeros(max(len(a_IPG_IIR), len(b_IPG_IIR)) - 1, dtype=np.float64)
    _, zi_IPG_IIR = lfilter(b_IPG_IIR, a_IPG_IIR, np.asarray(IPG_raw.iloc[:initial_length], dtype=np.float64), zi=zi_IPG_IIR)
    DIPG_IIR_peaks = []

    BCG_butterworth_IIR = []
    BCG_IIR_buffer = np.empty(sampling_freq, dtype=np.float64)
    nyquist = 0.5 * sampling_freq

    b_BCG_IIR, a_BCG_IIR = butter(3, [0.9 / nyquist, 9 / nyquist], btype='band')
    zi_BCG_IIR = np.zeros(max(len(a_BCG_IIR), len(b_BCG_IIR)) - 1, dtype=np.float64)
    _, zi_BCG_IIR = lfilter(b_BCG_IIR, a_BCG_IIR, np.asarray(BCG_raw.iloc[:initial_length], dtype=np.float64), zi=zi_BCG_IIR)
    BCG_IIR_peaks = []

    DIPG_IIR_OK = []
    BCG_IIR_OK = []
    state_IIR = None


    count_Bessel = 0
    Bessel_buffer_size = 1

    DIPG_Bessel = []
    IPG_Bessel_buffer = np.empty(Bessel_buffer_size, dtype=np.float64)

    b_IPG_Bessel, a_IPG_Bessel = bessel(3, [0.5, 5], btype='band', norm='mag', fs=sampling_freq)
    zi_IPG_Bessel = np.zeros(max(len(a_IPG_Bessel), len(b_IPG_Bessel)) - 1, dtype=np.float64)
    IPG_Bessel_initial_filtered, zi_IPG_Bessel = lfilter(b_IPG_Bessel, a_IPG_Bessel,
                                                         np.asarray(IPG_raw.iloc[:initial_length], dtype=np.float64), zi=zi_IPG_Bessel)
    DIPG_Bessel_peaks = []

    BCG_Bessel = []
    BCG_Bessel_buffer = np.empty(Bessel_buffer_size, dtype=np.float64)
    b_BCG_Bessel, a_BCG_Bessel = bessel(3, [0.9, 9], btype='band', norm='mag', fs=sampling_freq)
    zi_BCG_Bessel = np.zeros(max(len(a_BCG_Bessel), len(b_BCG_Bessel)) - 1, dtype=np.float64)
    BCG_Bessel_initial_filtered, zi_BCG_Bessel = lfilter(b_BCG_Bessel, a_BCG_Bessel,
                                                         np.asarray(BCG_raw.iloc[:initial_length], dtype=np.float64), zi=zi_BCG_Bessel)
    BCG_Bessel_peaks = []

    Bessel_stream_state = init_streaming_derivative_state(IPG_Bessel_initial_filtered[-1], BCG_Bessel_initial_filtered[-1])
    DIPG_Bessel_OK = []
    BCG_Bessel_OK = []
    state_Bessel = None


    count_FIR = 0

    fir_numtaps = 201
    fir_group_delay_samples = (fir_numtaps - 1) // 2

    DIPG_FIR = []
    IPG_FIR_buffer = np.empty(sampling_freq, dtype=np.float64)

    b_IPG_FIR = firwin(numtaps=fir_numtaps, cutoff=[0.5, 5], pass_zero=False, fs=sampling_freq, window='hamming')
    a_IPG_FIR = np.array([1.0], dtype=np.float64)
    zi_IPG_FIR = np.zeros(len(b_IPG_FIR) - 1, dtype=np.float64)
    _, zi_IPG_FIR = lfilter( b_IPG_FIR, a_IPG_FIR, np.asarray(IPG_raw.iloc[:initial_length], dtype=np.float64), zi=zi_IPG_FIR)
    DIPG_FIR_peaks = []

    BCG_FIR = []
    BCG_FIR_buffer = np.empty(sampling_freq, dtype=np.float64)

    b_BCG_FIR = firwin(numtaps=fir_numtaps, cutoff=[0.9, 9], pass_zero=False, fs=sampling_freq, window='hamming')
    a_BCG_FIR = np.array([1.0], dtype=np.float64)
    zi_BCG_FIR = np.zeros(len(b_BCG_FIR) - 1, dtype=np.float64)
    _, zi_BCG_FIR = lfilter( b_BCG_FIR, a_BCG_FIR, np.asarray(BCG_raw.iloc[:initial_length], dtype=np.float64), zi=zi_BCG_FIR)
    BCG_FIR_peaks = []

    DIPG_FIR_OK = []
    BCG_FIR_OK = []
    state_FIR = None


    for i in range(0,len(IPG_simulation)):
        sample_IPG = IPG_simulation.values[i]
        sample_BCG = BCG_simulation.values[i]


        IPG_LWQZPF_buffer[count_LWQZPF] = sample_IPG
        BCG_LWQZPF_buffer[count_LWQZPF] = sample_BCG
        count_LWQZPF += 1
        if count_LWQZPF == singledistance:
            count_LWQZPF = 0

            currentDIPG_LWQZPF = np.concatenate((DIPG_pre, IPG_LWQZPF_buffer))

            currentDIPG_LWQZPF = Realtime_Processed(currentDIPG_LWQZPF,0.5, 5)
            currentDIPG_LWQZPF = differentiate(currentDIPG_LWQZPF)

            DIPG_LWQZPF = Hann_Stitch(DIPG_LWQZPF, currentDIPG_LWQZPF, overlap_size)
            DIPG_pre = IPG_LWQZPF_buffer[-overlap_size:].copy()

            currentBCG_LWQZPF = np.concatenate((BCG_pre, BCG_LWQZPF_buffer))
            currentBCG_LWQZPF = Realtime_Processed(currentBCG_LWQZPF, 0.9, 9)
            BCG_LWQZPF = Hann_Stitch(BCG_LWQZPF, currentBCG_LWQZPF, overlap_size)
            BCG_pre = BCG_LWQZPF_buffer[-overlap_size:].copy()


            stable_end_LWQZPF = initial_length + len(DIPG_LWQZPF) - overlap_size - 1
            state_LWQZPF, new_pairs_LWQZPF = locate_online_DIPG_BCG_points(DIPG_LWQZPF, BCG_LWQZPF, stable_end_LWQZPF,
                state_LWQZPF, heart_cycle_samples=singledistance, sampling_freq=sampling_freq, global_start=initial_length,
                initial_search_delay_samples=0)

            DIPG_LWQZPF_peaks = state_LWQZPF["DIPG_peaks"]
            BCG_LWQZPF_peaks = state_LWQZPF["BCG_J_peaks"]
            DIPG_LWQZPF_OK = state_LWQZPF["DIPG_OK"]
            BCG_LWQZPF_OK = state_LWQZPF["BCG_OK"]


        IPG_IIR_buffer[count_IIR] = sample_IPG
        BCG_IIR_buffer[count_IIR] = sample_BCG
        count_IIR += 1
        if count_IIR == sampling_freq:
            count_IIR = 0

            currentDIPG_IIR, zi_IPG_IIR = lfilter( b_IPG_IIR, a_IPG_IIR, IPG_IIR_buffer, zi=zi_IPG_IIR)
            currentDIPG_IIR = differentiate(currentDIPG_IIR)
            DIPG_butterworth_IIR.extend(currentDIPG_IIR.tolist())

            currentBCG_IIR, zi_BCG_IIR = lfilter(b_BCG_IIR, a_BCG_IIR, BCG_IIR_buffer, zi=zi_BCG_IIR)
            BCG_butterworth_IIR.extend(currentBCG_IIR.tolist())


            stable_end_IIR = initial_length + len(DIPG_butterworth_IIR) - 1
            state_IIR, new_pairs_IIR = locate_online_DIPG_BCG_points(DIPG_butterworth_IIR, BCG_butterworth_IIR,
                stable_end_IIR, state_IIR, heart_cycle_samples=singledistance, sampling_freq=sampling_freq,
                global_start=initial_length, initial_search_delay_samples=0)
            DIPG_IIR_peaks = state_IIR["DIPG_peaks"]
            BCG_IIR_peaks = state_IIR["BCG_J_peaks"]
            DIPG_IIR_OK = state_IIR["DIPG_OK"]
            BCG_IIR_OK = state_IIR["BCG_OK"]


        IPG_Bessel_buffer[0] = sample_IPG
        BCG_Bessel_buffer[0] = sample_BCG

        currentIPG_Bessel, zi_IPG_Bessel = lfilter(b_IPG_Bessel, a_IPG_Bessel,
                                                          IPG_Bessel_buffer, zi=zi_IPG_Bessel)
        currentBCG_Bessel, zi_BCG_Bessel = lfilter(b_BCG_Bessel, a_BCG_Bessel,
                                                          BCG_Bessel_buffer, zi=zi_BCG_Bessel)
        currentDIPG_Bessel, currentBCG_Bessel_center = streaming_center_gradient(
            currentIPG_Bessel[0], currentBCG_Bessel[0], Bessel_stream_state, sampling_freq)

        if currentDIPG_Bessel is not None:
            DIPG_Bessel.append(currentDIPG_Bessel)
            BCG_Bessel.append(currentBCG_Bessel_center)

            if len(DIPG_Bessel) % Bessel_detection_hop == 0:
                stable_end_Bessel = initial_length + len(DIPG_Bessel) - 1
                state_Bessel, new_pairs_Bessel = locate_online_DIPG_BCG_points(DIPG_Bessel, BCG_Bessel,
                    stable_end_Bessel, state_Bessel, heart_cycle_samples=singledistance,
                    sampling_freq=sampling_freq, global_start=initial_length,
                    initial_search_delay_samples=0)
                if state_Bessel is not None:
                    DIPG_Bessel_peaks = state_Bessel["DIPG_peaks"]
                    BCG_Bessel_peaks = state_Bessel["BCG_J_peaks"]
                    DIPG_Bessel_OK = state_Bessel["DIPG_OK"]
                    BCG_Bessel_OK = state_Bessel["BCG_OK"]


        IPG_FIR_buffer[count_FIR] = sample_IPG
        BCG_FIR_buffer[count_FIR] = sample_BCG
        count_FIR += 1
        if count_FIR == sampling_freq:
            count_FIR = 0

            currentDIPG_FIR, zi_IPG_FIR = lfilter(b_IPG_FIR, a_IPG_FIR, IPG_FIR_buffer, zi=zi_IPG_FIR)
            currentDIPG_FIR = differentiate(currentDIPG_FIR)
            DIPG_FIR.extend(currentDIPG_FIR.tolist())

            currentBCG_FIR, zi_BCG_FIR = lfilter(b_BCG_FIR, a_BCG_FIR, BCG_FIR_buffer, zi=zi_BCG_FIR)
            BCG_FIR.extend(currentBCG_FIR.tolist())


            stable_end_FIR = initial_length + len(DIPG_FIR) - 1
            state_FIR, new_pairs_FIR = locate_online_DIPG_BCG_points(DIPG_FIR, BCG_FIR, stable_end_FIR, state_FIR,
                                                                     heart_cycle_samples=singledistance,
                                                                     sampling_freq=sampling_freq,
                                                                     global_start=initial_length,
                                                                     initial_search_delay_samples=fir_group_delay_samples)

            DIPG_FIR_peaks = state_FIR["DIPG_peaks"]
            BCG_FIR_peaks = state_FIR["BCG_J_peaks"]
            DIPG_FIR_OK = state_FIR["DIPG_OK"]
            BCG_FIR_OK = state_FIR["BCG_OK"]


    if len(DIPG_Bessel) > 0:
        stable_end_Bessel = initial_length + len(DIPG_Bessel) - 1
        state_Bessel, _ = locate_online_DIPG_BCG_points(DIPG_Bessel, BCG_Bessel, stable_end_Bessel,
            state_Bessel, heart_cycle_samples=singledistance, sampling_freq=sampling_freq,
            global_start=initial_length, initial_search_delay_samples=0)
        if state_Bessel is not None:
            DIPG_Bessel_peaks = state_Bessel["DIPG_peaks"]
            BCG_Bessel_peaks = state_Bessel["BCG_J_peaks"]
            DIPG_Bessel_OK = state_Bessel["DIPG_OK"]
            BCG_Bessel_OK = state_Bessel["BCG_OK"]


    IPG_raw = np.float64(IPG_raw)
    DIPG_butterworth_IIR = np.float64(DIPG_butterworth_IIR)
    DIPG_Bessel = np.float64(DIPG_Bessel)
    DIPG_FIR = np.float64(DIPG_FIR)
    DIPG_LWQZPF = np.float64(DIPG_LWQZPF)

    BCG_raw = np.float64(BCG_raw)
    BCG_butterworth_IIR = np.float64(BCG_butterworth_IIR)
    BCG_Bessel = np.float64(BCG_Bessel)
    BCG_FIR = np.float64(BCG_FIR)
    BCG_LWQZPF = np.float64(BCG_LWQZPF)


    IPG_butterworth_0 = butterworth_filter(IPG_raw, 0.5, 5, 3, 200, 0)
    DIPG_butterworth_0 = differentiate(IPG_butterworth_0)
    BCG_butterworth_0 = butterworth_filter(BCG_raw, 0.9, 9, 3, 200, 0)

    DIPG_filtfilt_input = np.asarray(DIPG_butterworth_0[initial_length:],dtype=np.float64)
    BCG_filtfilt_input = np.asarray(BCG_butterworth_0[initial_length:],dtype=np.float64)

    state_filtfilt = None
    stable_end_filtfilt = (initial_length + len(DIPG_filtfilt_input) - 1)

    state_filtfilt, new_pairs_filtfilt = locate_online_DIPG_BCG_points(DIPG_filtfilt_input, BCG_filtfilt_input,
        stable_end_filtfilt, state_filtfilt, heart_cycle_samples=singledistance, sampling_freq=sampling_freq,
        global_start=initial_length, initial_search_delay_samples=0)

    DIPG_filtfilt_peaks = np.asarray(state_filtfilt["DIPG_peaks"], dtype=np.float64)
    BCG_filtfilt_peaks = np.asarray(state_filtfilt["BCG_J_peaks"], dtype=np.float64)
    DIPG_filtfilt_OK = np.asarray(state_filtfilt["DIPG_OK"], dtype=np.int8)
    BCG_filtfilt_OK = np.asarray(state_filtfilt["BCG_OK"],dtype=np.int8)


    DIPG_FIR = DIPG_FIR[fir_group_delay_samples:]
    BCG_FIR = BCG_FIR[fir_group_delay_samples:]
    DIPG_FIR_peaks = np.asarray(DIPG_FIR_peaks, dtype=np.float64) - fir_group_delay_samples
    BCG_FIR_peaks = np.asarray(BCG_FIR_peaks, dtype=np.float64) - fir_group_delay_samples


    n_keep = initial_length
    common_length = min(len(DIPG_LWQZPF), len(BCG_LWQZPF), len(DIPG_butterworth_IIR), len(BCG_butterworth_IIR),
                        len(DIPG_Bessel), len(BCG_Bessel), len(DIPG_FIR), len(BCG_FIR),
                        len(DIPG_butterworth_0) - n_keep, len(BCG_butterworth_0) - n_keep)

    DIPG_raw = differentiate(IPG_raw)[n_keep:n_keep + common_length]
    DIPG_butterworth_0 = DIPG_butterworth_0[n_keep:n_keep + common_length]
    DIPG_butterworth_IIR = DIPG_butterworth_IIR[:common_length]
    DIPG_Bessel = DIPG_Bessel[:common_length]
    DIPG_FIR = DIPG_FIR[:common_length]
    DIPG_LWQZPF = DIPG_LWQZPF[:common_length]

    BCG_raw = BCG_raw[n_keep:n_keep + common_length]
    BCG_butterworth_0 = BCG_butterworth_0[n_keep:n_keep + common_length]
    BCG_butterworth_IIR = BCG_butterworth_IIR[:common_length]
    BCG_Bessel = BCG_Bessel[:common_length]
    BCG_FIR = BCG_FIR[:common_length]
    BCG_LWQZPF = BCG_LWQZPF[:common_length]


    def crop_and_shift_peak_pairs(dipg_peaks, bcg_peaks, dipg_ok, bcg_ok, crop_global_start, crop_length):


        dipg_peaks = np.asarray(dipg_peaks, dtype=np.float64)
        bcg_peaks = np.asarray(bcg_peaks, dtype=np.float64)
        dipg_ok = np.asarray(dipg_ok, dtype=np.int8)
        bcg_ok = np.asarray(bcg_ok, dtype=np.int8)

        event_len = min(len(dipg_peaks), len(bcg_peaks), len(dipg_ok), len(bcg_ok))
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


    DIPG_LWQZPF_peaks, BCG_LWQZPF_peaks, DIPG_LWQZPF_OK, BCG_LWQZPF_OK = crop_and_shift_peak_pairs(
        DIPG_LWQZPF_peaks, BCG_LWQZPF_peaks, DIPG_LWQZPF_OK, BCG_LWQZPF_OK, n_keep, common_length)
    DIPG_IIR_peaks, BCG_IIR_peaks, DIPG_IIR_OK, BCG_IIR_OK = crop_and_shift_peak_pairs(
        DIPG_IIR_peaks, BCG_IIR_peaks, DIPG_IIR_OK, BCG_IIR_OK, n_keep, common_length)
    DIPG_Bessel_peaks, BCG_Bessel_peaks, DIPG_Bessel_OK, BCG_Bessel_OK = crop_and_shift_peak_pairs(
        DIPG_Bessel_peaks, BCG_Bessel_peaks, DIPG_Bessel_OK, BCG_Bessel_OK, n_keep, common_length)
    DIPG_FIR_peaks, BCG_FIR_peaks, DIPG_FIR_OK, BCG_FIR_OK = crop_and_shift_peak_pairs(
        DIPG_FIR_peaks, BCG_FIR_peaks, DIPG_FIR_OK, BCG_FIR_OK, n_keep, common_length)
    DIPG_filtfilt_peaks, BCG_filtfilt_peaks, DIPG_filtfilt_OK, BCG_filtfilt_OK = crop_and_shift_peak_pairs(
        DIPG_filtfilt_peaks, BCG_filtfilt_peaks, DIPG_filtfilt_OK, BCG_filtfilt_OK, n_keep, common_length)


    reference_valid = ((DIPG_filtfilt_OK == 1) & np.isfinite(DIPG_filtfilt_peaks))
    REFERENCE_DIPG_peaks = DIPG_filtfilt_peaks[reference_valid].copy()


    DIPG_LWQZPF_peaks, BCG_LWQZPF_peaks, DIPG_LWQZPF_OK, BCG_LWQZPF_OK = align_peak_pairs_by_reference(
        REFERENCE_DIPG_peaks, DIPG_LWQZPF_peaks, BCG_LWQZPF_peaks, DIPG_LWQZPF_OK, BCG_LWQZPF_OK, singledistance)
    DIPG_IIR_peaks, BCG_IIR_peaks, DIPG_IIR_OK, BCG_IIR_OK = align_peak_pairs_by_reference(
        REFERENCE_DIPG_peaks, DIPG_IIR_peaks, BCG_IIR_peaks, DIPG_IIR_OK, BCG_IIR_OK, singledistance)
    DIPG_Bessel_peaks, BCG_Bessel_peaks, DIPG_Bessel_OK, BCG_Bessel_OK = align_peak_pairs_by_reference(
        REFERENCE_DIPG_peaks, DIPG_Bessel_peaks, BCG_Bessel_peaks, DIPG_Bessel_OK, BCG_Bessel_OK,
        singledistance)
    DIPG_FIR_peaks, BCG_FIR_peaks, DIPG_FIR_OK, BCG_FIR_OK = align_peak_pairs_by_reference(
        REFERENCE_DIPG_peaks, DIPG_FIR_peaks, BCG_FIR_peaks, DIPG_FIR_OK, BCG_FIR_OK, singledistance)
    DIPG_filtfilt_peaks, BCG_filtfilt_peaks, DIPG_filtfilt_OK, BCG_filtfilt_OK = align_peak_pairs_by_reference(
        REFERENCE_DIPG_peaks, DIPG_filtfilt_peaks, BCG_filtfilt_peaks, DIPG_filtfilt_OK, BCG_filtfilt_OK, singledistance)


    OK_length = [len(DIPG_LWQZPF_OK), len(BCG_LWQZPF_OK), len(DIPG_IIR_OK), len(BCG_IIR_OK),
                 len(DIPG_Bessel_OK), len(BCG_Bessel_OK), len(DIPG_FIR_OK), len(BCG_FIR_OK),
                 len(DIPG_filtfilt_OK), len(BCG_filtfilt_OK)]

    if len(set(OK_length)) != 1:
        raise ValueError("The ten peak-status arrays must have equal lengths.")


    ALL_OK = ((DIPG_LWQZPF_OK == 1) & (BCG_LWQZPF_OK == 1) & (DIPG_IIR_OK == 1) & (BCG_IIR_OK == 1) &
              (DIPG_Bessel_OK == 1) & (BCG_Bessel_OK == 1) & (DIPG_FIR_OK == 1) & (BCG_FIR_OK == 1) &
              (DIPG_filtfilt_OK == 1) & (BCG_filtfilt_OK == 1))
    if  np.sum(ALL_OK) < 3:
        continue


    if True:
        font_size_tick = 10
        font_size_label = 12
        font_size_title = 14
        font_size_legend = 11


        time_axis = (np.arange(common_length) + n_keep) / sampling_freq

        DIPG_raw_plot = normalize_signal(DIPG_raw)
        DIPG_IIR_plot = normalize_signal(DIPG_butterworth_IIR)
        DIPG_Bessel_plot = normalize_signal(DIPG_Bessel)
        DIPG_FIR_plot = normalize_signal(DIPG_FIR)
        DIPG_0_plot = normalize_signal(DIPG_butterworth_0)
        DIPG_LWQZPF_plot = normalize_signal(DIPG_LWQZPF)

        valid = ((DIPG_IIR_OK == 1) & np.isfinite(DIPG_IIR_peaks))
        DIPG_IIR_peaks_plot = DIPG_IIR_peaks[valid].astype(int)
        valid = ((DIPG_Bessel_OK == 1) & np.isfinite(DIPG_Bessel_peaks))
        DIPG_Bessel_peaks_plot = DIPG_Bessel_peaks[valid].astype(int)
        valid = ((DIPG_FIR_OK == 1) & np.isfinite(DIPG_FIR_peaks))
        DIPG_FIR_peaks_plot = DIPG_FIR_peaks[valid].astype(int)
        valid = ((DIPG_filtfilt_OK == 1) & np.isfinite(DIPG_filtfilt_peaks))
        DIPG_filtfilt_peaks_plot = DIPG_filtfilt_peaks[valid].astype(int)
        valid = ((DIPG_LWQZPF_OK == 1) & np.isfinite(DIPG_LWQZPF_peaks))
        DIPG_LWQZPF_peaks_plot = DIPG_LWQZPF_peaks[valid].astype(int)

        fig, axes = plt.subplots(6, 1, figsize=(15, 10), sharex=True)
        dipg_subplot_data = [
            ('(a) Unfiltered DIPG', DIPG_raw_plot, None, color_warm[0]),
            ('(b) Butterworth IIR', DIPG_IIR_plot, DIPG_IIR_peaks_plot, color_warm[1]),
            ('(c) Bessel IIR', DIPG_Bessel_plot, DIPG_Bessel_peaks_plot, color_warm[2]),
            ('(d) FIR', DIPG_FIR_plot, DIPG_FIR_peaks_plot, color_warm[3]),
            ('(e) LW-QZPF', DIPG_LWQZPF_plot, DIPG_LWQZPF_peaks_plot, color_warm[4]),
            ('(f) Offline filtfilt', DIPG_0_plot, DIPG_filtfilt_peaks_plot, color_warm[5])
        ]

        for ax, (method_name, signal_plot, peaks_plot, line_color) in zip(axes, dipg_subplot_data):
            ax.plot(time_axis, signal_plot, color=line_color, linewidth=1.8)
            if peaks_plot is not None:
                ax.plot(
                    time_axis[peaks_plot], signal_plot[peaks_plot], '^',
                    color='red', linestyle='None', markersize=6
                )
            ax.set_title(method_name, loc='left', fontsize=font_size_title, fontweight='bold')
            ax.tick_params(axis='both', labelsize=font_size_tick)
            ax.set_ylim(-5, 105)

        axes[-1].set_xlabel('Time (s)', fontsize=font_size_label)
        fig.text(0.02, 0.5, 'Normalized amplitude', rotation='vertical', va='center', fontsize=font_size_label)

        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.96, hspace=0.28)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
