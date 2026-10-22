from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory
from scipy import stats


excel_path = Path("./dataset/TestSet_129samples.xlsx")
sheet_name = "Sheet1"

reference_column = "Filtfilt_PTT"


method_column_candidates = {
    "Bessel IIR": ["Bessel_PTT"],
    "FIR": ["FIR_PTT"],
    "LW-QZPF": ["LWQZPF_PTT"],
}


plot_methods = ["Bessel IIR", "FIR", "LW-QZPF"]


ptt_unit = "ms"


bland_altman_y_margin_ratio = 0.25
bland_altman_min_y_margin = 1.0

output_file = Path("./result/Figure6.png")


color_blue = "#2F6B9A"
color_dark_blue = "#174A70"
color_yellow = "#D9A21B"
color_gray = "#7A7A7A"
color_light_gray = "#D9E2EA"
color_outlier = "#C75B39"

font_size_tick = 9
font_size_label = 10
font_size_title = 11
font_size_legend = 8.5

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": font_size_title,
        "axes.labelsize": font_size_label,
        "axes.linewidth": 0.8,
        "xtick.labelsize": font_size_tick,
        "ytick.labelsize": font_size_tick,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": font_size_legend,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    }
)


def format_p_value(p_value):
    if p_value < 0.001:
        return r"$p$ < 0.001"
    return rf"$p$ = {p_value:.3f}"


def add_common_axis_style(ax):
    ax.grid(True, color=color_light_gray, linewidth=0.55, alpha=0.65, zorder=0)
    ax.tick_params(top=False, right=False, labelsize=font_size_tick)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)


def resolve_method_columns(columns):

    resolved = {}
    missing = {}
    for method, candidates in method_column_candidates.items():
        matched = next((name for name in candidates if name in columns), None)
        if matched is None:
            missing[method] = candidates
        else:
            resolved[method] = matched

    if missing:
        details = "\n".join(
            f"  {method}: expected one of {candidates}"
            for method, candidates in missing.items()
        )
        raise KeyError(
            "Could not find PTT columns for the following methods:\n"
            f"{details}\nExisting Excel columns: {list(columns)}"
        )
    return resolved


def calculate_agreement(reference_values, test_values, method):

    if len(reference_values) < 3:
        raise ValueError(f"{method}: at least three valid paired observations are required.")
    if np.ptp(reference_values) == 0 or np.ptp(test_values) == 0:
        raise ValueError(f"{method}: Pearson correlation cannot use a constant column.")

    pearson_r, p_value = stats.pearsonr(reference_values, test_values)
    regression = stats.linregress(reference_values, test_values)

    mean_values = (reference_values + test_values) / 2.0


    differences = test_values - reference_values
    bias = np.mean(differences)
    difference_sd = np.std(differences, ddof=1)
    lower_loa = bias - 1.96 * difference_sd
    upper_loa = bias + 1.96 * difference_sd
    loa_width = upper_loa - lower_loa

    outside_loa = (differences < lower_loa) | (differences > upper_loa)

    return {
        "pearson_r": pearson_r,
        "p_value": p_value,
        "regression": regression,
        "mean_values": mean_values,
        "differences": differences,
        "bias": bias,
        "difference_sd": difference_sd,
        "lower_loa": lower_loa,
        "upper_loa": upper_loa,
        "loa_width": loa_width,
        "outside_loa": outside_loa,
    }


def plot_pearson(ax, reference_values, test_values, result, method, unit, panel_label):

    plot_min = min(reference_values.min(), test_values.min())
    plot_max = max(reference_values.max(), test_values.max())
    axis_padding = max((plot_max - plot_min) * 0.06, 1.0)
    axis_min = plot_min - axis_padding
    axis_max = plot_max + axis_padding

    x_fit = np.linspace(axis_min, axis_max, 300)
    regression = result["regression"]
    y_fit = regression.intercept + regression.slope * x_fit

    fitted_at_observations = (
        regression.intercept + regression.slope * reference_values
    )
    residuals = test_values - fitted_at_observations
    residual_standard_error = np.sqrt(
        np.sum(residuals**2) / (len(reference_values) - 2)
    )
    x_sum_of_squares = np.sum((reference_values - reference_values.mean()) ** 2)
    t_critical = stats.t.ppf(0.975, df=len(reference_values) - 2)
    confidence_half_width = t_critical * residual_standard_error * np.sqrt(
        1.0 / len(reference_values)
        + (x_fit - reference_values.mean()) ** 2 / x_sum_of_squares
    )

    ax.fill_between(
        x_fit,
        y_fit - confidence_half_width,
        y_fit + confidence_half_width,
        color=color_blue,
        alpha=0.14,
        linewidth=0,
        label="95% CI",
        zorder=1,
    )

    ax.scatter(
        reference_values,
        test_values,
        s=24,
        color=color_blue,
        edgecolors="none",
        linewidths=0,
        alpha=0.82,
        label="Subjects",
        zorder=3,
    )
    ax.plot(x_fit, y_fit, color=color_dark_blue, linewidth=1.5, label="Linear fit", zorder=4)
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        color=color_yellow,
        linewidth=1.3,
        linestyle="--",
        label="Identity line",
        zorder=2,
    )

    statistics_text = (
        rf"$r$ = {result['pearson_r']:.4f}"
        + "\n"
        + format_p_value(result["p_value"])
        + f"\n$n$ = {len(reference_values)}"
    )
    bbox_style = {
        "boxstyle": "round,pad=0.28",
        "facecolor": "white",
        "edgecolor": color_light_gray,
        "linewidth": 0.7,
        "alpha": 0.94,
    }
    ax.text(
        0.045,
        0.955,
        statistics_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#222222",
        bbox=bbox_style,
        zorder=5,
    )

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Filtfilt PTT ({unit})", fontsize=font_size_label)
    ax.set_ylabel(f"{method} PTT ({unit})", fontsize=font_size_label)
    ax.set_title(

        f"({panel_label}) {method}: Pearson correlation",
        loc="left",
        fontsize=font_size_title,
        fontweight="bold",
    )
    ax.legend(loc="lower right", frameon=False, handlelength=2.0)
    add_common_axis_style(ax)


def plot_bland_altman(
    ax,
    result,
    method,
    unit,
    panel_label,
):

    mean_values = result["mean_values"]
    differences = result["differences"]
    outside_loa = result["outside_loa"]
    bias = result["bias"]
    lower_loa = result["lower_loa"]
    upper_loa = result["upper_loa"]

    ax.scatter(
        mean_values[~outside_loa],
        differences[~outside_loa],
        s=24,
        color=color_blue,
        edgecolors="none",
        linewidths=0,
        alpha=0.82,
        label="Within 95% LoA",
        zorder=3,
    )

    if np.any(outside_loa):
        ax.scatter(
            mean_values[outside_loa],
            differences[outside_loa],
            s=28,
            color=color_outlier,
            edgecolors="none",
            linewidths=0,
            alpha=0.90,
            label="Outside 95% LoA",
            zorder=4,
        )

    ax.axhline(0, color=color_gray, linewidth=0.8, linestyle=":", zorder=1)
    ax.axhline(bias, color=color_dark_blue, linewidth=1.5, zorder=2)
    ax.axhline(lower_loa, color=color_yellow, linewidth=1.3, linestyle="--", zorder=2)
    ax.axhline(upper_loa, color=color_yellow, linewidth=1.3, linestyle="--", zorder=2)

    y_candidates = np.concatenate(
        [differences, np.array([0.0, bias, lower_loa, upper_loa])]
    )
    y_span = np.ptp(y_candidates)
    y_padding = max(
        y_span * bland_altman_y_margin_ratio,
        bland_altman_min_y_margin,
    )
    ax.set_ylim(y_candidates.min() - y_padding, y_candidates.max() + y_padding)

    line_label_transform = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.98,
        upper_loa,
        f"Upper LoA = {upper_loa:.2f}",
        transform=line_label_transform,
        ha="right",
        va="bottom",
        color="#8A6400",
    )
    ax.text(
        0.98,
        bias,
        f"Bias = {bias:.2f}",
        transform=line_label_transform,
        ha="right",
        va="bottom",
        color=color_dark_blue,
    )
    ax.text(
        0.98,
        lower_loa,
        f"Lower LoA = {lower_loa:.2f}",
        transform=line_label_transform,
        ha="right",
        va="top",
        color="#8A6400",
    )

    outside_count = int(np.sum(outside_loa))
    outside_percentage = outside_count / len(differences) * 100

    bbox_style = {
        "boxstyle": "round,pad=0.28",
        "facecolor": "white",
        "edgecolor": color_light_gray,
        "linewidth": 0.7,
        "alpha": 0.94,
    }
    ax.text(
        0.045,
        0.955,
        f"Outside LoA: {outside_count}/{len(differences)} "
        f"({outside_percentage:.1f}%)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#222222",
        bbox=bbox_style,
        zorder=5,
    )

    ax.set_xlabel(f"Mean PTT of the two methods ({unit})", fontsize=font_size_label)
    ax.set_ylabel(f"Difference: {method} - filtfilt ({unit})", fontsize=font_size_label)
    ax.set_title(

        f"({panel_label}) {method}: Bland-Altman analysis",
        loc="left",
        fontsize=font_size_title,
        fontweight="bold",
    )
    ax.legend(loc="lower right", frameon=False)
    add_common_axis_style(ax)


if not excel_path.exists():
    raise FileNotFoundError(f"Excel file not found: {excel_path.resolve()}")

raw_data = pd.read_excel(excel_path, sheet_name=sheet_name)

if reference_column not in raw_data.columns:
    raise KeyError(
        f"Missing reference column '{reference_column}'. "
        f"Existing Excel columns: {list(raw_data.columns)}"
    )

method_columns = resolve_method_columns(raw_data.columns)


data = raw_data.copy()

numeric_columns = [reference_column, *method_columns.values()]
for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")


fig, axes = plt.subplots(
    nrows=len(plot_methods),
    ncols=2,
    figsize=(8.0, 3.8 * len(plot_methods)),
    constrained_layout=True,

)

panel_labels = list("abcdef")

for method, test_column in method_columns.items():

    method_data = data.dropna(subset=[reference_column, test_column]).copy()


    reference_values = method_data[reference_column].to_numpy(dtype=float)
    test_values = method_data[test_column].to_numpy(dtype=float)

    result = calculate_agreement(reference_values, test_values, method)

    if method in plot_methods:
        row = plot_methods.index(method)
        plot_pearson(
            axes[row, 0],
            reference_values,
            test_values,
            result,
            method,
            ptt_unit,
            panel_labels[2 * row],
        )

        plot_bland_altman(
            axes[row, 1],
            result,
            method,
            ptt_unit,
            panel_labels[2 * row + 1],
        )

output_file.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_file, dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig)
