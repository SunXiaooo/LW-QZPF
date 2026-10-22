"""Create Figure 10: correlations of LW-QZPF PTT with clinical indices."""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


INPUT_FILE = Path("./dataset/TestSet_129samples.xlsx")
OUTPUT_DIR = Path("./result")
OUTPUT_FILE = OUTPUT_DIR / "Figure8.png"
SHEET_NAME = "Sheet1"
EXPECTED_SAMPLE_COUNT = 129
BOOTSTRAP_REPETITIONS = 4000
RANDOM_SEED = 20261012


argument_parser = argparse.ArgumentParser(
    description="Create Figure 8 using Spearman or Pearson correlation."
)
argument_parser.add_argument(
    "--method",
    choices=("spearman", "pearson"),
    default="pearson",
    help="Correlation method used for the estimate, p-value, and bootstrap CI.",
)
arguments, _ = argument_parser.parse_known_args()
CORRELATION_METHOD = arguments.method


def benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg adjusted p-values in the original order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def calculate_correlation(x, y, method):
    """Return a coefficient and p-value for the selected correlation method."""
    if method == "spearman":
        result = stats.spearmanr(x, y)
    elif method == "pearson":
        result = stats.pearsonr(x, y)
    else:
        raise ValueError(f"Unsupported correlation method: {method}")

    return float(result[0]), float(result[1])


def bootstrap_correlation_ci(x, y, method):
    """Estimate a subject-level bootstrap CI for the selected correlation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    correlations = []

    for _ in range(BOOTSTRAP_REPETITIONS):
        indices = rng.integers(0, len(x), len(x))
        correlation, _ = calculate_correlation(x[indices], y[indices], method)
        if np.isfinite(correlation):
            correlations.append(correlation)

    return np.percentile(correlations, [2.5, 97.5])


def format_q_value(value):
    """Format an FDR-adjusted p-value for the figure annotation."""
    return "<0.001" if value < 0.001 else f"={value:.3f}"


clinical_indices = [
    ("Age", "Age"),
    ("BMI", "BMI"),
    ("HeartRate", "Heart rate"),
    ("SBP", "SBP"),
    ("DBP", "DBP"),
    ("baPWV_Omron", "baPWV"),
    ("cfPWV_Complior", "cfPWV"),
    ("ABI_Omron", "ABI"),
]
method_settings = {
    "spearman": {"display_name": "Spearman", "symbol": "ρ"},
    "pearson": {"display_name": "Pearson", "symbol": "r"},
}
selected_method = method_settings[CORRELATION_METHOD]
required_columns = ["LWQZPF_PTT", *[column for column, _ in clinical_indices]]

data = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, usecols=required_columns)

if len(data) != EXPECTED_SAMPLE_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_SAMPLE_COUNT} samples, but found {len(data)} in {INPUT_FILE}."
    )

data = data.apply(pd.to_numeric, errors="raise")
if not np.isfinite(data.to_numpy(dtype=float)).all():
    raise ValueError("The analysis columns contain missing or non-finite values.")

ptt = data["LWQZPF_PTT"].to_numpy(dtype=float)
rows = []

for column, label in clinical_indices:
    values = data[column].to_numpy(dtype=float)
    coefficient, p_value = calculate_correlation(ptt, values, CORRELATION_METHOD)
    ci_low, ci_high = bootstrap_correlation_ci(ptt, values, CORRELATION_METHOD)
    rows.append(
        {
            "Clinical index": label,
            "Correlation coefficient": coefficient,
            "CI low": float(ci_low),
            "CI high": float(ci_high),
            "p value": p_value,
        }
    )

results = pd.DataFrame(rows)
results["FDR q"] = benjamini_hochberg(results["p value"])
plot_data = results.iloc[::-1].reset_index(drop=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "font.size": 10,
    }
)

figure, axis = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
y_positions = np.arange(len(plot_data))
correlations = plot_data["Correlation coefficient"].to_numpy()
ci_low = plot_data["CI low"].to_numpy()
ci_high = plot_data["CI high"].to_numpy()
colors = np.where(plot_data["FDR q"].to_numpy() < 0.05, "#1769AA", "#8A8A8A")

for index, color in enumerate(colors):
    axis.errorbar(
        correlations[index],
        y_positions[index],
        xerr=[
            [correlations[index] - ci_low[index]],
            [ci_high[index] - correlations[index]],
        ],
        fmt="o",
        color=color,
        ecolor=color,
        markersize=7,
        elinewidth=2,
        capsize=4,
        zorder=3,
    )

axis.axvline(0, color="#444444", linewidth=1, linestyle="--")
axis.set_yticks(y_positions, plot_data["Clinical index"])
axis.set_xlabel(
    f"{selected_method['display_name']} {selected_method['symbol']} (95% CI)"
)
# axis.set_title(
#     f"Association of LW-QZPF PTT with clinical indices "
#     f"(independent test set, n={len(data)})"
# )
axis.set_xlim(-0.85, 0.85)
axis.grid(axis="x", alpha=0.2)

for index, row in plot_data.iterrows():
    axis.text(
        0.52,
        index,
        f"{selected_method['symbol']}={row['Correlation coefficient']:.2f}  "
        f"q{format_q_value(row['FDR q'])}",
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=9,
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
figure.savefig(OUTPUT_FILE, dpi=360, bbox_inches="tight")
plt.show(block=True)
plt.close(figure)

print(results.to_string(index=False))
print(f"Saved: {OUTPUT_FILE}")
