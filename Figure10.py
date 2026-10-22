from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


INPUT_FILE = Path("./dataset/TestSet_129samples.xlsx")
OUTPUT_DIR = Path("./result")
OUTPUT_FILE = OUTPUT_DIR / "Figure10.png"
SHEET_NAME = "Sheet1"
EXPECTED_SAMPLE_COUNT = 129


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def format_q_value(value):
    return "<0.001" if value < 0.001 else f"={value:.3f}"


def significance_stars(value):
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "ns"


def select_first_available(columns, candidates):
    return next((candidate for candidate in candidates if candidate in columns), None)


disease_candidates = [
    ("Hypertension", ["Hypertension"]),
    ("LEAD", ["LEAD"]),
]
markers = [
    ("LWQZPF_PTT", "LW-QZPF PTT", -1),
    ("cfPWV_Complior", "cfPWV", 1),
    ("baPWV_Omron", "baPWV", 1),
]

all_data = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

if len(all_data) != EXPECTED_SAMPLE_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_SAMPLE_COUNT} samples, but found {len(all_data)} in {INPUT_FILE}."
    )

available_diseases = []
for label, candidates in disease_candidates:
    column = select_first_available(all_data.columns, candidates)
    if column is not None:
        available_diseases.append((column, label))

if not available_diseases:
    raise ValueError("No supported disease indicator columns were found in the input workbook.")

required_columns = [
    *[column for column, _ in available_diseases],
    *[column for column, _, _ in markers],
]
data = all_data[required_columns].apply(pd.to_numeric, errors="raise")

if not np.isfinite(data.to_numpy(dtype=float)).all():
    raise ValueError("The analysis columns contain missing or non-finite values.")

for column, label in available_diseases:
    values = set(data[column].unique())
    if not values.issubset({0, 1}):
        raise ValueError(f"{label} must be coded as 0 and 1.")
    if values != {0, 1}:
        raise ValueError(f"{label} must contain both the absent and present groups.")

rows = []
for disease_column, disease_label in available_diseases:
    disease_status = data[disease_column].to_numpy(dtype=int)
    for marker_column, marker_label, risk_direction in markers:
        raw_values = data[marker_column].to_numpy(dtype=float)
        absent = raw_values[disease_status == 0]
        present = raw_values[disease_status == 1]
        mann_whitney = stats.mannwhitneyu(present, absent, alternative="two-sided")

        risk_values = raw_values * risk_direction
        risk_absent = risk_values[disease_status == 0]
        risk_present = risk_values[disease_status == 1]
        auc_statistic = stats.mannwhitneyu(
            risk_present, risk_absent, alternative="two-sided"
        ).statistic
        risk_oriented_auc = float(auc_statistic / (len(risk_present) * len(risk_absent)))

        rows.append(
            {
                "Disease": disease_label,
                "Marker": marker_label,
                "Absent n": len(absent),
                "Present n": len(present),
                "Mann-Whitney p": float(mann_whitney.pvalue),
                "Risk-oriented AUC": risk_oriented_auc,
            }
        )

results = pd.DataFrame(rows)
results["FDR q"] = benjamini_hochberg(results["Mann-Whitney p"])

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
    }
)

row_count = len(available_diseases)
figure, axes = plt.subplots(
    row_count,
    len(markers),
    figsize=(12, 3.15 * row_count),
    constrained_layout=True,
    squeeze=False,
)

for row_index, (disease_column, disease_label) in enumerate(available_diseases):
    for column_index, (marker_column, marker_label, _) in enumerate(markers):
        axis = axes[row_index, column_index]
        values = [
            data.loc[data[disease_column] == group, marker_column].to_numpy(dtype=float)
            for group in (0, 1)
        ]
        boxplot = axis.boxplot(
            values,
            widths=0.48,
            showfliers=True,
            patch_artist=True,
            boxprops={"edgecolor": "#333333", "linewidth": 1.1},
            whiskerprops={"color": "#333333", "linewidth": 1.0},
            capprops={"color": "#333333", "linewidth": 1.0},
            medianprops={"color": "#222222", "linewidth": 1.5},
            flierprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "#555555",
                "markeredgewidth": 0.8,
                "markersize": 4,
            },
        )
        for patch, color in zip(boxplot["boxes"], ["#6F9FC1", "#C56A6A"]):
            patch.set_facecolor(color)
        axis.set_xticks([1, 2])

        result = results.loc[
            (results["Disease"] == disease_label) & (results["Marker"] == marker_label)
        ].iloc[0]

        disease_name = disease_label if disease_label.isupper() else disease_label.lower()
        axis.set_xticklabels(
            [f"Without\n{disease_name}", f"With\n{disease_name}"]
        )
        axis.set_ylabel(marker_label)

        y_min, y_max = axis.get_ylim()
        y_range = y_max - y_min
        bracket_y = y_max + 0.040 * y_range
        bracket_height = 0.025 * y_range
        axis.plot(
            [1, 1, 2, 2],
            [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
            color="#222222",
            linewidth=1.1,
            clip_on=False,
        )
        p_value = result["Mann-Whitney p"]
        axis.text(
            1.5,
            bracket_y + bracket_height + 0.012 * y_range,
            f"{significance_stars(p_value)}",
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=9,
        )
        statistics_y = y_max + 0.260 * y_range
        axis.set_ylim(y_min, y_max + 0.360 * y_range)
        axis.text(
            1.5,
            statistics_y,
            f"FDR q{format_q_value(result['FDR q'])}, "
            f"AUC={result['Risk-oriented AUC']:.3f}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=9,
        )
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#333333")
        axis.spines["bottom"].set_color("#333333")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
figure.savefig(OUTPUT_FILE, dpi=320, bbox_inches="tight")
plt.close(figure)
