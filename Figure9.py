from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


INPUT_FILE = Path("./dataset/TestSet_129samples.xlsx")
OUTPUT_DIR = Path("./result")
OUTPUT_FILE = OUTPUT_DIR / "Figure9.png"
SHEET_NAME = "Sheet1"
EXPECTED_SAMPLE_SIZE = 129

MARKERS = {
    "LW-QZPF PTT": ("LWQZPF_PTT", 1.0),
    "cfPWV": ("cfPWV_Complior", 1.0),
    "baPWV": ("baPWV_Omron", 1.0),
}

CLINICAL_INDICES = {
    "Age": "Age",
    "SBP": "SBP",
    "DBP": "DBP",
    "baPWV": "baPWV_Omron",
    "cfPWV": "cfPWV_Complior",
    "ABI": "ABI_Omron",
}

required_columns = {
    column for column, _direction in MARKERS.values()
} | set(CLINICAL_INDICES.values())

data = pd.read_excel(
    INPUT_FILE,
    sheet_name=SHEET_NAME,
    usecols=sorted(required_columns),
)
if len(data) != EXPECTED_SAMPLE_SIZE:
    raise ValueError(
        f"Expected {EXPECTED_SAMPLE_SIZE} participants, found {len(data)}."
    )

for column in required_columns:
    data[column] = pd.to_numeric(data[column], errors="raise")
if not np.isfinite(data[list(required_columns)].to_numpy(float)).all():
    raise ValueError("Figure 9 variables contain missing or non-finite values.")

correlations = pd.DataFrame(
    index=list(MARKERS),
    columns=list(CLINICAL_INDICES),
    dtype=float,
)
for marker_label, (marker_column, direction) in MARKERS.items():
    marker_values = data[marker_column].to_numpy(float) * direction
    for clinical_label, clinical_column in CLINICAL_INDICES.items():
        result = stats.spearmanr(
            marker_values,
            data[clinical_column].to_numpy(float),
        )
        correlations.loc[marker_label, clinical_label] = float(result[0])

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
    }
)
figure, axis = plt.subplots(figsize=(10.4, 4.0), constrained_layout=True)
heatmap = axis.imshow(
    correlations.to_numpy(float),
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    aspect="auto",
)
axis.set_xticks(np.arange(len(correlations.columns)))
axis.set_xticklabels(correlations.columns, rotation=30, ha="right")
axis.set_yticks(np.arange(len(correlations.index)))
axis.set_yticklabels(correlations.index)
for row in range(len(correlations.index)):
    for column in range(len(correlations.columns)):
        value = correlations.iloc[row, column]
        text_color = "white" if abs(value) >= 0.72 else "#1F1F1F"
        axis.text(
            column,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=12,
        )
axis.set_xticks(np.arange(-0.5, len(correlations.columns), 1), minor=True)
axis.set_yticks(np.arange(-0.5, len(correlations.index), 1), minor=True)
axis.grid(which="minor", color="white", linewidth=1.2)
axis.tick_params(which="minor", bottom=False, left=False)
axis.tick_params(axis="both", length=0)
for spine in axis.spines.values():
    spine.set_visible(False)

axis.set_title("Clinical correlations in the independent test set", pad=12)
colorbar = figure.colorbar(heatmap, ax=axis, pad=0.025)
colorbar.set_label("Spearman rho (risk-oriented)")
colorbar.outline.set_linewidth(0.6)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
figure.savefig(OUTPUT_FILE, dpi=600, bbox_inches="tight", facecolor="white")
plt.close(figure)
