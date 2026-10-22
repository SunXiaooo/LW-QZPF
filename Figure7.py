from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


INPUT_FILE = Path("./dataset/TestSet_129samples.xlsx")
OUTPUT_DIR = Path("./result")
OUTPUT_FILE = OUTPUT_DIR / "Figure7.png"
SHEET_NAME = "Sheet1"
EXPECTED_SAMPLE_SIZE = 129


data = pd.read_excel(
    INPUT_FILE,
    sheet_name=SHEET_NAME,
    usecols=["cfPTT_Complior", "LWQZPF_PTT"],
)
if len(data) != EXPECTED_SAMPLE_SIZE:
    raise ValueError(
        f"Expected {EXPECTED_SAMPLE_SIZE} participants, found {len(data)}."
    )

x = pd.to_numeric(data["cfPTT_Complior"], errors="raise").to_numpy(float)
y = pd.to_numeric(data["LWQZPF_PTT"], errors="raise").to_numpy(float)
if not (np.isfinite(x).all() and np.isfinite(y).all()):
    raise ValueError("Figure 7 variables contain missing or non-finite values.")

regression = stats.linregress(x, y)
pearson_r, pearson_p = stats.pearsonr(x, y)

x_grid = np.linspace(x.min(), x.max(), 300)
y_fit = regression.intercept + regression.slope * x_grid
residuals = y - (regression.intercept + regression.slope * x)
residual_se = np.sqrt(np.sum(residuals**2) / (len(x) - 2))
sxx = np.sum((x - x.mean()) ** 2)
mean_se = residual_se * np.sqrt(1 / len(x) + (x_grid - x.mean()) ** 2 / sxx)
t_critical = stats.t.ppf(0.975, df=len(x) - 2)
ci_low = y_fit - t_critical * mean_se
ci_high = y_fit + t_critical * mean_se

fisher_z = np.arctanh(pearson_r)
fisher_se = 1 / np.sqrt(len(x) - 3)
r_low, r_high = np.tanh(
    fisher_z + np.array([-1, 1]) * stats.norm.ppf(0.975) * fisher_se
)
p_text = "p<0.001" if pearson_p < 0.001 else f"p={pearson_p:.3f}"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
    }
)
figure, axis = plt.subplots(figsize=(7.2, 5.7), constrained_layout=True)
axis.scatter(
    x,
    y,
    s=38,
    alpha=0.72,
    color="#1769AA",
    edgecolors="white",
    linewidths=0.45,
    label="Participants",
    zorder=3,
)
axis.fill_between(
    x_grid,
    ci_low,
    ci_high,
    color="#F29F3D",
    alpha=0.24,
    label="95% CI of fitted mean",
    zorder=1,
)
axis.plot(
    x_grid,
    y_fit,
    color="#D56A00",
    linewidth=2.3,
    label="Linear fit",
    zorder=2,
)
axis.set_xlabel("Complior CFPTT (ms)")
axis.set_ylabel("LW-QZPF PTT (ms)")
axis.set_title("Association of LW-QZPF PTT with CFPTT")
axis.text(
    0.04,
    0.96,
    (
        f"n={len(x)}\n"
        f"Pearson r={pearson_r:.3f}\n"
        f"95% CI {r_low:.3f}-{r_high:.3f}\n"
        f"{p_text}"
    ),
    transform=axis.transAxes,
    va="top",
    ha="left",
    bbox={
        "boxstyle": "round,pad=0.35",
        "facecolor": "white",
        "edgecolor": "#B8B8B8",
        "alpha": 0.92,
    },
)
axis.grid(alpha=0.18)
axis.legend(frameon=False, loc="lower right")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
figure.savefig(OUTPUT_FILE, dpi=600, bbox_inches="tight", facecolor="white")
plt.close(figure)
