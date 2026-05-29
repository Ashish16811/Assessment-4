"""
============================================================
PRT564 – Data Analytics and Visualisation
Assessment 4: Group Project Report

OVERVIEW
--------
This script implements the full data analytics and classification
pipeline on the NT Infrastructure Plan & Pipeline 2022 (NTIPP)
dataset, enriched with ABS Regional Population 2024 data.

Classification Task:
  Predict whether an NT infrastructure project is a SHORT-TERM
  priority — i.e., scheduled for delivery within the 0-5 year
  planning horizon (y=1) or not (y=0).

Models:
  Model 1: Gaussian Naive Bayes   
  Model 2: Support Vector Machine
  Model 3: Random Forest          

============================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

# standard library
import os
import warnings
from statistics import mean   # used by lecturer in svm_std_kfold.py

warnings.filterwarnings("ignore")

# data & science
import numpy  as np
import pandas as pd
from scipy.stats import skew, norm, mannwhitneyu

# visualisation
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on all OS
import matplotlib.pyplot    as plt
import matplotlib.patches   as mpatches
import seaborn              as sns

# scikit-learn — exact imports matching lecture sample codes
from sklearn                import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.pipeline        import make_pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.naive_bayes     import GaussianNB
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier, IsolationForest
from sklearn.metrics         import (
    confusion_matrix, ConfusionMatrixDisplay,   # svm_cm.py pattern
    RocCurveDisplay, PrecisionRecallDisplay,    # svm_roc_prc.py pattern
    classification_report                       # svm_gridsearch_wine.py pattern
)


# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS
# Update DATA_DIR to match your machine if running locally.
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = r"C:\Users\iamas\Downloads\G3 Presentation"
NTIPP_FILE = os.path.join(DATA_DIR, "ntipp-2022-powerbi-xls.XLSX")
ABS_FILE   = os.path.join(DATA_DIR, "ABS Population Data.xlsx")
FIG_DIR    = os.path.join(DATA_DIR, "outputs", "figures")
RPT_DIR    = os.path.join(DATA_DIR, "outputs", "reports")

# create output folders automatically if they do not exist
for folder in [FIG_DIR, RPT_DIR]:
    os.makedirs(folder, exist_ok=True)

# helper: full path to a figure file
def fig(name):  return os.path.join(FIG_DIR, name)
def rpt(name):  return os.path.join(RPT_DIR, name)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_STATE = 42   # fixed seed → reproducible results
np.random.seed(RANDOM_STATE)

# ABS SA3 → NTIPP region mapping
# Territory Wide excluded from geographic population comparison
# because it covers the entire NT and has no distinct SA3 area
REGION_TO_SA3 = {
    "Greater Darwin":   ["Darwin City", "Darwin Suburbs",
                          "Litchfield", "Palmerston"],
    "Central Australia": ["Alice Springs"],
    "Barkly":           ["Barkly"],
    "East Arnhem":      ["East Arnhem"],
    "Big Rivers":       ["Katherine"],
    "Top End Rural":    ["Daly - Tiwi - West Arnhem"],
}

# Approximate land areas (km²) for population density calculation
REGION_AREA_KM2 = {
    "Greater Darwin":   3_213,
    "Central Australia": 626_000,
    "Barkly":           320_000,
    "East Arnhem":       97_000,
    "Big Rivers":       320_000,
    "Top End Rural":    549_000,
    "Territory Wide":   500_000,
}

# Planning horizon columns in the NTIPP dataset
HORIZON_COLS = ["2022-23", "2023-24", "0-5", "5-10",
                "0-10",    "0-15",    "10-15", "15+"]

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (consistent across all figures)
# ─────────────────────────────────────────────────────────────────────────────

C_BLUE   = "#1565C0"     # short-term class / SVM
C_GREY   = "#607D8B"     # other class / Naive Bayes
C_GREEN  = "#1A6E3C"     # Random Forest / positive
C_DARK   = "#0D2B45"     # titles
C_TEAL   = "#0F8B9D"     # project bars
C_GOLD   = "#E7A11A"     # population bars
C_RED    = "#C62828"     # reference lines / outliers
C_GRID   = "#E8EEF4"     # gridlines

# region and sector colour sets
REG_COLS = ["#0F8B9D", "#2F7DE1", "#2DB55D", "#E7A11A",
            "#C53334", "#6F3A8A", "#2E6B2E"]
SEC_COLS = ["#0F8B9D", "#2F7DE1", "#2DB55D", "#E7A11A", "#C53334",
            "#6F3A8A", "#2E6B2E", "#8B5CF6", "#EC4899", "#14B8A6",
            "#F59E0B", "#64748B", "#0EA5E9", "#A3E635", "#F97316"]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def clean_axes(ax):
    """
    Remove top and right spines — matches lecture figure style.
    Makes charts cleaner and more professional.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B0BEC5")
    ax.spines["bottom"].set_color("#B0BEC5")
    ax.tick_params(colors="#455A64", labelsize=10)


def save_fig(name):
    """Save current figure at 300 DPI to FIG_DIR."""
    plt.savefig(fig(name), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Saved: {name}")


def gaussian_overlay(ax, data, bins_array):
    """
    Overlay a fitted normal distribution curve on a histogram.
    Replicates the approach from the Assessment 3 code style.
    Helps visualise how close the distribution is to normal.
    """
    mu = np.mean(data)
    sd = np.std(data, ddof=1)
    if sd <= 0:
        return
    x      = np.linspace(data.min(), data.max(), 400)
    bin_w  = bins_array[1] - bins_array[0]
    y_fit  = norm.pdf(x, mu, sd) * len(data) * bin_w
    ax.plot(x, y_fit, color="#37474F", linewidth=2.2,
            label="Normal fit", zorder=6)


def get_iqr_bounds(data_col):
    """
    Calculate IQR method bounds.
    Directly replicates the lecture's mammals_IQR_outlier.py pattern.

    IQR = Q3 - Q1
    Lower bound = Q1 - 1.5 * IQR
    Upper bound = Q3 + 1.5 * IQR

    Returns:
        lower_bound (float), upper_bound (float)
    """
    Q1, Q3 = np.percentile(data_col, [25, 75])
    IQR    = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    return lower_bound, upper_bound


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# SECTION A: DATA LOADING AND PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("PRT564 Assessment 4 — Classification Pipeline")
print("Group 3 | Charles Darwin University | 2026")
print("=" * 60)
print("\n[A] Loading and preprocessing datasets...")


# A1. Load NTIPP dataset
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_excel(NTIPP_FILE, sheet_name="PBIData")
print(f"    NTIPP dataset: {df.shape[0]} rows x {df.shape[1]} columns")


# A2. Parse the 'Est Cost $M' column
# ─────────────────────────────────────────────────────────────────────────────
# The cost column contains a mix of numeric values and bullet symbols (●).
# Bullet symbols represent projects where the NT Government chose not to
# disclose the estimated budget.
# pd.to_numeric with errors='coerce' safely converts '●' to NaN.

df["cost_num"] = pd.to_numeric(df["Est Cost $M"], errors="coerce")

n_missing_cost = df["cost_num"].isna().sum()
cost_median    = df["cost_num"].median()

print(f"    Missing cost values: {n_missing_cost} ({n_missing_cost/888*100:.1f}%)")
print(f"    Imputation method: Median = ${cost_median:.2f}M")
print(f"    (Median chosen over mean because raw cost skewness = 7.24)")

# Apply median imputation — robust to the extreme right skew (gamma_1 = 7.24)
df["cost_imputed"] = df["cost_num"].fillna(cost_median)
df["cost_imputed"] = df["cost_imputed"].clip(lower=0.1)  # avoid log(0)


# A3. Log10 transformation of cost
# ─────────────────────────────────────────────────────────────────────────────
# Skewness of raw cost = 7.24 (strongly right-skewed)
# After log10 transformation, skewness = 0.55 (approximately symmetric)
# This transformation was validated in Assessment 3 using the Shapiro-Wilk test.

df["log_cost"] = np.log10(df["cost_imputed"])

raw_skewness = float(skew(df["cost_imputed"]))
log_skewness = float(skew(df["log_cost"]))

print(f"    Skewness: {raw_skewness:.3f} --> {log_skewness:.3f} (after log10 transform)")


# A4. Parse planning horizon columns → binary flags (0 or 1)
# ─────────────────────────────────────────────────────────────────────────────
# Each horizon column uses '●' to indicate a project falls in that window.
# We convert each to a clean binary integer flag.

for col in HORIZON_COLS:
    if col in df.columns:
        df[f"h_{col}"] = df[col].apply(
            lambda x: 1 if str(x).strip() == "●" else 0
        )


# A5. Construct the classification target variable (y)
# ─────────────────────────────────────────────────────────────────────────────
# y = 1  --> project has 0-5 year horizon (SHORT-TERM priority)
# y = 0  --> project does NOT have 0-5 year horizon (other horizon)
#
# This directly addresses Research Questions 3 and 4 from Assessment 3:
# RQ3: Can Naive Bayes predict short-term funding priority?
# RQ4: Does SVM outperform Naive Bayes for this task?

df["short_term"] = df["h_0-5"].astype(int)

n_pos = df["short_term"].sum()
n_neg = 888 - n_pos

print(f"\n    Classification target: short_term (0-5 year horizon)")
print(f"    Class 1 (short-term):  {n_pos} projects ({n_pos/888*100:.1f}%)")
print(f"    Class 0 (other):       {n_neg} projects ({n_neg/888*100:.1f}%)")
print(f"    Class imbalance ratio: 1:{n_neg//n_pos}")
print(f"    --> class_weight='balanced' required to handle this imbalance")


# A6. Load and integrate ABS population data
# ─────────────────────────────────────────────────────────────────────────────
# ABS Table 3 = Total Persons by SA2.
# We aggregate SA3 areas to match the 7 NTIPP planning regions.
# Territory Wide is excluded from the geographic comparison because it
# represents NT-wide projects rather than a specific geographic area.

abs_raw = pd.read_excel(ABS_FILE, sheet_name="Table 3", header=6)
nt_data = abs_raw[abs_raw["S/T name"] == "Northern Territory"].copy()
sa3_pop = nt_data.groupby("SA3 name")["no..18"].sum()

# build the region population dictionary
region_pop = {}
for region, sa3_list in REGION_TO_SA3.items():
    region_pop[region] = int(sa3_pop.reindex(sa3_list, fill_value=0).sum())

nt_total_pop = sum(region_pop.values())

print(f"\n    ABS Integration: NT total population = {nt_total_pop:,}")
for reg, pop in region_pop.items():
    print(f"      {reg:20s}: {pop:,} ({pop/nt_total_pop*100:.1f}%)")

# merge population into NTIPP on the Region key
region_pop_df = pd.DataFrame(
    list(region_pop.items()), columns=["Region", "ABS_Population"]
)
df = df.merge(region_pop_df, on="Region", how="left")

# Territory Wide: assign NT average population
df["ABS_Population"] = df["ABS_Population"].fillna(nt_total_pop / 6)


# A7. Feature Engineering (new features for Assessment 4)
# ─────────────────────────────────────────────────────────────────────────────
# These 6 new features extend the Assessment 3 feature set.
# Each captures a different analytical dimension of the NTIPP data.

df["region_area_km2"]   = df["Region"].map(REGION_AREA_KM2).fillna(300_000)
df["pop_density"]       = df["ABS_Population"] / df["region_area_km2"]
df["log_pop_density"]   = np.log1p(df["pop_density"])
# Feature: per-capita investment (higher = more expensive relative to local population)
df["cost_per_cap"]      = df["cost_imputed"] / (df["ABS_Population"] / 1_000).clip(0.1)
df["log_cost_per_cap"]  = np.log1p(df["cost_per_cap"])
# Feature: F = Funded project; P = Priority project
df["is_funded"]         = (df["Category"] == "F").astype(int)
# Feature: has an immediate (2022-23 or 2023-24) budget commitment?
df["is_near_term"]      = (
    (df.get("h_2022-23", pd.Series(0, index=df.index)) == 1) |
    (df.get("h_2023-24", pd.Series(0, index=df.index)) == 1)
).astype(int)
# Feature: is the project in a remote/very remote region?
df["is_remote"]         = df["Region"].isin(
    {"Barkly", "East Arnhem", "Top End Rural"}
).astype(int)
# Feature: how many planning windows does this project span?
h_flag_cols             = [f"h_{c}" for c in HORIZON_COLS if f"h_{c}" in df.columns]
df["n_horizons"]        = df[h_flag_cols].sum(axis=1)


# A8. One-hot encoding of categorical variables
# ─────────────────────────────────────────────────────────────────────────────
# drop_first=True avoids the dummy variable trap (perfect multicollinearity).
# This is consistent with rf_build.py: pd.get_dummies(df, drop_first=True)

CAT_COLS   = [c for c in ["Region", "Industry Sector", "Sub Sector",
                           "Enabling Infrastructure", "Category"]
              if c in df.columns]
df_encoded = pd.get_dummies(df, columns=CAT_COLS, drop_first=True, dtype=int)


# A9. Assemble feature matrix X and target y
# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: Horizon columns MUST be excluded from X.
# y is derived from the 0-5 horizon column, so including any horizon
# column in X would create data leakage (the model would see the answer).

CONT_FEATS   = ["log_cost", "log_pop_density", "log_cost_per_cap",
                "is_funded", "is_near_term",  "is_remote", "n_horizons"]

DUMMY_FEATS  = [
    c for c in df_encoded.columns
    if any(c.startswith(p.split()[0]) for p in CAT_COLS)
    and not any(h in c for h in HORIZON_COLS + ["h_", "short_term"])
]

ALL_FEATS    = CONT_FEATS + DUMMY_FEATS
ALL_FEATS    = [f for f in ALL_FEATS if f in df_encoded.columns]

X = df_encoded[ALL_FEATS].astype(float).values
y = df["short_term"].values

# fill any remaining NaN with column median (defensive programming)
for col_idx in range(X.shape[1]):
    nan_mask = np.isnan(X[:, col_idx])
    if nan_mask.any():
        X[nan_mask, col_idx] = np.nanmedian(X[:, col_idx])

print(f"\n    Feature matrix assembled: {X.shape[0]} samples x {X.shape[1]} features")
print(f"    (7 continuous engineered + {X.shape[1]-7} one-hot encoded dummies)")


# A10. Train-test split
# ─────────────────────────────────────────────────────────────────────────────
# 80% training, 20% test. Stratified to preserve the 32.4%/67.6% class ratio.
# This is consistent with all lecture sample codes using train_test_split.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"    Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

# save preprocessed data for reference
df.to_csv(rpt("ntipp_preprocessed.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# SECTION B: EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

print("\n[B] Generating EDA figures...")


# ── B1. Figure 1: Cost Distribution — Before vs After Log10 Transform ─────────
# Pattern: side-by-side histograms with Gaussian overlay, mean/median/mode lines
# Matches the Assessment 3 code approach.

fig_b1, axes = plt.subplots(1, 2, figsize=(16, 6.5), facecolor="white")
fig_b1.subplots_adjust(left=0.07, right=0.97, top=0.83, bottom=0.18, wspace=0.22)

panels = [
    (df["cost_imputed"], "Before Transformation — Raw Cost ($M)",
     "Estimated Cost ($M)", C_BLUE, raw_skewness,
     "Skewness formula:  skew = E[(X - mean)^3] / std^3"),
    (df["log_cost"], "After Transformation — log10(Cost $M)",
     "log10(Estimated Cost $M)", C_GREEN, log_skewness,
     "Y = log10(X)  |  Same transformation used in Assessment 3")
]

for ax, (data, title, xlabel, color, sk_val, formula) in zip(axes, panels):
    ax.set_facecolor("#F8FAFC")
    _, bins, _ = ax.hist(data, bins=14, color=color, edgecolor="black",
                         linewidth=0.8, alpha=0.87, zorder=3)
    gaussian_overlay(ax, data, bins)

    mu_v = np.mean(data)
    md_v = np.median(data)
    mo_v = (bins[np.argmax(np.histogram(data, 14)[0])] +
            bins[np.argmax(np.histogram(data, 14)[0]) + 1]) / 2

    ax.axvline(mu_v, color=C_RED,     lw=1.8, ls="--", label=f"Mean = {mu_v:.2f}")
    ax.axvline(md_v, color=C_GOLD,    lw=1.8, ls=":",  label=f"Median = {md_v:.2f}")
    ax.axvline(mo_v, color="#7B1FA2", lw=1.8, ls="-.", label=f"Mode = {mo_v:.2f}")

    stats_box = (f"n = {len(data)}\n"
                 f"Mean     = {mu_v:.2f}\n"
                 f"Median   = {md_v:.2f}\n"
                 f"Skewness = {sk_val:+.3f}")
    ax.text(0.97, 0.97, stats_box, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.40", facecolor="#FFF7E6",
                      edgecolor=C_GOLD, linewidth=1.1, alpha=0.97))
    ax.text(0.03, 0.97, formula, transform=ax.transAxes,
            ha="left", va="top", fontsize=8.5, style="italic", color=C_GREEN,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#E6F4EB",
                      edgecolor=C_GREEN, linewidth=0.9, alpha=0.92))

    ax.legend(fontsize=9, frameon=False)
    clean_axes(ax)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10, color=C_DARK)
    ax.set_xlabel(xlabel, fontsize=11, color=C_DARK)
    ax.set_ylabel("Frequency", fontsize=11, color=C_DARK)

fig_b1.suptitle("Figure 1  —  Log10 Transformation: NT Infrastructure Project Cost",
                fontsize=15, fontweight="bold", color=C_DARK, y=0.96)
fig_b1.text(
    0.5, 0.06,
    f"Skewness reduced: {raw_skewness:.2f}  -->  {log_skewness:.2f}  |  "
    f"Median imputation: {n_missing_cost} missing values  -->  ${cost_median:.2f}M  |  "
    "Distribution approaches normality after transformation",
    ha="center", fontsize=9.5, color=C_DARK,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#EEF6FF",
              edgecolor=C_BLUE, linewidth=1)
)
save_fig("fig01_cost_distribution.png")


# ── B2. Figure 2: Classification Target Distribution ──────────────────────────

fig_b2, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")

labels = ["Not Short-Term\n(Other horizons)", "Short-Term Priority\n(0-5 yr)"]
counts = [n_neg, n_pos]
colors = [C_GREY, C_BLUE]

bars = axes[0].bar(labels, counts, color=colors, edgecolor="white",
                   width=0.45, zorder=3)
for bar, cnt in zip(bars, counts):
    pct = cnt / 888 * 100
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 7,
                 f"{cnt}\n({pct:.1f}%)",
                 ha="center", va="bottom", fontsize=12,
                 fontweight="bold", color=C_DARK)
axes[0].set_ylim(0, max(counts) * 1.22)
axes[0].set_ylabel("Number of Projects", fontsize=11)
axes[0].set_title("Class Count Distribution", fontsize=12,
                  fontweight="bold", color=C_DARK)
clean_axes(axes[0])
axes[0].yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
axes[0].set_axisbelow(True)

axes[1].pie(counts, labels=labels, autopct="%1.1f%%",
            colors=colors, startangle=80,
            textprops={"fontsize": 11},
            wedgeprops={"edgecolor": "white", "linewidth": 2.5})
axes[1].set_title("Class Proportion", fontsize=12,
                  fontweight="bold", color=C_DARK)

fig_b2.suptitle(
    f"Figure 2  —  Classification Target: Short-Term Priority  (n = {888})\n"
    f"y = 1  if project horizon = 0-5 year,   y = 0  otherwise"
    f"   |   Class ratio 1:{n_neg // n_pos}  -->  class_weight='balanced' required",
    fontsize=11, fontweight="bold", color=C_DARK, y=1.02
)
plt.tight_layout()
save_fig("fig02_class_distribution.png")


# ── B3. Figure 3: Short-Term Priority Rate by Region ─────────────────────────

reg_grp = df.groupby("Region")["short_term"].agg(["sum", "count"])
reg_grp["rate"] = (reg_grp["sum"] / reg_grp["count"] * 100).round(1)
reg_grp = reg_grp.sort_values("rate", ascending=True)
overall_rate = df["short_term"].mean() * 100

fig_b3, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
bar_colors = [REG_COLS[i % len(REG_COLS)] for i in range(len(reg_grp))]
bars = ax.barh(reg_grp.index, reg_grp["rate"], color=bar_colors,
               edgecolor="white", height=0.52, zorder=3)

for bar, (_, row) in zip(bars, reg_grp.iterrows()):
    ax.text(bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{row['rate']:.1f}%   (n = {int(row['count'])})",
            va="center", fontsize=10, color=C_DARK)

ax.axvline(overall_rate, color=C_RED, linestyle="--", linewidth=1.8,
           zorder=5, label=f"Overall = {overall_rate:.1f}%")
ax.set_xlabel("Short-Term Priority Rate (%)", fontsize=11, color=C_DARK)
ax.set_title("Figure 3  —  Short-Term Priority Rate by NT Planning Region",
             fontsize=13, fontweight="bold", color=C_DARK, pad=10)
ax.set_xlim(0, reg_grp["rate"].max() + 18)
ax.legend(fontsize=10, frameon=False)
clean_axes(ax)
ax.xaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

top_region = reg_grp["rate"].idxmax()
plt.tight_layout()
save_fig("fig03_region_priority_rate.png")


# ── B4. Figure 4: Sector Analysis (HORIZONTAL bars — fixes label overlap) ─────

sec_grp = df.groupby("Industry Sector")["short_term"].agg(["sum", "count"])
sec_grp["rate"] = (sec_grp["sum"] / sec_grp["count"] * 100).round(1)
sec_grp = sec_grp[sec_grp["count"] >= 5].sort_values("rate", ascending=True)

fig_b4, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor="white")
fig_b4.subplots_adjust(wspace=0.38, left=0.04, right=0.97)

# LEFT: short-term priority rate — horizontal bars eliminate label overlap
s_colors = [SEC_COLS[i % len(SEC_COLS)] for i in range(len(sec_grp))]
bars = axes[0].barh(sec_grp.index, sec_grp["rate"], color=s_colors,
                    edgecolor="white", height=0.55, zorder=3)
for bar, v in zip(bars, sec_grp["rate"]):
    axes[0].text(bar.get_width() + 0.4,
                 bar.get_y() + bar.get_height() / 2,
                 f"{v:.1f}%", va="center", fontsize=9.5, color=C_DARK)
axes[0].axvline(overall_rate, color=C_RED, linestyle="--", linewidth=1.5,
                label=f"Overall = {overall_rate:.1f}%")
axes[0].set_xlabel("Short-Term Priority Rate (%)", fontsize=11)
axes[0].set_title("Short-Term Priority Rate\nby Industry Sector",
                  fontsize=12, fontweight="bold", color=C_DARK)
axes[0].set_xlim(0, sec_grp["rate"].max() + 18)
axes[0].legend(fontsize=9, frameon=False)
clean_axes(axes[0])
axes[0].xaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
axes[0].set_axisbelow(True)

# RIGHT: project count — horizontal bars
sec_cnt = sec_grp.sort_values("count", ascending=True)
c_cols  = [SEC_COLS[list(sec_grp.index).index(i) % len(SEC_COLS)]
           for i in sec_cnt.index]
bars2 = axes[1].barh(sec_cnt.index, sec_cnt["count"], color=c_cols,
                     edgecolor="white", height=0.55, zorder=3)
for bar, v in zip(bars2, sec_cnt["count"]):
    axes[1].text(bar.get_width() + 0.8,
                 bar.get_y() + bar.get_height() / 2,
                 str(int(v)), va="center", fontsize=9.5, color=C_DARK)
axes[1].set_xlabel("Number of Projects", fontsize=11)
axes[1].set_title("Project Count\nby Industry Sector",
                  fontsize=12, fontweight="bold", color=C_DARK)
clean_axes(axes[1])
axes[1].xaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
axes[1].set_axisbelow(True)

fig_b4.suptitle(
    "Figure 4  —  EDA: Industry Sector — Priority Rates and Project Distribution",
    fontsize=13, fontweight="bold", color=C_DARK, y=1.01
)
save_fig("fig04_sector_analysis.png")


# ── B5. Figure 5: Cost by Class (Boxplot + Histogram, Mann-Whitney U test) ───

d0      = df.loc[df["short_term"] == 0, "log_cost"].dropna()
d1      = df.loc[df["short_term"] == 1, "log_cost"].dropna()
u_stat, p_val = mannwhitneyu(d0, d1, alternative="two-sided")

fig_b5, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
fig_b5.subplots_adjust(wspace=0.30)

# boxplot
bp = axes[0].boxplot(
    [d0, d1], patch_artist=True, widths=0.45,
    labels=["Class 0\n(Other)", "Class 1\n(0-5 yr)"],
    medianprops=dict(color="white", linewidth=2.5),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2)
)
for patch, color in zip(bp["boxes"], [C_GREY, C_BLUE]):
    patch.set_facecolor(color)
for flier in bp["fliers"]:
    flier.set(marker="o", markerfacecolor="#888888", markersize=4, alpha=0.5)

axes[0].set_ylabel("log10(Estimated Cost $M)", fontsize=11)
axes[0].set_title("log_cost by Priority Class", fontsize=12,
                  fontweight="bold", color=C_DARK)
clean_axes(axes[0])
axes[0].yaxis.grid(True, color=C_GRID, linewidth=0.8)

sig_text  = "Statistically significant (p < 0.05)" if p_val < 0.05 else "Not significant (p >= 0.05)"
sig_color = C_GREEN if p_val < 0.05 else C_RED
sig_fc    = "#E6F4EB" if p_val < 0.05 else "#FCE8E8"
axes[0].text(
    0.5, 0.04,
    f"Mann-Whitney U = {u_stat:.0f},  p = {p_val:.4f}\n{sig_text}",
    transform=axes[0].transAxes, ha="center", fontsize=9.5,
    color=sig_color,
    bbox=dict(boxstyle="round,pad=0.35", facecolor=sig_fc,
              edgecolor=sig_color, linewidth=1)
)

# overlapping histograms
for data, color, label, alpha in [
    (d0, C_GREY, "Class 0 (Other)", 0.60),
    (d1, C_BLUE, "Class 1 (0-5yr)", 0.75)
]:
    axes[1].hist(data, bins=14, color=color, alpha=alpha,
                 edgecolor="white", label=label, zorder=3)
axes[1].axvline(d0.median(), color=C_GREY, linewidth=2, linestyle="--",
                label=f"Median C0 = {d0.median():.2f}")
axes[1].axvline(d1.median(), color=C_BLUE, linewidth=2, linestyle="--",
                label=f"Median C1 = {d1.median():.2f}")
axes[1].set_xlabel("log10(Estimated Cost $M)", fontsize=11)
axes[1].set_ylabel("Frequency", fontsize=11)
axes[1].set_title("Distribution Overlap by Class", fontsize=12,
                  fontweight="bold", color=C_DARK)
axes[1].legend(fontsize=8.5, frameon=False)
clean_axes(axes[1])

fig_b5.suptitle("Figure 5  —  log10(Cost) Distribution by Priority Class",
                fontsize=13, fontweight="bold", color=C_DARK, y=1.01)
plt.tight_layout()
save_fig("fig05_cost_by_class.png")


# ── B6. Figure 6: ABS Integration (6 geographic regions only) ────────────────
# Territory Wide excluded — no specific geographic population

geo_regions  = list(REGION_TO_SA3.keys())
proj_by_reg  = df[df["Region"].isin(geo_regions)].groupby("Region").size()
proj_share   = proj_by_reg / proj_by_reg.sum() * 100
pop_vals_geo = pd.Series({r: region_pop[r] for r in geo_regions})
pop_share    = pop_vals_geo / pop_vals_geo.sum() * 100

# align on common index
common = proj_share.index.intersection(pop_share.index)
ps_proj = proj_share[common];  ps_pop = pop_share[common]
x_pos = np.arange(len(common));  bar_w = 0.38

fig_b6, ax = plt.subplots(figsize=(13, 6.2), facecolor="white")
b1 = ax.bar(x_pos - bar_w / 2, ps_proj.values, bar_w,
            label="Project Share (%)", color=C_TEAL, edgecolor="white", zorder=3)
b2 = ax.bar(x_pos + bar_w / 2, ps_pop.values,  bar_w,
            label="ABS Population Share (%)", color=C_GOLD, edgecolor="white", zorder=3)

for bar, v in zip(b1, ps_proj.values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"{v:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_TEAL)
for bar, v in zip(b2, ps_pop.values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"{v:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#8B6000")

ax.set_xticks(x_pos)
ax.set_xticklabels(common, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Share (%)", fontsize=11)
ax.legend(frameon=False, fontsize=10)
ax.set_title(
    "Figure 6  —  Heterogeneous Data Integration: Project Share vs ABS Population Share\n"
    "(6 geographic regions; Territory Wide excluded — it spans all NT)",
    fontsize=12, fontweight="bold", color=C_DARK, pad=10
)
clean_axes(ax)
ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
save_fig("fig06_abs_integration.png")


# ── B7. Figure 7: Data Quality — Missing Value Analysis ───────────────────────

miss_vals = {
    "Est Cost $M":      n_missing_cost / 888 * 100,
    "Industry Sector":  0.0,
    "Sub Sector":       df["Sub Sector"].isna().sum() / 888 * 100,
    "Region":           0.0,
    "Category":         0.0,
    "ABS Population":   0.0,
}
miss_s  = pd.Series(miss_vals)
bar_fc  = ["#C62828" if v > 10 else "#E7A11A" if v > 0 else "#2DB55D"
           for v in miss_s.values]

fig_b7, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")

axes[0].barh(miss_s.index, miss_s.values, color=bar_fc,
             edgecolor="white", height=0.5, zorder=3)
for i, (k, v) in enumerate(miss_s.items()):
    axes[0].text(v + 0.15, i, f"{v:.1f}%", va="center",
                 fontsize=10, fontweight="bold")
axes[0].set_xlabel("Missing Values (%)", fontsize=11)
axes[0].set_title("Missing Values per Variable", fontsize=12,
                  fontweight="bold", color=C_DARK)
patches = [mpatches.Patch(color="#C62828", label="> 10% — Imputation required"),
           mpatches.Patch(color="#E7A11A", label="1–10% — Minor imputation"),
           mpatches.Patch(color="#2DB55D", label="0% — Complete")]
axes[0].legend(handles=patches, fontsize=9, frameon=False)
clean_axes(axes[0])
axes[0].xaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)

comp = [100 - v for v in miss_s.values]
axes[1].bar(miss_s.index, comp, color=bar_fc, edgecolor="white",
            width=0.5, zorder=3)
for i, (k, c) in enumerate(zip(miss_s.index, comp)):
    axes[1].text(i, c + 0.5, f"{c:.0f}%", ha="center", va="bottom",
                 fontsize=9.5, fontweight="bold")
axes[1].set_ylim(0, 110)
axes[1].set_ylabel("Completeness (%)", fontsize=11)
axes[1].set_title("Data Completeness per Variable", fontsize=12,
                  fontweight="bold", color=C_DARK)
axes[1].tick_params(axis="x", rotation=35)
clean_axes(axes[1])
axes[1].yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
axes[1].set_axisbelow(True)

fig_b7.suptitle(
    f"Figure 7  —  Data Quality Assessment  (n = {888} projects)\n"
    f"Median imputation: {n_missing_cost} missing cost values  -->  ${cost_median:.2f}M",
    fontsize=12, fontweight="bold", color=C_DARK, y=1.03
)
plt.tight_layout()
save_fig("fig07_missing_values.png")


# ── B8. Figure 8: Correlation Heatmap ─────────────────────────────────────────

cont_cols = ["log_cost", "log_pop_density", "log_cost_per_cap",
             "is_funded", "is_near_term", "is_remote",
             "n_horizons", "short_term"]
corr_df   = df[cont_cols].copy()
corr_df.columns = ["log_cost", "log_pop_dens", "log_cost_percap",
                   "is_funded", "is_near_term", "is_remote",
                   "n_horizons", "short_term"]
corr      = corr_df.corr()

fig_b8, ax = plt.subplots(figsize=(9, 7), facecolor="white")
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax,
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 10},
            vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8, "label": "Pearson r"})
ax.set_title(
    "Figure 8  —  Pearson Correlation Matrix: Continuous Features\n"
    "r(X,Y) = sum[(xi - x_bar)(yi - y_bar)] / sqrt[sum(xi-x_bar)^2 * sum(yi-y_bar)^2]",
    fontsize=11, fontweight="bold", color=C_DARK, pad=10
)
ax.tick_params(axis="x", rotation=35, labelsize=9.5)
ax.tick_params(axis="y", rotation=0,  labelsize=9.5)
plt.tight_layout()
save_fig("fig08_correlation_heatmap.png")


# ── B9. Figure 9: Category Analysis + Horizon Complexity ─────────────────────

fig_b9, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")

cat_grp = df.groupby(["Category", "short_term"]).size().unstack(fill_value=0)
cat_grp.columns = ["Other", "Short-Term (0-5yr)"]
cat_grp.plot(kind="bar", ax=axes[0], color=[C_GREY, C_BLUE],
             edgecolor="white", width=0.5)
axes[0].set_title("Category vs Short-Term Priority\n(P = Priority  |  F = Funded)",
                  fontsize=11, fontweight="bold", color=C_DARK)
axes[0].set_xlabel("Project Category", fontsize=11)
axes[0].set_ylabel("Number of Projects", fontsize=11)
axes[0].tick_params(axis="x", rotation=0)
axes[0].legend(fontsize=9.5, frameon=False)
clean_axes(axes[0])
axes[0].yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
axes[0].set_axisbelow(True)

hn = df["n_horizons"].value_counts().sort_index()
axes[1].bar(hn.index.astype(str), hn.values, color=C_BLUE,
            edgecolor="white", width=0.6, zorder=3)
for i, v in enumerate(hn.values):
    axes[1].text(i, v + 3, str(v), ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
axes[1].set_xlabel("Number of Planning Horizons per Project", fontsize=11)
axes[1].set_ylabel("Number of Projects", fontsize=11)
axes[1].set_title("Planning Horizon Complexity\n(Number of horizon windows per project)",
                  fontsize=11, fontweight="bold", color=C_DARK)
clean_axes(axes[1])
axes[1].yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
axes[1].set_axisbelow(True)

fig_b9.suptitle("Figure 9  —  Project Category and Planning Horizon Complexity",
                fontsize=13, fontweight="bold", color=C_DARK, y=1.02)
plt.tight_layout()
save_fig("fig09_category_horizon.png")


# ── B10. Figure 10: Outlier Detection using IQR Method (Week 11) ──────────────
# Replicates the IQR pattern from mammals_IQR_outlier.py

lower_b, upper_b = get_iqr_bounds(df["log_cost"])

# mark outliers (matching the lecture's approach: -1 for outlier, 1 for inlier)
df["outlier_flag"] = 1
df.loc[(df["log_cost"] < lower_b) | (df["log_cost"] > upper_b), "outlier_flag"] = -1
outliers    = df[df["outlier_flag"] == -1]
n_outliers  = len(outliers)

print(f"\n    IQR Outlier Detection (Week 11):")
print(f"      Lower bound: {lower_b:.3f}  |  Upper bound: {upper_b:.3f}")
print(f"      Outliers detected: {n_outliers} ({n_outliers/888*100:.1f}%)")

fig_b10, ax = plt.subplots(figsize=(12, 5.5), facecolor="white")

# plot inliers
inliers_df = df[df["outlier_flag"] == 1]
ax.scatter(inliers_df.index, inliers_df["log_cost"],
           c=C_GREEN, s=20, alpha=0.5, label="Inliers", zorder=3)

# plot outliers
ax.scatter(outliers.index, outliers["log_cost"],
           c="red", s=40, edgecolor="darkred", linewidth=0.8,
           label=f"Outliers (n={n_outliers})", zorder=5)

ax.axhline(upper_b, color=C_RED, linestyle="--", linewidth=1.5,
           label=f"IQR Upper bound = {upper_b:.2f}")
ax.axhline(lower_b, color=C_GOLD, linestyle="--", linewidth=1.5,
           label=f"IQR Lower bound = {lower_b:.2f}")

ax.set_xlabel("Project Index", fontsize=11)
ax.set_ylabel("log10(Estimated Cost $M)", fontsize=11)
ax.set_title(
    "Figure 10  —  Outlier Detection: IQR Method on log10(Cost)  (Week 11)\n"
    f"IQR = Q3 - Q1  |  Bounds = Q1 -/+ 1.5*IQR  |  {n_outliers} outliers detected ({n_outliers/888*100:.1f}%)",
    fontsize=12, fontweight="bold", color=C_DARK, pad=10
)
ax.legend(fontsize=9.5, frameon=True, framealpha=0.9)
clean_axes(ax)
ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
plt.tight_layout()
save_fig("fig10_iqr_outlier_detection.png")


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# SECTION C: CLASSIFICATION MODELS
# Follows the EXACT patterns from lecture sample codes:
#   svm_nb_rf.py     → classifiers = [...] loop
#   svm_cm.py        → ConfusionMatrixDisplay
#   svm_roc_prc.py   → RocCurveDisplay.from_estimator
#   svm_std_kfold.py → cross_validate with mean() from statistics
#   svm_gridsearch_wine.py → GridSearchCV with classification_report
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

print("\n[C] Training classification models...")
print("    Pattern: svm_nb_rf.py (classifiers list loop)")


# ── C1. Simple Classifier Comparison (svm_nb_rf.py pattern) ──────────────────
# This is the primary pattern from the lecture's svm_nb_rf.py:
#   classifiers = [('SVM', SVC(...)), ('GaussianNB', GaussianNB()), ...]
#   for name, clf in classifiers:
#       clf.fit(X_train, y_train)
#       y_hat = clf.predict(X_test)
#
# NOTE: y_hat (not y_pred) — this matches the lecture sample code naming.

classifiers = [
    # Model 1: Gaussian Naive Bayes (Week 7) — used as baseline benchmark
    ("Gaussian Naive Bayes",        GaussianNB()),

    # Model 2: SVM with standardisation pipeline (Week 8)
    # make_pipeline ensures StandardScaler is ONLY fitted on training data
    # within each CV fold — prevents data leakage
    ("SVM (Linear kernel)",         make_pipeline(
                                        StandardScaler(),
                                        SVC(kernel="linear",
                                            class_weight="balanced",
                                            random_state=RANDOM_STATE))),

    # Model 3: Random Forest Classifier (Week 10)
    # Does NOT require standardisation — tree splits are scale-invariant
    ("Random Forest Classifier",    RandomForestClassifier(
                                        n_estimators=200,
                                        class_weight="balanced",
                                        random_state=RANDOM_STATE)),
]


print(f"\n    {'Classifier':<30}  {'Accuracy':>9}  {'Precision':>10}  "
      f"{'Recall':>8}  {'F1':>8}  {'AUC':>8}")
print("    " + "-" * 70)

# store predictions for later use in figures
all_y_hat  = {}
all_y_prob = {}
all_clf    = {}

for name, clf in classifiers:

    # A. Train the classifier on the training set
    clf.fit(X_train, y_train)

    # B. Predict classes on the test set
    #    Using y_hat — matching the lecture sample code naming convention
    y_hat = clf.predict(X_test)

    # C. Get probability scores for ROC/AUC
    #    GaussianNB and RandomForest have predict_proba natively
    #    SVC in pipeline also returns proba because probability=True by default
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
    else:
        y_prob = clf.decision_function(X_test)

    # D. Compute metrics (matching svm_nb_rf.py pattern)
    acc  = metrics.accuracy_score(y_test, y_hat) * 100
    prec = metrics.precision_score(y_test, y_hat, zero_division=0)
    recl = metrics.recall_score(y_test, y_hat, zero_division=0)
    f1   = metrics.f1_score(y_test, y_hat, zero_division=0)
    auc  = metrics.roc_auc_score(y_test, y_prob)

    print(f"    {name:<30}  {acc:>8.3f}%  {prec:>10.3f}  "
          f"{recl:>8.3f}  {f1:>8.3f}  {auc:>8.3f}")

    # store for figures
    all_y_hat[name]  = y_hat
    all_y_prob[name] = y_prob
    all_clf[name]    = clf


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# SECTION D: MODEL EVALUATION FIGURES
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

print("\n[D] Generating evaluation figures...")


# ── D1. Figure 11: Confusion Matrices (svm_cm.py pattern) ─────────────────────
# Uses ConfusionMatrixDisplay exactly as taught in the lecture:
#
#   cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
#   disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=...)
#   disp.plot()
#
# For all three classifiers side by side.

class_names = ["Other (y=0)", "Short-Term (y=1)"]

fig_d1, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="white")

for ax, (name, clf) in zip(axes, all_clf.items()):
    y_hat = all_y_hat[name]

    # compute confusion matrix (lecture: svm_cm.py)
    cm   = confusion_matrix(y_test, y_hat)

    # display using ConfusionMatrixDisplay (lecture pattern)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")

    # add TP / TN / FP / FN labels (Week 9 lecture labels)
    for (r, c), lbl in [((0,0),"TN"), ((0,1),"FP"),
                         ((1,0),"FN"), ((1,1),"TP")]:
        ax.text(c + 0.5, r + 0.80, lbl, ha="center", va="center",
                fontsize=8.5, color="gray")

    acc  = metrics.accuracy_score(y_test, y_hat) * 100
    f1   = metrics.f1_score(y_test, y_hat, zero_division=0)
    auc  = metrics.roc_auc_score(y_test, all_y_prob[name])

    ax.set_title(f"{name}\nAcc={acc:.1f}%  F1={f1:.3f}  AUC={auc:.3f}",
                 fontsize=10.5, fontweight="bold", color=C_DARK, pad=8)

fig_d1.suptitle(
    "Figure 11  —  Confusion Matrices: All Three Classifiers  (Test Set)\n"
    "Accuracy = (TP+TN)/(TP+TN+FP+FN)  |  "
    "Precision = TP/(TP+FP)  |  "
    "Recall = TP/(TP+FN)  |  "
    "F1 = 2*P*R/(P+R)",
    fontsize=10.5, fontweight="bold", color=C_DARK, y=1.06
)
plt.tight_layout()
save_fig("fig11_confusion_matrices.png")


# ── D2. Figure 12: ROC Curves (svm_roc_prc.py pattern) ────────────────────────
# The lecture uses:
#   RocCurveDisplay.from_estimator(model, X_test, y_test)
#   plt.show()
#
# We combine all three on one axes for comparison.

clf_colors = {
    "Gaussian Naive Bayes":      "#90CAF9",
    "SVM (Linear kernel)":       "#1976D2",
    "Random Forest Classifier":  C_DARK,
}
clf_lwidths = {
    "Gaussian Naive Bayes":      2.0,
    "SVM (Linear kernel)":       2.5,
    "Random Forest Classifier":  3.0,
}

fig_d2, ax = plt.subplots(figsize=(8, 7), facecolor="white")

for name, clf in all_clf.items():
    auc    = metrics.roc_auc_score(y_test, all_y_prob[name])
    color  = clf_colors[name]
    lw     = clf_lwidths[name]
    label  = f"{name}  (AUC = {auc:.3f})"

    # use RocCurveDisplay.from_estimator — matching svm_roc_prc.py
    RocCurveDisplay.from_estimator(
        clf, X_test, y_test, ax=ax,
        name=label, color=color, lw=lw
    )

ax.plot([0, 1], [0, 1], "k--", linewidth=1.2,
        label="Random Classifier (AUC = 0.500)")
ax.set_xlabel("False Positive Rate  (FPR = FP / (FP + TN))", fontsize=11)
ax.set_ylabel("True Positive Rate  (Recall = TP / (TP + FN))", fontsize=11)
ax.set_title(
    "Figure 12  —  ROC Curves: All Three Classifiers\n"
    "AUC = area under the TPR vs FPR curve  |  Higher AUC = better discrimination",
    fontsize=12, fontweight="bold", color=C_DARK, pad=10
)
ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
clean_axes(ax)
ax.grid(color=C_GRID, linewidth=0.8)
plt.tight_layout()
save_fig("fig12_roc_curves.png")


# ── D3. Figure 13: Precision-Recall Curves (svm_roc_prc.py pattern) ───────────
# The lecture's svm_roc_prc.py also plots:
#   PrecisionRecallDisplay.from_estimator(model, X_test, y_test)

fig_d3, ax = plt.subplots(figsize=(8, 7), facecolor="white")

for name, clf in all_clf.items():
    color = clf_colors[name]
    lw    = clf_lwidths[name]
    PrecisionRecallDisplay.from_estimator(
        clf, X_test, y_test, ax=ax,
        name=name, color=color, lw=lw
    )

ax.set_xlabel("Recall  (TP / (TP + FN))", fontsize=11)
ax.set_ylabel("Precision  (TP / (TP + FP))", fontsize=11)
ax.set_title(
    "Figure 13  —  Precision-Recall Curves: All Three Classifiers",
    fontsize=12, fontweight="bold", color=C_DARK, pad=10
)
ax.legend(loc="upper right", fontsize=10, frameon=True)
clean_axes(ax)
ax.grid(color=C_GRID, linewidth=0.8)
plt.tight_layout()
save_fig("fig13_precision_recall_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# SECTION E: K-FOLD CROSS-VALIDATION (svm_std_kfold.py pattern)
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
# The lecture's svm_std_kfold.py uses:
#   from statistics import mean
#   scoring = ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']
#   pipe = make_pipeline(StandardScaler(), svm.SVC(kernel='linear'))
#   scores = cross_validate(pipe, X, y, cv=10, scoring=scoring)
#   print("Mean accuracy: %.3f%%" % (mean(scores['test_accuracy'])*100))

print("\n[E] 10-fold Cross-Validation (svm_std_kfold.py pattern)...")

# scoring keys matching the lecture's svm_std_kfold.py exactly
scoring = ["accuracy", "recall_macro", "precision_macro", "f1_macro"]

# store CV results for each classifier
cv_results = {}

for name, clf in classifiers:
    print(f"\n    Cross-validating: {name} ...")

    # run 10-fold cross-validation
    scores = cross_validate(clf, X, y, cv=10, scoring=scoring)

    cv_results[name] = scores

    # print using mean() from statistics — matching svm_std_kfold.py exactly
    print(f"      Mean accuracy:  {mean(scores['test_accuracy'])*100:.3f}%")
    print(f"      Mean precision: {mean(scores['test_precision_macro']):.3f}")
    print(f"      Mean recall:    {mean(scores['test_recall_macro']):.3f}")
    print(f"      Mean F1:        {mean(scores['test_f1_macro']):.3f}")


# ── Figure 14: Cross-Validation Boxplot ───────────────────────────────────────

fig_e1, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")

# Accuracy boxplot
acc_data = [cv_results[n]["test_accuracy"] for n, _ in classifiers]
f1_data  = [cv_results[n]["test_f1_macro"] for n, _ in classifiers]
clf_names = [n for n, _ in classifiers]

for ax, data, ylabel, title in [
    (axes[0], acc_data, "Accuracy",  "10-Fold CV: Accuracy"),
    (axes[1], f1_data,  "F1-Score (macro)", "10-Fold CV: F1-Score"),
]:
    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    labels=["Naive\nBayes", "SVM", "Random\nForest"],
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for patch, color in zip(bp["boxes"], ["#90CAF9", "#1976D2", C_DARK]):
        patch.set_facecolor(color)

    # scatter individual fold scores (shows distribution per fold)
    for i, (vals, color) in enumerate(
        zip(data, ["#90CAF9", "#1976D2", C_DARK])
    ):
        ax.scatter([i + 1] * len(vals), vals, color=color,
                   alpha=0.55, s=30, zorder=5)

    # annotate mean +/- std using mean() from statistics (lecture pattern)
    for i, vals in enumerate(data):
        ax.text(i + 1, max(vals) + 0.015,
                f"mean={mean(vals):.3f}\n+/-{np.std(vals):.3f}",
                ha="center", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#f8fafc",
                          edgecolor="#94a3b8", linewidth=0.7))

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", color=C_DARK)
    lo = max(0, min(v.min() for v in data) - 0.06)
    hi = min(1.05, max(v.max() for v in data) + 0.12)
    ax.set_ylim(lo, hi)
    clean_axes(ax)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)

fig_e1.suptitle(
    "Figure 14  —  10-Fold Cross-Validation Performance  (Week 9)\n"
    "CV Score = (1/k) * sum of Score(fold_i)  |  k = 10  |  "
    "Lower std = more stable generalisation",
    fontsize=12, fontweight="bold", color=C_DARK, y=1.04
)
plt.tight_layout()
save_fig("fig14_cv_performance.png")


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# SECTION F: GRIDSEARCHCV HYPERPARAMETER TUNING (svm_gridsearch_wine.py pattern)
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
# The lecture's svm_gridsearch_wine.py uses:
#   tuned_parameters = [{'kernel': ['rbf'], 'gamma': [...], 'C': [...]},
#                       {'kernel': ['linear'], 'C': [...]}]
#   clf = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy')
#   clf.fit(X_train, y_train)
#   print(clf.best_params_)
#   print(classification_report(y_true, y_pred))

print("\n[F] GridSearchCV Hyperparameter Tuning...")
print("    Pattern: svm_gridsearch_wine.py")

scores_gs = ["accuracy", "f1_macro"]

# ── F1. SVM GridSearchCV ──────────────────────────────────────────────────────
# Matching the tuned_parameters format from svm_gridsearch_wine.py

svm_tuned_parameters = [
    {"svc__kernel": ["rbf"],    "svc__gamma": [1e-3, 1e-4],
     "svc__C": [1, 10, 100, 1000]},
    {"svc__kernel": ["linear"], "svc__C": [1, 10, 100, 1000]}
]

for score in scores_gs:
    print(f"\n    [SVM] Tuning hyper-parameters for: {score}")

    svm_gs = GridSearchCV(
        make_pipeline(StandardScaler(),
                      SVC(class_weight="balanced",
                          random_state=RANDOM_STATE)),
        svm_tuned_parameters,
        scoring=score,
        cv=5,
        n_jobs=-1
    )
    svm_gs.fit(X_train, y_train)

    print(f"    Best parameters: {svm_gs.best_params_}")
    print(f"    Best CV {score}: {svm_gs.best_score_:.4f}")

    # classification report (svm_gridsearch_wine.py pattern)
    y_hat_gs = svm_gs.predict(X_test)
    print(f"\n    Classification Report (SVM — {score} tuning):")
    print(classification_report(y_test, y_hat_gs,
                                target_names=["Other (y=0)", "Short-Term (y=1)"],
                                zero_division=0))


# ── F2. Random Forest GridSearchCV ───────────────────────────────────────────
# Matching the rf_paramsearch.py pattern

rf_tuned_parameters = {
    "n_estimators": [100, 250, 500],
    "max_depth":    [5, 10, 20, None],
    "class_weight": ["balanced", None]
}

for score in scores_gs:
    print(f"\n    [RF] Tuning hyper-parameters for: {score}")

    rf_gs = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        rf_tuned_parameters,
        scoring=score,
        cv=5,
        n_jobs=-1,
        verbose=0
    )
    rf_gs.fit(X_train, y_train)

    print(f"    Best parameters: {rf_gs.best_params_}")
    print(f"    Best CV {score}: {rf_gs.best_score_:.4f}")

    y_hat_rf_gs = rf_gs.predict(X_test)
    print(f"\n    Classification Report (RF — {score} tuning):")
    print(classification_report(y_test, y_hat_rf_gs,
                                target_names=["Other (y=0)", "Short-Term (y=1)"],
                                zero_division=0))

# use the best RF (f1 tuned) for feature importance
rf_best_model = rf_gs.best_estimator_


# ── Figure 15: Hyperparameter Tuning Curves ───────────────────────────────────

fig_f1, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")

# SVM: C vs F1 for linear and rbf kernels
C_range  = [0.01, 0.1, 1, 10, 100, 1000]
f1_lin   = []; f1_rbf = []

for C_val in C_range:
    for lst, kern in [(f1_lin, "linear"), (f1_rbf, "rbf")]:
        pipe_tmp = make_pipeline(
            StandardScaler(),
            SVC(C=C_val, kernel=kern, class_weight="balanced",
                random_state=RANDOM_STATE)
        )
        pipe_tmp.fit(X_train, y_train)
        y_tmp = pipe_tmp.predict(X_test)
        lst.append(metrics.f1_score(y_test, y_tmp, zero_division=0))

best_C_val = svm_gs.best_params_.get("svc__C", 1)

axes[0].plot(C_range, f1_lin, "o-", color="#1976D2", lw=2.2, ms=8,
             label="Linear kernel")
axes[0].plot(C_range, f1_rbf, "s-", color=C_DARK,   lw=2.2, ms=8,
             label="RBF kernel")
axes[0].axvline(best_C_val, color=C_RED, linestyle="--", linewidth=1.5,
                label=f"Best C = {best_C_val}")
for xv, yl, yr in zip(C_range, f1_lin, f1_rbf):
    axes[0].text(xv, yl + 0.006, f"{yl:.3f}", ha="center",
                 va="bottom", fontsize=7.5, color="#1976D2")
axes[0].set_xscale("log")
axes[0].set_xlabel("Regularisation C  (log scale)", fontsize=11)
axes[0].set_ylabel("F1-Score (test)", fontsize=11)
axes[0].set_title("SVM: C vs F1-Score\nLinear vs RBF Kernel",
                  fontsize=12, fontweight="bold", color=C_DARK)
axes[0].legend(frameon=False, fontsize=9.5)
clean_axes(axes[0])
axes[0].grid(color=C_GRID, linewidth=0.8)

# RF: n_estimators vs F1
ne_range = [50, 100, 200, 300, 500]
f1_rf_ne = []
best_depth  = rf_gs.best_params_.get("max_depth", 10)
best_cw     = rf_gs.best_params_.get("class_weight", "balanced")
best_ne     = rf_gs.best_params_.get("n_estimators", 200)

for ne in ne_range:
    rf_tmp = RandomForestClassifier(
        n_estimators=ne, max_depth=best_depth,
        class_weight=best_cw, random_state=RANDOM_STATE
    )
    rf_tmp.fit(X_train, y_train)
    y_tmp = rf_tmp.predict(X_test)
    f1_rf_ne.append(metrics.f1_score(y_test, y_tmp, zero_division=0))

axes[1].plot(ne_range, f1_rf_ne, "D-", color=C_GREEN, lw=2.2, ms=8,
             label=f"max_depth = {best_depth}")
axes[1].axvline(best_ne, color=C_RED, linestyle="--", linewidth=1.5,
                label=f"Best n = {best_ne}")
for xn, yf in zip(ne_range, f1_rf_ne):
    axes[1].text(xn, yf + 0.004, f"{yf:.3f}", ha="center",
                 va="bottom", fontsize=8, color=C_GREEN)
axes[1].set_xlabel("Number of Trees (n_estimators)", fontsize=11)
axes[1].set_ylabel("F1-Score (test)", fontsize=11)
axes[1].set_title("Random Forest: n_estimators vs F1-Score",
                  fontsize=12, fontweight="bold", color=C_DARK)
axes[1].legend(frameon=False, fontsize=9.5)
clean_axes(axes[1])
axes[1].grid(color=C_GRID, linewidth=0.8)

fig_f1.suptitle("Figure 15  —  Hyperparameter Tuning: SVM and Random Forest",
                fontsize=13, fontweight="bold", color=C_DARK, y=1.02)
plt.tight_layout()
save_fig("fig15_hyperparameter_tuning.png")


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# SECTION G: FEATURE IMPORTANCE (Week 10 — Random Forest)
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

print("\n[G] Feature Importance (Week 10 — Random Forest)...")

# The Random Forest best estimator gives us feature importances
# This tells us WHICH features are most predictive of short-term priority
imps        = rf_best_model.feature_importances_
top_n       = 15
top_idx     = np.argsort(imps)[::-1][:top_n]
top_feats   = [ALL_FEATS[i] for i in top_idx]
top_imps    = imps[top_idx]

print(f"\n    Top {top_n} features (Random Forest Gini Importance):")
for i, (feat, imp) in enumerate(zip(top_feats, top_imps)):
    print(f"      {i+1:2d}. {feat:<45s} {imp:.4f}")

# clean labels for display (no LaTeX, no special chars)
def clean_feat_label(name):
    return (name
            .replace("Industry_Sector_", "Sector: ")
            .replace("Region_",          "Region: ")
            .replace("Sub_Sector_",      "SubSec: ")
            .replace("Enabling_Infrastructure_", "Infra: ")
            .replace("_", " "))

def feat_color(name):
    if name in ["log_cost", "log_cost_per_cap", "n_horizons"]: return C_DARK
    if "pop" in name.lower() or "density" in name.lower(): return C_GREEN
    if name.startswith("is_"): return "#6F3A8A"
    if "Sector" in name or "Industry" in name: return "#2F7DE1"
    if "Region" in name: return C_GOLD
    return C_GREY

clean_labels = [clean_feat_label(f) for f in top_feats]
f_colors     = [feat_color(f) for f in top_feats]

fig_g1, ax = plt.subplots(figsize=(11, 7), facecolor="white")
bars = ax.barh(range(len(top_feats)), top_imps[::-1],
               color=f_colors[::-1], edgecolor="white", height=0.6, zorder=3)
ax.set_yticks(range(len(top_feats)))
ax.set_yticklabels(clean_labels[::-1], fontsize=10)

for bar, v in zip(bars, top_imps[::-1]):
    ax.text(bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}", va="center", fontsize=9)

ax.set_xlabel("Feature Importance  (Mean Decrease in Gini Impurity)", fontsize=11)
ax.set_title(
    "Figure 16  —  Random Forest Feature Importances  (Week 10)\n"
    "Importance(feature j) = (1/T) * sum over trees * sum over nodes "
    "[ p(node) * Gini_decrease(j) ]",
    fontsize=11, fontweight="bold", color=C_DARK, pad=10
)
legend_els = [
    mpatches.Patch(facecolor=C_DARK,    label="Engineered cost features"),
    mpatches.Patch(facecolor=C_GREEN,   label="ABS population features"),
    mpatches.Patch(facecolor="#6F3A8A", label="Binary flag features"),
    mpatches.Patch(facecolor="#2F7DE1", label="Sector encoding"),
    mpatches.Patch(facecolor=C_GOLD,    label="Region encoding"),
]
ax.legend(handles=legend_els, loc="lower right", fontsize=9, frameon=True)
clean_axes(ax)
ax.xaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
save_fig("fig16_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# SECTION H: FINAL COMPARATIVE SUMMARY FIGURE
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

print("\n[H] Final comparative performance figure...")

metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
clf_labels   = ["Naive Bayes", "SVM", "Random Forest"]
bar_colors   = ["#90CAF9", "#1976D2", C_DARK]

# recompute final metrics from stored predictions
final_metrics = {}
for name, _ in classifiers:
    y_hat  = all_y_hat[name]
    y_prob = all_y_prob[name]
    final_metrics[name] = [
        metrics.accuracy_score(y_test, y_hat),
        metrics.precision_score(y_test, y_hat, zero_division=0),
        metrics.recall_score(y_test, y_hat, zero_division=0),
        metrics.f1_score(y_test, y_hat, zero_division=0),
        metrics.roc_auc_score(y_test, y_prob),
    ]

x_pos = np.arange(len(metric_names))
bar_w = 0.25

fig_h1, ax = plt.subplots(figsize=(13, 6), facecolor="white")

for i, (name, _) in enumerate(classifiers):
    vals = final_metrics[name]
    bars = ax.bar(x_pos + (i - 1) * bar_w, vals, bar_w,
                  label=name, color=bar_colors[i],
                  edgecolor="white", zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=7.5, color=C_DARK)

ax.set_xticks(x_pos)
ax.set_xticklabels(metric_names, fontsize=11)
ax.set_ylim(0.3, 1.10)
ax.set_ylabel("Score", fontsize=11)
ax.set_title(
    "Figure 17  —  Comparative Model Performance  (Held-Out Test Set)\n"
    "All metrics at default decision threshold = 0.5",
    fontsize=12, fontweight="bold", color=C_DARK, pad=10
)
ax.legend(frameon=False, fontsize=10.5)
clean_axes(ax)
ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
save_fig("fig17_model_comparison.png")


# ── Figure 18: Analytics Pipeline Flowchart ────────────────────────────────────

fig_h2, ax = plt.subplots(figsize=(17, 5.8), facecolor="white")
ax.set_xlim(0, 17); ax.set_ylim(0, 5.5); ax.axis("off")

stages = [
    ("01\nBusiness\nUnderstanding",  "Define RQs\nShort-term\ntarget",         0.9,  2.8, C_DARK,    "#BFD7ED"),
    ("02\nData\nUnderstanding",      "888 records\n18 variables\nNulls + skew", 2.9,  2.8, "#1A5276", "#D4E6F1"),
    ("03\nData\nPreparation",        "Median impute\nlog10 cost\n147 missing",  4.9,  2.8, C_GREEN,   "#D5F5E3"),
    ("04\nABS\nIntegration",         "SA3 to Region\npop density\nAPA 2024",    6.9,  2.8, "#7D6608", "#FDEBD0"),
    ("05\nFeature\nEngineering",     "7 new features\nis_remote\nn_horizons",   8.9,  2.8, "#6E2F8E", "#F4ECF7"),
    ("06\nEDA",                      "10 figures\nPatterns &\ninsights",        10.9, 2.8, C_TEAL,    "#D0ECE7"),
    ("07\nClassification\nModels",   "NB  (Wk7)\nSVM (Wk8)\nRF  (Wk10)",       12.9, 2.8, "#154360", "#D6EAF8"),
    ("08\nEvaluation\n& Insights",   "Acc/F1/AUC\n10-fold CV\nPolicy recs",     14.9, 2.8, "#922B21", "#FADBD8"),
]

for lbl, detail, cx, cy, bd, bl in stages:
    rect = mpatches.FancyBboxPatch((cx-0.82, cy-1.15), 1.64, 2.3,
                                    boxstyle="round,pad=0.08",
                                    facecolor=bl, edgecolor=bd, linewidth=2)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.62, lbl, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.18", facecolor=bd, edgecolor="none"))
    ax.text(cx, cy - 0.23, detail, ha="center", va="center",
            fontsize=7, color="#16324F", linespacing=1.5)

for i in range(len(stages) - 1):
    x0 = stages[i][2] + 0.82; x1 = stages[i+1][2] - 0.82; cy = stages[i][3]
    ax.annotate("", xy=(x1, cy), xytext=(x0, cy),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.8))

tools_txt = ("Python 3  |  pandas  |  numpy  |  scikit-learn  |  "
             "matplotlib  |  seaborn  |  scipy  |  "
             "NTIPP 2022 (888 records) + ABS Population 2024")
ax.text(8.5, 0.55, tools_txt, ha="center", va="center", fontsize=8.5,
        color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#EEF6FF",
                  edgecolor=C_BLUE, linewidth=1.1))
ax.set_title(
    "Figure 18  —  PRT564 Assessment 4: Analytics and Classification Pipeline",
    fontsize=13, fontweight="bold", color=C_DARK, y=0.97, pad=8
)
save_fig("fig18_analytics_pipeline.png")


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# SECTION I: SAVE REPORTS
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

print("\n[I] Saving reports...")

# detailed classification reports (svm_gridsearch_wine.py pattern)
clf_short_names = {"Gaussian Naive Bayes": "nb",
                   "SVM (Linear kernel)":  "svm",
                   "Random Forest Classifier": "rf"}

for name, _ in classifiers:
    y_hat = all_y_hat[name]
    short = clf_short_names[name]
    rpt_txt = classification_report(
        y_test, y_hat,
        target_names=["Other (y=0)", "Short-Term (y=1)"],
        zero_division=0
    )
    with open(rpt(f"classification_report_{short}.txt"), "w") as fp:
        fp.write(f"Classification Report: {name}\n")
        fp.write("=" * 50 + "\n")
        fp.write(rpt_txt)

# performance summary CSV
rows = []
for name, _ in classifiers:
    y_hat  = all_y_hat[name]
    y_prob = all_y_prob[name]
    rows.append({
        "Model":         name,
        "Accuracy_%":    round(metrics.accuracy_score(y_test, y_hat) * 100, 3),
        "Precision":     round(metrics.precision_score(y_test, y_hat, zero_division=0), 4),
        "Recall":        round(metrics.recall_score(y_test, y_hat, zero_division=0), 4),
        "F1":            round(metrics.f1_score(y_test, y_hat, zero_division=0), 4),
        "ROC_AUC":       round(metrics.roc_auc_score(y_test, y_prob), 4),
        "CV_Acc_Mean_%": round(mean(cv_results[name]["test_accuracy"]) * 100, 3),
        "CV_F1_Mean":    round(mean(cv_results[name]["test_f1_macro"]), 4),
    })
pd.DataFrame(rows).to_csv(rpt("model_performance_summary.csv"), index=False)

# feature importance CSV (RF)
fi_df = pd.DataFrame({
    "Feature":    ALL_FEATS,
    "Importance": rf_best_model.feature_importances_
}).sort_values("Importance", ascending=False)
fi_df.to_csv(rpt("feature_importance_rf.csv"), index=False)

print("    Saved: classification_report_nb/svm/rf.txt")
print("    Saved: model_performance_summary.csv")
print("    Saved: feature_importance_rf.csv")
print("    Saved: ntipp_preprocessed.csv")


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY PRINTOUT
# Copy these numbers directly into the report
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("FINAL SUMMARY — COPY THESE INTO YOUR REPORT")
print("=" * 60)

print(f"\nDATASET")
print(f"  Total projects        : {888}")
print(f"  Missing cost values   : {n_missing_cost} ({n_missing_cost/888*100:.1f}%)  -->  imputed ${cost_median:.2f}M")
print(f"  Cost skewness         : {raw_skewness:.4f}  -->  {log_skewness:.4f}  (after log10)")
print(f"  Short-term (y=1)      : {n_pos} ({n_pos/888*100:.1f}%)")
print(f"  Other (y=0)           : {n_neg} ({n_neg/888*100:.1f}%)")
print(f"  Feature matrix        : {X.shape[0]} x {X.shape[1]}")
print(f"  Train / Test          : {len(X_train)} / {len(X_test)}  (80/20 stratified)")
print(f"  IQR outliers (log_cost): {n_outliers} ({n_outliers/888*100:.1f}%)")

print(f"\nMODEL PERFORMANCE  (test set, n={len(X_test)})")
print(f"  {'Model':<30}  {'Acc%':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'AUC':>7}")
print("  " + "-" * 65)
for name, _ in classifiers:
    y_hat  = all_y_hat[name]
    y_prob = all_y_prob[name]
    print(f"  {name:<30}  "
          f"{metrics.accuracy_score(y_test,y_hat)*100:>7.3f}  "
          f"{metrics.precision_score(y_test,y_hat,zero_division=0):>7.3f}  "
          f"{metrics.recall_score(y_test,y_hat,zero_division=0):>7.3f}  "
          f"{metrics.f1_score(y_test,y_hat,zero_division=0):>7.3f}  "
          f"{metrics.roc_auc_score(y_test,y_prob):>7.3f}")

print(f"\n10-FOLD CROSS-VALIDATION  (mean accuracy)")
for name, _ in classifiers:
    cv_acc = mean(cv_results[name]["test_accuracy"]) * 100
    cv_std = np.std(cv_results[name]["test_accuracy"]) * 100
    cv_f1  = mean(cv_results[name]["test_f1_macro"])
    print(f"  {name:<30}  Acc={cv_acc:.3f}%  (+/-{cv_std:.3f}%)  F1={cv_f1:.3f}")

print(f"\nTOP 5 FEATURES  (Random Forest Gini Importance)")
for i in range(5):
    print(f"  {i+1}. {ALL_FEATS[top_idx[i]]:<45}  {imps[top_idx[i]]:.4f}")

n_figs = len([f for f in os.listdir(FIG_DIR) if f.endswith(".png")])
print(f"\nOUTPUTS")
print(f"  {n_figs} figures  -->  {FIG_DIR}")
print(f"  reports  -->  {RPT_DIR}")

print("\n" + "=" * 60)
print("All done. Pipeline complete.")
print("=" * 60)
