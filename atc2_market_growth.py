# %% [markdown]
# # ATC2 Market Growth Analysis — Module 1
#
# This replaces the legacy MEFI pipeline rather than renaming its grouping key.
# Key design changes: the source is `atc3_region_facil` grouped by categorical
# `atcStep2Cd`; Matrix A is Growth Amount × CAGR with 2022 market-size bubbles;
# Matrix B separately validates endpoint CAGR against structural WLS growth; and
# structural decline remains an independent, statistically significant warning.

# %% STEP 00. Imports / Parameters
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython.display import display

DATABASE_PATH = Path("healthcare.duckdb")
SOURCE_TABLE = "atc3_region_facil"
ATC_MASTER_TABLE = "atc_master"
START_DIAG_YM = 202001
END_DIAG_YM = 202212
START_YEAR = 2020
END_YEAR = 2022
ANALYSIS_MONTHS = 36
RECENCY_DECAY = 0.90
MAD_SCALE = 1.4826
GAP_THRESHOLD_PP = 10
GAP_Z_THRESHOLD = 3
DECLINE_TSTAT_THRESHOLD = 2
RECENT_YOY_MONTHS = 12
TOP_N_GROUPS = 12

plt.rcParams.update({
    "figure.figsize": (13, 7), "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False,
})
pd.set_option("display.max_columns", 100)


def robust_zscore(values: pd.Series) -> pd.Series:
    """Return median/MAD z-scores; use standard deviation only when MAD is zero."""
    values = pd.Series(values, dtype="float64")
    median = values.median()
    scale = MAD_SCALE * np.median(np.abs(values.dropna() - median))
    if not np.isfinite(scale) or scale == 0:
        std = values.std()
        return pd.Series(0.0, index=values.index) if not np.isfinite(std) or std == 0 else (values - values.mean()) / std
    return (values - median) / scale


def add_name_label(frame: pd.DataFrame) -> pd.Series:
    """Keep an ATC2 code visible if its master-name lookup is missing."""
    return frame["group_name"].fillna(frame["group_id"])


# %% STEP 01. Source QA
con = duckdb.connect(str(DATABASE_PATH), read_only=True)
source_qa = con.execute(f"""
    WITH filtered AS (
        SELECT * FROM {SOURCE_TABLE}
        WHERE CAST(diagYm AS INTEGER) BETWEEN {START_DIAG_YM} AND {END_DIAG_YM}
    ), monthly_observations AS (
        SELECT atcStep2Cd, COUNT(DISTINCT diagYm) AS month_observations
        FROM filtered GROUP BY atcStep2Cd
    )
    SELECT
        (SELECT MIN(CAST(diagYm AS INTEGER)) FROM {SOURCE_TABLE}) AS source_min_diag_ym,
        (SELECT MAX(CAST(diagYm AS INTEGER)) FROM {SOURCE_TABLE}) AS source_max_diag_ym,
        (SELECT COUNT(DISTINCT diagYm) FROM filtered) AS analysis_distinct_months,
        (SELECT COUNT(DISTINCT atcStep2Cd) FROM filtered) AS analysis_distinct_atc2,
        (SELECT LIST(DISTINCT insupTpCd ORDER BY insupTpCd) FROM filtered) AS insup_tp_values,
        (SELECT LIST(DISTINCT medInstType ORDER BY medInstType) FROM filtered) AS med_inst_type_values,
        (SELECT MIN(month_observations) FROM monthly_observations) AS atc2_month_observations_min,
        (SELECT MAX(month_observations) FROM monthly_observations) AS atc2_month_observations_max
""").df()
source_qa["is_exactly_36_months"] = source_qa["analysis_distinct_months"].eq(ANALYSIS_MONTHS)
print("STEP 01 — source and period QA")
display(source_qa)
assert bool(source_qa.loc[0, "is_exactly_36_months"]), "Analysis window must contain exactly 36 distinct months."


# %% STEP 02. ATC2 Master Join
atc2_master = con.execute(f"""
    SELECT CAST(atc_code AS VARCHAR) AS group_id, MAX(atc_name) AS group_name
    FROM {ATC_MASTER_TABLE}
    WHERE LENGTH(CAST(atc_code AS VARCHAR)) = 3
    GROUP BY 1
""").df()
print("STEP 02 — ATC2 master (three-character codes only)")
display(atc2_master.head())


# %% STEP 03. Monthly Market Panel
# DuckDB performs the large source aggregation. Only the small ATC2 × month panel
# is brought into pandas for statistical modelling and plotting.
monthly_atc2 = con.execute(f"""
    WITH atc2_master AS (
        SELECT CAST(atc_code AS VARCHAR) AS group_id, MAX(atc_name) AS group_name
        FROM {ATC_MASTER_TABLE}
        WHERE LENGTH(CAST(atc_code AS VARCHAR)) = 3
        GROUP BY 1
    ), monthly AS (
        SELECT
            CAST(atcStep2Cd AS VARCHAR) AS group_id,
            CAST(SUBSTR(CAST(diagYm AS VARCHAR), 1, 4) AS INTEGER) AS year,
            CAST(SUBSTR(CAST(diagYm AS VARCHAR), 5, 2) AS INTEGER) AS month,
            SUM(msupUseAmt) AS market_amt
        FROM {SOURCE_TABLE}
        WHERE CAST(diagYm AS INTEGER) BETWEEN {START_DIAG_YM} AND {END_DIAG_YM}
        GROUP BY 1, 2, 3
    )
    SELECT monthly.group_id, master.group_name, monthly.year, monthly.month, monthly.market_amt
    FROM monthly
    LEFT JOIN atc2_master AS master USING (group_id)
    ORDER BY monthly.group_id, monthly.year, monthly.month
""").df()
monthly_atc2["date"] = pd.to_datetime(monthly_atc2[["year", "month"]].assign(day=1))
monthly_atc2["market_amt"] = monthly_atc2["market_amt"].astype(float)
monthly_atc2["market_amt_억원"] = monthly_atc2["market_amt"] / 1e8
monthly_atc2["log_market_amt"] = np.where(monthly_atc2["market_amt"] > 0, np.log(monthly_atc2["market_amt"]), np.nan)
monthly_atc2 = monthly_atc2.sort_values(["group_id", "date"]).reset_index(drop=True)
print("STEP 03 — monthly ATC2 market panel")
display(monthly_atc2.head())


# %% STEP 04. Structural Trend
# WLS is the primary model. HuberT RLM is used only to diagnose unusually large
# residuals, never as a replacement for the WLS trend estimate.
def fit_structural_trend(group: pd.DataFrame) -> pd.Series:
    data = group.dropna(subset=["log_market_amt"]).sort_values("date").copy()
    if len(data) < 24:
        return pd.Series({key: np.nan for key in (
            "trend_beta_monthly", "normalized_annual_growth", "normalized_annual_growth_pct",
            "trend_r2", "trend_tstat", "trend_pvalue", "robust_residual_outlier_count",
            "robust_residual_outlier_rate",
        )})
    data["time"] = np.arange(len(data))
    data["weight"] = RECENCY_DECAY ** (data["time"].max() - data["time"])
    month_dummies = pd.get_dummies(data["month"].astype(str), prefix="month", drop_first=True, dtype=float)
    exog = sm.add_constant(pd.concat([data[["time"]], month_dummies], axis=1), has_constant="add")
    wls = sm.WLS(data["log_market_amt"], exog, weights=data["weight"]).fit()
    beta = wls.params["time"]
    try:
        rlm = sm.RLM(data["log_market_amt"], exog, M=sm.robust.norms.HuberT()).fit()
        residuals = rlm.resid
        residual_scale = MAD_SCALE * np.median(np.abs(residuals - np.median(residuals)))
        outlier_count = int((np.abs((residuals - np.median(residuals)) / residual_scale) >= 3).sum()) if residual_scale > 0 else 0
    except (np.linalg.LinAlgError, ValueError):
        outlier_count = np.nan
    return pd.Series({
        "trend_beta_monthly": beta,
        "normalized_annual_growth": np.exp(beta * 12) - 1,
        "normalized_annual_growth_pct": (np.exp(beta * 12) - 1) * 100,
        "trend_r2": wls.rsquared, "trend_tstat": wls.tvalues["time"], "trend_pvalue": wls.pvalues["time"],
        "robust_residual_outlier_count": outlier_count,
        "robust_residual_outlier_rate": outlier_count / len(data) if pd.notna(outlier_count) else np.nan,
    })

trend_atc2 = (monthly_atc2.groupby(["group_id", "group_name"], dropna=False, group_keys=False)
              .apply(fit_structural_trend).reset_index())
print("STEP 04 — recency-weighted structural trends")
display(trend_atc2.head())


# %% STEP 05. Endpoint Growth
annual_market = monthly_atc2.groupby(["group_id", "group_name", "year"], dropna=False, as_index=False)["market_amt"].sum()
growth_atc2 = annual_market.pivot(index=["group_id", "group_name"], columns="year", values="market_amt").reset_index()
growth_atc2.columns.name = None
for year in (START_YEAR, END_YEAR):
    if year not in growth_atc2:
        growth_atc2[year] = np.nan
growth_atc2 = growth_atc2.rename(columns={START_YEAR: "market_2020", END_YEAR: "market_2022"})
growth_atc2["market_2022_억원"] = growth_atc2["market_2022"] / 1e8
growth_atc2["growth_amount"] = growth_atc2["market_2022"] - growth_atc2["market_2020"]
growth_atc2["growth_amount_억원"] = growth_atc2["growth_amount"] / 1e8
growth_atc2["cagr"] = np.where(
    (growth_atc2["market_2020"] > 0) & (growth_atc2["market_2022"] >= 0),
    (growth_atc2["market_2022"] / growth_atc2["market_2020"]) ** (1 / 2) - 1, np.nan,
)
growth_atc2["cagr_pct"] = growth_atc2["cagr"] * 100


# %% STEP 06. CAGR vs Structural Gap
diagnostic_atc2 = growth_atc2.merge(trend_atc2, on=["group_id", "group_name"], how="left")
diagnostic_atc2["cagr_trend_gap"] = diagnostic_atc2["cagr"] - diagnostic_atc2["normalized_annual_growth"]
diagnostic_atc2["cagr_trend_gap_pp"] = diagnostic_atc2["cagr_trend_gap"] * 100
diagnostic_atc2["gap_robust_z"] = robust_zscore(diagnostic_atc2["cagr_trend_gap_pp"])
diagnostic_atc2["endpoint_distortion_flag"] = (
    diagnostic_atc2["cagr_trend_gap_pp"].abs().ge(GAP_THRESHOLD_PP)
    | diagnostic_atc2["gap_robust_z"].abs().ge(GAP_Z_THRESHOLD)
)


# %% STEP 07. Rolling YoY / Momentum
monthly_atc2["yoy_growth"] = monthly_atc2.groupby("group_id")["market_amt"].pct_change(periods=12)
monthly_atc2["yoy_growth_pct"] = monthly_atc2["yoy_growth"] * 100

def calc_yoy_momentum(group: pd.DataFrame) -> pd.Series:
    data = group.dropna(subset=["yoy_growth"]).sort_values("date").copy()
    result = {"mean_yoy_growth": data["yoy_growth"].mean(), "yoy_growth_sd": data["yoy_growth"].std()}
    if len(data) < RECENT_YOY_MONTHS:
        return pd.Series(result | {key: np.nan for key in (
            "yoy_beta_recent", "yoy_beta_recent_tstat", "yoy_beta_recent_pvalue",
            "early_yoy_beta", "late_yoy_beta", "yoy_beta_change", "yoy_beta_change_tstat", "yoy_beta_change_pvalue",
        )})
    # Recent period is the final 12 available monthly YoY observations (2022-01 to 2022-12 for complete data).
    recent = data.tail(RECENT_YOY_MONTHS).assign(time=np.arange(RECENT_YOY_MONTHS))
    recent_fit = sm.OLS(recent["yoy_growth"], sm.add_constant(recent["time"])).fit()
    result.update({"yoy_beta_recent": recent_fit.params["time"], "yoy_beta_recent_tstat": recent_fit.tvalues["time"], "yoy_beta_recent_pvalue": recent_fit.pvalues["time"]})
    # With 36 monthly values, 24 YoY observations split into early (2021) and late (2022) 12-month slopes.
    if len(data) >= 24:
        two_years = data.tail(24).copy().assign(time=np.arange(24))
        two_years["late"] = (two_years["time"] >= 12).astype(int)
        two_years["time_late"] = two_years["time"] * two_years["late"]
        change_fit = sm.OLS(two_years["yoy_growth"], sm.add_constant(two_years[["time", "late", "time_late"]])).fit()
        result.update({
            "early_yoy_beta": change_fit.params["time"],
            "late_yoy_beta": change_fit.params["time"] + change_fit.params["time_late"],
            "yoy_beta_change": change_fit.params["time_late"],
            "yoy_beta_change_tstat": change_fit.tvalues["time_late"],
            "yoy_beta_change_pvalue": change_fit.pvalues["time_late"],
        })
    else:
        result.update({key: np.nan for key in ("early_yoy_beta", "late_yoy_beta", "yoy_beta_change", "yoy_beta_change_tstat", "yoy_beta_change_pvalue")})
    return pd.Series(result)

momentum_atc2 = (monthly_atc2.groupby(["group_id", "group_name"], dropna=False, group_keys=False)
                 .apply(calc_yoy_momentum).reset_index())
diagnostic_atc2 = diagnostic_atc2.merge(momentum_atc2, on=["group_id", "group_name"], how="left")


# %% STEP 08. Absolute Decline
diagnostic_atc2["is_declining"] = (
    diagnostic_atc2["normalized_annual_growth"].lt(0)
    & diagnostic_atc2["trend_tstat"].abs().ge(DECLINE_TSTAT_THRESHOLD)
)
diagnostic_atc2["decline_severity_rank"] = pd.Series(pd.NA, index=diagnostic_atc2.index, dtype="Int64")
declining_index = diagnostic_atc2.index[diagnostic_atc2["is_declining"]]
diagnostic_atc2.loc[declining_index, "decline_severity_rank"] = (
    diagnostic_atc2.loc[declining_index, "normalized_annual_growth"].rank(ascending=True, method="min").astype("Int64")
)


# %% STEP 09. Matrix A
# Medians provide relative positioning within the limited ATC2 universe, avoiding arbitrary absolute thresholds.
growth_amount_median = diagnostic_atc2["growth_amount"].median()
cagr_median = diagnostic_atc2["cagr"].median()
def classify_segment(row: pd.Series) -> str | float:
    if pd.isna(row["growth_amount"]) or pd.isna(row["cagr"]): return np.nan
    if row["growth_amount"] >= growth_amount_median and row["cagr"] >= cagr_median: return "Star"
    if row["growth_amount"] >= growth_amount_median: return "Cash Cow"
    if row["cagr"] >= cagr_median: return "Rising"
    return "Laggard"
diagnostic_atc2["segment"] = diagnostic_atc2.apply(classify_segment, axis=1)


# %% STEP 10. Matrix B
# Matrix B intentionally uses the two separate growth definitions; it does not redefine Matrix A.

# %% STEP 11. Diagnostic Table
DIAGNOSTIC_COLUMNS = [
    "group_id", "group_name", "market_2020", "market_2022", "market_2022_억원", "growth_amount", "growth_amount_억원",
    "cagr", "cagr_pct", "trend_beta_monthly", "normalized_annual_growth", "normalized_annual_growth_pct", "trend_r2", "trend_tstat", "trend_pvalue",
    "robust_residual_outlier_count", "robust_residual_outlier_rate", "cagr_trend_gap", "cagr_trend_gap_pp", "gap_robust_z", "endpoint_distortion_flag",
    "mean_yoy_growth", "yoy_growth_sd", "yoy_beta_recent", "yoy_beta_recent_tstat", "yoy_beta_recent_pvalue",
    "early_yoy_beta", "late_yoy_beta", "yoy_beta_change", "yoy_beta_change_tstat", "yoy_beta_change_pvalue",
    "is_declining", "decline_severity_rank", "segment",
]
diagnostic_atc2 = diagnostic_atc2[DIAGNOSTIC_COLUMNS].sort_values(["segment", "growth_amount"], ascending=[True, False], na_position="last").reset_index(drop=True)
print("STEP 11 — complete diagnostic table")
display(diagnostic_atc2)


# %% STEP 12. Anomaly Groups
anomaly_groups = {
    "Star + Declining": diagnostic_atc2.query("segment == 'Star' and is_declining"),
    "Cash Cow + Declining": diagnostic_atc2.query("segment == 'Cash Cow' and is_declining"),
    # Low reliability is reported transparently using the existing t-stat and endpoint-distortion criteria.
    "Rising + Low Reliability": diagnostic_atc2.query("segment == 'Rising' and (abs(trend_tstat) < @DECLINE_TSTAT_THRESHOLD or endpoint_distortion_flag)"),
    "Laggard + Recovery": diagnostic_atc2.query("segment == 'Laggard' and yoy_beta_recent > 0"),
    "Endpoint/Structural divergence": diagnostic_atc2.query("abs(cagr_trend_gap_pp) >= @GAP_THRESHOLD_PP or abs(gap_robust_z) >= @GAP_Z_THRESHOLD"),
    "Negative Growth Amount + CAGR": diagnostic_atc2.query("growth_amount < 0 and cagr < 0"),
}
for title, frame in anomaly_groups.items():
    print(f"STEP 12 — {title}: {len(frame)} groups")
    display(frame)


# %% STEP 13. Segment Summary
segment_summary = (diagnostic_atc2.dropna(subset=["segment"]).groupby("segment", as_index=False)
                   .agg(group_count=("group_id", "size"), total_2022_market=("market_2022", "sum"),
                        total_growth_amount=("growth_amount", "sum"), median_cagr=("cagr", "median"),
                        median_structural_growth=("normalized_annual_growth", "median"), declining_count=("is_declining", "sum")))
segment_summary["group_share"] = segment_summary["group_count"] / segment_summary["group_count"].sum()
segment_summary["declining_share"] = segment_summary["declining_count"] / segment_summary["group_count"]
display(segment_summary)


# %% STEP 14. Visualization
plot_names = add_name_label(diagnostic_atc2)
# Figure 1
overall_monthly = monthly_atc2.groupby("date", as_index=False)["market_amt_억원"].sum()
plt.figure(); plt.plot(overall_monthly["date"], overall_monthly["market_amt_억원"], linewidth=2)
plt.title("Figure 1 — ATC2 Total Monthly Market Size"); plt.xlabel("Month"); plt.ylabel("Market Size (억원)"); plt.tight_layout(); plt.show()
# Figure 2
plt.figure(); plt.hist(diagnostic_atc2["normalized_annual_growth_pct"].dropna(), bins=20, edgecolor="white")
plt.title("Figure 2 — Structural Annual Growth Distribution"); plt.xlabel("Normalized Annual Growth (%)"); plt.ylabel("ATC2 count"); plt.tight_layout(); plt.show()
# Figure 3: Matrix A — never log-scale growth amount because negative values are meaningful.
colors = {"Star": "#2E86AB", "Cash Cow": "#F6AE2D", "Rising": "#3DA35D", "Laggard": "#9B59B6"}
fig, ax = plt.subplots()
for segment, frame in diagnostic_atc2.groupby("segment", dropna=True):
    ax.scatter(frame["growth_amount_억원"], frame["cagr_pct"], s=np.sqrt(frame["market_2022_억원"].clip(lower=0)) * 30 + 30,
               c=colors[segment], alpha=.7, label=segment, edgecolors="white")
decline = diagnostic_atc2[diagnostic_atc2["is_declining"]]
ax.scatter(decline["growth_amount_억원"], decline["cagr_pct"], s=np.sqrt(decline["market_2022_억원"].clip(lower=0)) * 30 + 45,
           facecolors="none", edgecolors="black", linewidths=1.8, label="Structural decline")
for (_, row), label in zip(diagnostic_atc2.iterrows(), plot_names): ax.annotate(label, (row["growth_amount_억원"], row["cagr_pct"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.axvline(growth_amount_median / 1e8, color="grey", linestyle="--"); ax.axhline(cagr_median * 100, color="grey", linestyle="--")
ax.set(title="Figure 3 — Matrix A: Growth Amount × CAGR", xlabel="Growth Amount (억원)", ylabel="CAGR (%)"); ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left"); plt.tight_layout(); plt.show()
# Figure 4: Matrix B
fig, ax = plt.subplots(); ax.scatter(diagnostic_atc2["cagr_pct"], diagnostic_atc2["normalized_annual_growth_pct"], alpha=.7)
limits = np.array([diagnostic_atc2[["cagr_pct", "normalized_annual_growth_pct"]].min().min(), diagnostic_atc2[["cagr_pct", "normalized_annual_growth_pct"]].max().max()])
ax.plot(limits, limits, "k--", label="Y = X")
flag = diagnostic_atc2[diagnostic_atc2["endpoint_distortion_flag"]]; ax.scatter(flag["cagr_pct"], flag["normalized_annual_growth_pct"], facecolors="none", edgecolors="red", s=100, label="Endpoint distortion")
for (_, row), label in zip(diagnostic_atc2.iterrows(), plot_names): ax.annotate(label, (row["cagr_pct"], row["normalized_annual_growth_pct"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set(title="Figure 4 — Matrix B: CAGR × Structural Annual Growth", xlabel="CAGR (%)", ylabel="Structural Annual Growth (%)"); ax.legend(); plt.tight_layout(); plt.show()
# Figure 5
plt.figure(); plt.hist(diagnostic_atc2["cagr_trend_gap_pp"].dropna(), bins=20, edgecolor="white"); plt.axvline(0, color="black", linestyle="--")
plt.title("Figure 5 — CAGR − Structural Growth Gap"); plt.xlabel("Percentage points"); plt.ylabel("ATC2 count"); plt.tight_layout(); plt.show()
# Figure 6
momentum_rank = diagnostic_atc2.dropna(subset=["yoy_beta_recent"]).sort_values("yoy_beta_recent")
plot_momentum = pd.concat([momentum_rank.head(TOP_N_GROUPS), momentum_rank.tail(TOP_N_GROUPS)]).drop_duplicates("group_id")
plt.figure(figsize=(13, 8)); plt.barh(add_name_label(plot_momentum), plot_momentum["yoy_beta_recent"] * 100)
plt.title("Figure 6 — Recent YoY Momentum: Bottom / Top Groups"); plt.xlabel("YoY slope (percentage points per month)"); plt.tight_layout(); plt.show()
# Figure 7
plt.figure(); plt.scatter(diagnostic_atc2["yoy_beta_recent"] * 100, diagnostic_atc2["normalized_annual_growth_pct"], alpha=.7)
plt.title("Figure 7 — Momentum × Structural Growth"); plt.xlabel("Recent YoY slope (percentage points per month)"); plt.ylabel("Structural Annual Growth (%)"); plt.tight_layout(); plt.show()


# %% STEP 15. Final QA
final_qa = pd.DataFrame([{
    "ATC2 group count": diagnostic_atc2["group_id"].nunique(),
    "36 months": monthly_atc2["date"].nunique() == ANALYSIS_MONTHS,
    "missing group_name count": diagnostic_atc2["group_name"].isna().sum(),
    "missing market amount count": monthly_atc2["market_amt"].isna().sum(),
    "missing trend result count": diagnostic_atc2["normalized_annual_growth"].isna().sum(),
    "missing CAGR count": diagnostic_atc2["cagr"].isna().sum(),
    "declining group count": diagnostic_atc2["is_declining"].sum(),
    "segment group count": diagnostic_atc2["segment"].notna().sum(),
    "segment sum equals ATC2 count": diagnostic_atc2["segment"].notna().sum() == diagnostic_atc2["group_id"].nunique(),
    "declining groups have segments": diagnostic_atc2.loc[diagnostic_atc2["is_declining"], "segment"].notna().all(),
}])
print("STEP 15 — final QA")
display(final_qa)
assert bool(final_qa.loc[0, "36 months"])
assert bool(final_qa.loc[0, "segment sum equals ATC2 count"])
assert bool(final_qa.loc[0, "declining groups have segments"])
