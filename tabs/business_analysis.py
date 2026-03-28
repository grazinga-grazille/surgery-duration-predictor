"""
tabs/business_analysis.py  —  Business Analysis tab

OR costs are calculated per case first, then summed per specialty.
Data: loadfile.csv, filtered to Mar 2024 – Mar 2025.

Cost formula (applied per case, always positive):
  net >= 0  (over-ran)  →  net      × $35 × 1.5
  net <  0  (under-ran) →  abs(net) × $35
"""

import os
import pickle
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────
_DATA_PATH  = "loadfile.csv"
DATE_START  = "2024-03-01"
DATE_END    = "2025-03-31"
_RATE       = 35.0
_RATE_OVER  = _RATE * 1.5   # $/min when surgery ran over booked time
_RATE_UNDER = _RATE          # $/min when surgery ran under booked time

_BLUE = "#4A90D9"
_TEAL = "#1ABC9C"

_CUSTOM_STOP_WORDS = [
    "removal", "extraction", "total", "non", "diagnostic", "performed",
    "procedure", "status", "post", "bilateral", "left", "right", "well",
    "tolerated", "bx", "fx", "tx", "dx", "hx", "sx", "ex",
]


# ── Data availability check ───────────────────────────────────────────────────

def _data_available() -> bool:
    return os.path.exists(_DATA_PATH)


def _show_data_unavailable():
    st.warning(
        f"**Data file not found** — `{_DATA_PATH}` is required for this analysis "
        "but is not present in this deployment (confidential health data is excluded "
        "from the repository). To view this section, run the app locally with the "
        "source data file in the project root directory.",
        icon="🔒",
    )


# ── Cost formula ──────────────────────────────────────────────────────────────

def _compute_cost(net_minutes: float) -> float:
    """Cost for a single case. Always returns a positive value.
      net >= 0  →  net      × $52.50  (over-ran, premium rate)
      net <  0  →  abs(net) × $35.00  (under-ran, base rate)
    """
    if net_minutes >= 0:
        return net_minutes * _RATE_OVER
    return abs(net_minutes) * _RATE_UNDER


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _clean_text(txt: str) -> str:
    txt = str(txt).lower()
    txt = re.sub(r"[^a-z ]", " ", txt)
    return " ".join(w for w in txt.split() if w not in _CUSTOM_STOP_WORDS)


# ── Data loading & batch prediction ──────────────────────────────────────────

@st.cache_data(show_spinner="Loading OR data and running predictions …")
def _load_comparison_data() -> pd.DataFrame:
    """Load 2024-2025 cases, run model predictions, compute per-case costs,
    then aggregate per specialty.

    Returns a DataFrame with columns:
      ProcedureSpecialtyDescription, case_count,
      baseline_net_min, predicted_net_min,
      baseline_cost, predicted_cost, savings
    """
    # ── Load model artifacts ──────────────────────────────────────────────────
    def _pkl(name):
        with open(f"models/{name}", "rb") as f:
            return pickle.load(f)

    rf_model        = _pkl("rf_model.pkl")
    tfidf           = _pkl("tfidf.pkl")
    svd             = _pkl("svd.pkl")
    feature_columns = _pkl("feature_columns.pkl")

    # ── Load and filter CSV ───────────────────────────────────────────────────
    cols = [
        "ProcedureDescription", "ProcedureSpecialtyDescription",
        "SurgeryDate", "PatientType", "Roomdescription",
        "SurgicalPriority", "book_dur", "ActualDurationMinutes",
    ]
    df = pd.read_csv(_DATA_PATH, usecols=cols)
    df["SurgeryDate"] = pd.to_datetime(df["SurgeryDate"])
    df = df.dropna(subset=["ActualDurationMinutes", "book_dur",
                            "SurgicalPriority", "ProcedureDescription"])
    mask = (df["SurgeryDate"] >= DATE_START) & (df["SurgeryDate"] <= DATE_END)
    df = df[mask].reset_index(drop=True)

    for col in ["PatientType", "Roomdescription", "ProcedureSpecialtyDescription"]:
        df[col] = df[col].str.strip()

    # ── Batch predict ─────────────────────────────────────────────────────────
    cleaned   = df["ProcedureDescription"].apply(_clean_text)
    tfidf_mat = tfidf.transform(cleaned)
    svd_mat   = svd.transform(tfidf_mat)

    svd_df = pd.DataFrame(
        svd_mat, columns=[f"SVD_{i}" for i in range(svd_mat.shape[1])]
    )
    X_num = pd.DataFrame({"SurgicalPriority": df["SurgicalPriority"].values})
    X_cat = pd.get_dummies(
        df[["PatientType", "Roomdescription", "ProcedureSpecialtyDescription"]],
        drop_first=True, dtype=int,
    )
    X = pd.concat([X_num, X_cat, svd_df], axis=1).reindex(
        columns=feature_columns, fill_value=0
    )
    df["predicted"] = np.exp(rf_model.predict(X))

    # ── Per-case net OR minutes ───────────────────────────────────────────────
    df["baseline_net"]  = df["ActualDurationMinutes"] - df["book_dur"]
    df["predicted_net"] = df["ActualDurationMinutes"] - df["predicted"]

    # ── Per-case cost (formula applied before aggregation) ───────────────────
    df["baseline_cost_case"]  = df["baseline_net"].apply(_compute_cost)
    df["predicted_cost_case"] = df["predicted_net"].apply(_compute_cost)

    # ── Aggregate per specialty ───────────────────────────────────────────────
    grp = (
        df.groupby("ProcedureSpecialtyDescription", as_index=False)
        .agg(
            case_count        =("ActualDurationMinutes", "count"),
            baseline_net_min  =("baseline_net",          "sum"),
            predicted_net_min =("predicted_net",         "sum"),
            baseline_cost     =("baseline_cost_case",    "sum"),
            predicted_cost    =("predicted_cost_case",   "sum"),
        )
    )
    grp["savings"] = grp["baseline_cost"] - grp["predicted_cost"]

    return grp.sort_values("baseline_cost").reset_index(drop=True)


# ── Tab renderer ──────────────────────────────────────────────────────────────

def render():
    st.subheader("Business Analysis")
    st.caption(
        "Quantifying the operational and financial impact of improved scheduling accuracy."
    )

    fi_tab, ru_tab = st.tabs([
        "💰 Financial Impact",
        "🏥 Resource Utilization Impact",
    ])

    with fi_tab:
        _render_financial_impact()

    with ru_tab:
        _render_resource_utilization()


# ── Chart helper ──────────────────────────────────────────────────────────────

def _bar_chart(specialties, values, color, y_title, key):
    fig = go.Figure(go.Bar(
        x=specialties,
        y=values,
        marker_color=color,
        text=[f"${v:,.0f}" for v in values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Cost: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        height=420,
        margin=dict(t=60, b=20, l=10, r=10),
        xaxis_title="Surgical Specialty",
        yaxis_title=y_title,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f0f4f8"),
        xaxis=dict(tickfont=dict(size=11), tickangle=-20),
        yaxis=dict(gridcolor="rgba(149,165,166,0.2)", tickformat="$,.0f",
                   rangemode="tozero"),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ── Financial impact renderer ─────────────────────────────────────────────────

def _render_financial_impact():
    if not _data_available():
        _show_data_unavailable()
        return
    data = _load_comparison_data()
    specialties   = data["ProcedureSpecialtyDescription"].tolist()
    baseline_cost = data["baseline_cost"].tolist()
    pred_cost     = data["predicted_cost"].tolist()
    savings       = data["savings"].tolist()
    total_cases   = int(data["case_count"].sum())
    total_saved   = sum(savings)

    caption = f"**{DATE_START}** to **{DATE_END}** · {total_cases:,} cases"

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Total Baseline Cost",
        f"${sum(baseline_cost):,.0f}",
        help=(
            "Total net OR scheduling cost across all specialties using the originally "
            "booked duration as the planned schedule. Each case is costed by how far "
            "its actual duration deviated from the booked time — overtime at \$52.50 per min, "
            "undertime (opportunity cost) at \$35 per min."
        ),
    )
    m2.metric(
        "Total Predicted Cost",
        f"${sum(pred_cost):,.0f}",
        help=(
            "Total net OR scheduling cost if the model's predicted duration had been "
            "used as the planned schedule instead of the booked duration. A lower value "
            "than the baseline means the model's predictions lead to smaller deviations "
            "from actual surgery times."
        ),
    )
    m3.metric(
        "Net Cost Impact",
        f"${total_saved:+,.0f}",
        delta="savings from model predictions" if total_saved >= 0 else "additional cost from model predictions",
        delta_color="normal" if total_saved >= 0 else "inverse",
        help=(
            "Difference between baseline cost and predicted cost (baseline − predicted). "
            "A positive value means the model's schedule predictions reduce overall OR "
            "scheduling cost compared to the original booked durations."
        ),
    )
    st.divider()

    # ── Baseline chart ────────────────────────────────────────────────────────
    st.markdown("#### Baseline Net OR Cost by Specialty")
    st.caption(
        caption + " · Cost of scheduling deviations using the originally booked duration as the plan"
    )
    _bar_chart(specialties, baseline_cost, _BLUE, "Net OR Cost ($)", "fi_baseline")

    # ── Prediction chart ──────────────────────────────────────────────────────
    st.markdown("#### Model Prediction Net OR Cost by Specialty")
    st.caption(
        caption + " · Cost of scheduling deviations using the model's predicted duration as the plan"
    )
    _bar_chart(specialties, pred_cost, _TEAL, "Net OR Cost ($)", "fi_predicted")

    # ── Cost impact chart ─────────────────────────────────────────────────────
    st.markdown("#### Cost Savings by Specialty")
    st.caption(
        "How much the model reduces (or increases) net OR scheduling cost per specialty "
        "compared to the baseline. Green = model saves money · Red = model adds cost"
    )
    savings_colors = ["#27AE60" if s >= 0 else "#E74C3C" for s in savings]
    fig_sav = go.Figure(go.Bar(
        x=specialties,
        y=savings,
        marker_color=savings_colors,
        text=[f"${s:+,.0f}" for s in savings],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Cost Savings: $%{y:+,.0f}<extra></extra>",
    ))
    fig_sav.add_hline(y=0, line_color="#95A5A6", line_width=1)
    fig_sav.update_layout(
        height=420,
        margin=dict(t=60, b=20, l=10, r=10),
        xaxis_title="Surgical Specialty",
        yaxis_title="Cost Savings vs. Baseline ($)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f0f4f8"),
        xaxis=dict(tickfont=dict(size=11), tickangle=-20),
        yaxis=dict(gridcolor="rgba(149,165,166,0.2)", tickformat="$,.0f"),
    )
    st.plotly_chart(fig_sav, use_container_width=True)


# ── Resource Utilization — data ───────────────────────────────────────────────

@st.cache_data(show_spinner="Building resource utilization data …")
def _load_resource_utilization_data():
    """For each specialty × day, compute:
      - surgeries performed
      - net time saved (booked − actual), floored at 0
      - how many additional high-count procedures fit in that saved time

    Saves a daily-level CSV and returns both the daily DataFrame and the
    per-specialty summary used for the chart.
    """
    cols = [
        "ProcedureDescription", "ProcedureSpecialtyDescription",
        "SurgeryDate", "book_dur", "ActualDurationMinutes",
    ]
    df = pd.read_csv(_DATA_PATH, usecols=cols)
    df["SurgeryDate"] = pd.to_datetime(df["SurgeryDate"])
    df = df.dropna(subset=["ActualDurationMinutes", "book_dur", "ProcedureDescription"])
    df = df[df["ActualDurationMinutes"] > 0]   # drop bad rows with zero/negative duration
    mask = (df["SurgeryDate"] >= DATE_START) & (df["SurgeryDate"] <= DATE_END)
    df = df[mask].copy()
    df["ProcedureSpecialtyDescription"] = df["ProcedureSpecialtyDescription"].str.strip()

    # ── Most common procedure per specialty (by count) + its avg duration ────
    proc_stats = (
        df.groupby(["ProcedureSpecialtyDescription", "ProcedureDescription"])
        .agg(count=("ActualDurationMinutes", "count"),
             avg_dur=("ActualDurationMinutes", "mean"))
        .reset_index()
    )
    top_proc = (
        proc_stats
        .loc[proc_stats.groupby("ProcedureSpecialtyDescription")["count"].idxmax()]
        .rename(columns={"ProcedureDescription": "most_common_procedure",
                         "count": "procedure_count",
                         "avg_dur": "avg_procedure_duration_min"})
        [["ProcedureSpecialtyDescription", "most_common_procedure",
          "procedure_count", "avg_procedure_duration_min"]]
    )

    # ── Per-specialty × day aggregation ──────────────────────────────────────
    df["time_diff"] = df["book_dur"] - df["ActualDurationMinutes"]
    daily = (
        df.groupby(["ProcedureSpecialtyDescription", "SurgeryDate"], as_index=False)
        .agg(surgeries_performed=("ActualDurationMinutes", "count"),
             net_time_saved_raw  =("time_diff",            "sum"))
    )
    daily["time_saved_min"] = daily["net_time_saved_raw"].clip(lower=0)
    daily = daily.drop(columns="net_time_saved_raw")

    # ── Merge avg procedure duration and compute additional surgeries ─────────
    daily = daily.merge(top_proc, on="ProcedureSpecialtyDescription", how="left")
    daily["additional_surgeries_possible"] = (
        daily["time_saved_min"] / daily["avg_procedure_duration_min"]
    ).apply(np.floor).fillna(0).astype(int)

    # ── Save daily CSV ────────────────────────────────────────────────────────
    csv_out = daily.rename(columns={
        "ProcedureSpecialtyDescription": "Specialty",
        "SurgeryDate":                   "Date",
        "surgeries_performed":           "Surgeries_Performed",
        "time_saved_min":                "Time_Saved_Min",
        "most_common_procedure":         "Most_Common_Procedure",
        "avg_procedure_duration_min":    "Avg_Procedure_Duration_Min",
        "additional_surgeries_possible": "Additional_Surgeries_Possible",
        "procedure_count":               "Procedure_Count_In_Period",
    })[["Date", "Specialty", "Surgeries_Performed", "Time_Saved_Min",
        "Most_Common_Procedure", "Avg_Procedure_Duration_Min",
        "Additional_Surgeries_Possible", "Procedure_Count_In_Period"]]
    csv_out.to_csv("resource_utilization.csv", index=False)

    # ── Per-specialty summary for chart ──────────────────────────────────────
    summary = (
        daily.groupby("ProcedureSpecialtyDescription", as_index=False)
        .agg(
            total_surgeries             =("surgeries_performed",           "sum"),
            total_additional_possible   =("additional_surgeries_possible", "sum"),
            most_common_procedure       =("most_common_procedure",         "first"),
            avg_procedure_duration_min  =("avg_procedure_duration_min",    "first"),
        )
        .sort_values("total_surgeries", ascending=False)
        .reset_index(drop=True)
    )
    return summary, csv_out


# ── Resource Utilization — renderer ──────────────────────────────────────────

def _render_resource_utilization():
    if not _data_available():
        _show_data_unavailable()
        return
    summary, _ = _load_resource_utilization_data()

    specialties  = summary["ProcedureSpecialtyDescription"].tolist()
    performed    = summary["total_surgeries"].tolist()
    additional   = summary["total_additional_possible"].tolist()
    total_cases  = sum(performed)
    total_extra  = sum(additional)

    st.caption(
        f"**{DATE_START}** to **{DATE_END}** · {total_cases:,} surgeries performed · "
        f"{total_extra:,} additional surgeries possible from recovered schedule time"
    )

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2 = st.columns(2)
    m1.metric("Total Surgeries Performed", f"{total_cases:,}")
    m2.metric("Additional Surgeries Possible", f"{total_extra:,}",
              help="High-count procedures that fit within recovered OR time across all days.")
    st.divider()

    # ── Total surgeries performed chart ───────────────────────────────────────
    st.markdown("#### Total Surgeries Performed by Specialty")
    fig1 = go.Figure(go.Bar(
        x=specialties,
        y=performed,
        marker_color=_BLUE,
        text=performed,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Surgeries Performed: %{y:,}<extra></extra>",
    ))
    fig1.update_layout(
        height=420,
        margin=dict(t=60, b=20, l=10, r=10),
        xaxis_title="Surgical Specialty",
        yaxis_title="Number of Surgeries",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f0f4f8"),
        xaxis=dict(tickfont=dict(size=11), tickangle=-20),
        yaxis=dict(gridcolor="rgba(149,165,166,0.2)", rangemode="tozero"),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Additional surgeries possible chart ───────────────────────────────────
    st.markdown("#### Additional Surgeries Possible from Recovered Schedule Time")
    st.caption(
        "Number of additional high-volume procedures that fit within OR time recovered "
        "on days where surgeries ran under their booked duration."
    )
    fig2 = go.Figure(go.Bar(
        x=specialties,
        y=additional,
        marker_color=_TEAL,
        text=additional,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Additional Possible: %{y:,}<extra></extra>",
    ))
    fig2.update_layout(
        height=420,
        margin=dict(t=60, b=20, l=10, r=10),
        xaxis_title="Surgical Specialty",
        yaxis_title="Additional Surgeries",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f0f4f8"),
        xaxis=dict(tickfont=dict(size=11), tickangle=-20),
        yaxis=dict(gridcolor="rgba(149,165,166,0.2)", rangemode="tozero"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Breakdown by specialty ────────────────────────────────────────────────
    st.markdown("#### Specialty Breakdown")
    ref = summary[["ProcedureSpecialtyDescription",
                   "avg_procedure_duration_min", "total_additional_possible"]].copy()
    ref.columns = ["Specialty", "Avg Procedure Duration (min)", "Additional Surgeries Possible"]
    ref["Avg Procedure Duration (min)"] = ref["Avg Procedure Duration (min)"].round(1)
    st.dataframe(ref, use_container_width=True, hide_index=True)
