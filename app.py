"""
app.py  —  Surgery Duration Predictor
Streamlit dashboard.
"""

import pickle
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Surgery Duration Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 0.9rem; color: #555; }
    .metric-card p  { margin: 0; font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }
    .predict-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        color: white;
    }
    .predict-box h1 { font-size: 3.5rem; margin: 0.2rem 0; }
    .predict-box p  { color: #aac; margin: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load artifacts ────────────────────────────────────────────────────────────
MODELS_DIR = "models"

CUSTOM_STOP_WORDS = [
    "removal", "extraction", "total", "non", "diagnostic", "performed",
    "procedure", "status", "post", "bilateral", "left", "right", "well",
    "tolerated", "bx", "fx", "tx", "dx", "hx", "sx", "ex",
]


@st.cache_resource(show_spinner="Loading model …")
def load_artifacts():
    def _load(name):
        with open(f"{MODELS_DIR}/{name}", "rb") as f:
            return pickle.load(f)

    return (
        _load("rf_model.pkl"),
        _load("tfidf.pkl"),
        _load("svd.pkl"),
        _load("feature_columns.pkl"),
        _load("categorical_values.pkl"),
        _load("model_stats.pkl"),
        _load("test_results.pkl"),
    )


try:
    rf_model, tfidf, svd, feature_columns, cat_values, model_stats, test_results = (
        load_artifacts()
    )
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_text(txt: str) -> str:
    txt = str(txt).lower()
    txt = re.sub(r"[^a-z ]", " ", txt)
    return " ".join(w for w in txt.split() if w not in CUSTOM_STOP_WORDS)


def predict(surgical_priority, patient_type, room, specialty, description):
    cleaned = clean_text(description)
    tfidf_vec = tfidf.transform([cleaned])
    svd_vec = svd.transform(tfidf_vec)

    svd_df = pd.DataFrame(svd_vec, columns=[f"SVD_{i}" for i in range(svd_vec.shape[1])])
    X_num = pd.DataFrame({"SurgicalPriority": [surgical_priority]})
    cat_df = pd.DataFrame(
        {"PatientType": [patient_type], "Roomdescription": [room],
         "ProcedureSpecialtyDescription": [specialty]}
    )
    X_cat = pd.get_dummies(cat_df, drop_first=True, dtype=int)
    X = pd.concat([X_num, X_cat, svd_df], axis=1).reindex(
        columns=feature_columns, fill_value=0
    )
    return float(np.exp(rf_model.predict(X)[0]))


def fmt_duration(minutes: float) -> str:
    h, m = int(minutes // 60), int(minutes % 60)
    return f"{h}h {m:02d}m" if h > 0 else f"{m} min"


# ── Sidebar — inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 Surgery Duration\nPredictor")
    st.caption("Department of Management Sciences · University of Waterloo")
    st.divider()

    if not model_loaded:
        st.error(
            "**Model not found.**\n\n"
            "Run `python train_and_save.py` first to train and save the model."
        )
        st.stop()

    st.subheader("Procedure Details")

    patient_type = st.selectbox("Patient Type", cat_values["PatientType"])
    room = st.selectbox("Operating Room", cat_values["Roomdescription"])
    specialty = st.selectbox(
        "Procedure Specialty", cat_values["ProcedureSpecialtyDescription"]
    )
    surgical_priority = st.slider(
        "Surgical Priority", min_value=1, max_value=5, value=2,
        help="1 = most urgent, 5 = elective"
    )
    description = st.text_area(
        "Procedure Description",
        placeholder="e.g. laparoscopic cholecystectomy with intraoperative cholangiogram",
        height=110,
    )
    booked = st.number_input(
        "Booked Duration (min) — optional",
        min_value=0, max_value=600, value=0, step=5,
        help="Enter the originally scheduled time to compare against the model."
    )

    run = st.button("Predict Duration", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
tab_predict, tab_model, tab_about = st.tabs(
    ["🔮 Prediction", "📊 Model Performance", "ℹ️ About"]
)

# ── Tab 1: Prediction ─────────────────────────────────────────────────────────
with tab_predict:
    if not run:
        st.info(
            "Fill in the procedure details in the sidebar and click **Predict Duration**."
        )

        # Show sample stats so the page isn't blank
        st.subheader("Dataset Snapshot")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Cases", f"{model_stats['train_size'] + model_stats['test_size']:,}")
        c2.metric("Avg Duration", f"{model_stats['y_mean']:.0f} min")
        c3.metric("Median Duration", f"{model_stats['y_median']:.0f} min")
        c4.metric("Std Deviation", f"{model_stats['y_std']:.0f} min")

    else:
        if not description.strip():
            st.warning("Please enter a procedure description in the sidebar.")
        else:
            with st.spinner("Running prediction …"):
                pred = predict(
                    surgical_priority, patient_type, room, specialty, description
                )

            # ── Prediction card
            st.markdown(
                f"""
                <div class="predict-box">
                    <p>Predicted Surgery Duration</p>
                    <h1>{pred:.0f} <span style="font-size:1.8rem">min</span></h1>
                    <p>{fmt_duration(pred)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

            if booked and booked > 0:
                st.subheader("Model vs. Booked Duration")

                diff_booked = pred - booked
                pct_diff = (diff_booked / booked) * 100

                m1, m2 = st.columns(2)
                m1.metric("Model Prediction", f"{pred:.0f} min")
                m2.metric(
                    "Booked Duration", f"{booked} min",
                    delta=f"{diff_booked:+.0f} min ({pct_diff:+.1f}%)",
                    delta_color="inverse",
                )

                if abs(pct_diff) < 10:
                    st.success("Model closely matches the booked duration (within 10%).")
                elif diff_booked > 0:
                    st.warning(
                        f"Model predicts **{abs(diff_booked):.0f} min more** than booked. "
                        "The schedule may be tight."
                    )
                else:
                    st.info(
                        f"Model predicts **{abs(diff_booked):.0f} min less** than booked. "
                        "There may be scheduling slack."
                    )

# ── Tab 2: Model Performance ──────────────────────────────────────────────────
with tab_model:
    st.subheader("Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{model_stats['r2']:.3f}", help="Closer to 1.0 is better.")
    c2.metric("RMSE", f"{model_stats['rmse']:.1f} min", help="Root Mean Squared Error.")
    c3.metric("MAE", f"{model_stats['mae']:.1f} min", help="Mean Absolute Error.")
    c4.metric("Test Cases", f"{model_stats['test_size']:,}")

    st.divider()

    # ── Plot 1: Actual vs Predicted ───────────────────────────────────────────
    st.subheader("Actual vs. Predicted Duration")

    sample = test_results.sample(n=min(2000, len(test_results)), random_state=42)
    cap = float(np.percentile(test_results["actual"], 99))
    sample = sample[(sample["actual"] <= cap) & (sample["predicted"] <= cap)]

    fig1 = px.scatter(
        sample, x="actual", y="predicted",
        opacity=0.35,
        labels={"actual": "Actual Duration (min)", "predicted": "Predicted Duration (min)"},
        color_discrete_sequence=["#4A90D9"],
    )
    fig1.add_shape(
        type="line", x0=0, y0=0, x1=cap, y1=cap,
        line=dict(color="#E74C3C", width=2, dash="dash"),
    )
    fig1.add_annotation(
        x=cap * 0.85, y=cap * 0.92, text="Perfect prediction",
        showarrow=False, font=dict(color="#E74C3C", size=12),
    )
    fig1.update_layout(height=420, margin=dict(t=20, b=20))
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Red dashed line = perfect prediction. Points above the line are under-predicted (surgery ran longer than expected).")

    st.divider()

    # ── Plot 2: Model vs Booked Duration ──────────────────────────────────────
    st.subheader("Model vs. Booked Duration")

    comparison = pd.DataFrame({
        "Source": ["Model Prediction", "Booked Duration"],
        "RMSE (min)": [model_stats["rmse"], model_stats["booked_rmse"]],
        "MAE (min)":  [model_stats["mae"],  model_stats["booked_mae"]],
    })

    col_rmse, col_mae = st.columns(2)

    with col_rmse:
        fig2a = px.bar(
            comparison, x="Source", y="RMSE (min)",
            color="Source", color_discrete_sequence=["#4A90D9", "#95A5A6"],
            text_auto=".1f",
        )
        fig2a.update_layout(showlegend=False, height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig2a, use_container_width=True)

    with col_mae:
        fig2b = px.bar(
            comparison, x="Source", y="MAE (min)",
            color="Source", color_discrete_sequence=["#4A90D9", "#95A5A6"],
            text_auto=".1f",
        )
        fig2b.update_layout(showlegend=False, height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig2b, use_container_width=True)

    rmse_imp = (1 - model_stats["rmse"] / model_stats["booked_rmse"]) * 100
    mae_imp  = (1 - model_stats["mae"]  / model_stats["booked_mae"])  * 100
    st.caption(
        f"The model reduces RMSE by **{rmse_imp:.1f}%** and MAE by **{mae_imp:.1f}%** "
        "compared to using the originally booked duration as a prediction."
    )

    st.divider()

    # ── Plot 3: Residuals Distribution ────────────────────────────────────────
    st.subheader("Residuals Distribution")

    residuals = test_results["residual"].clip(
        lower=float(np.percentile(test_results["residual"], 1)),
        upper=float(np.percentile(test_results["residual"], 99)),
    )
    fig3 = px.histogram(
        residuals, nbins=80,
        labels={"value": "Residual (Predicted − Actual, min)", "count": "Count"},
        color_discrete_sequence=["#4A90D9"],
    )
    fig3.add_vline(x=0, line_color="#E74C3C", line_dash="dash", line_width=2,
                   annotation_text="Zero error", annotation_position="top right")
    fig3.update_layout(height=380, margin=dict(t=20, b=20), showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    under = (test_results["residual"] < 0).mean() * 100
    over  = (test_results["residual"] > 0).mean() * 100
    st.caption(
        f"**{under:.1f}%** of cases are under-predicted (surgery ran longer than expected) · "
        f"**{over:.1f}%** are over-predicted."
    )

    st.divider()

    # ── Plot 4: MAE by Specialty ──────────────────────────────────────────────
    st.subheader("Prediction Error by Specialty")

    specialty_mae = (
        test_results.groupby("specialty")
        .apply(lambda g: mean_absolute_error(g["actual"], g["predicted"]))
        .reset_index()
        .rename(columns={0: "MAE (min)", "specialty": "Specialty"})
        .sort_values("MAE (min)", ascending=True)
    )

    fig4 = px.bar(
        specialty_mae, x="MAE (min)", y="Specialty",
        orientation="h",
        text_auto=".1f",
        color="MAE (min)",
        color_continuous_scale=["#4A90D9", "#E74C3C"],
    )
    fig4.update_layout(
        height=max(350, len(specialty_mae) * 28),
        margin=dict(t=20, b=20),
        coloraxis_showscale=False,
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig4, use_container_width=True)
    best  = specialty_mae.iloc[0]
    worst = specialty_mae.iloc[-1]
    st.caption(
        f"Best: **{best['Specialty']}** ({best['MAE (min)']:.1f} min MAE) · "
        f"Most challenging: **{worst['Specialty']}** ({worst['MAE (min)']:.1f} min MAE)"
    )

    st.divider()
    st.subheader("Model Configuration")
    config = {
        "Algorithm": "Random Forest Regressor",
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "Target transform": "log(ActualDurationMinutes)",
        "Text features": "TF-IDF bigrams → TruncatedSVD (150 components)",
        "Train / Test split": "80 / 20",
        "Training samples": f"{model_stats['train_size']:,}",
    }
    st.table(pd.DataFrame(config.items(), columns=["Parameter", "Value"]))

# ── Tab 3: About ──────────────────────────────────────────────────────────────
with tab_about:
    st.subheader("About This Project")
    st.markdown(
        """
        **Surgery Duration Predictor** is a machine learning application developed as part of
        the research conducted in Department of Management Sciences at the **University of Waterloo**.

        ### Problem
        Inaccurate surgery time estimates lead to OR scheduling inefficiencies, patient wait
        times, and unnecessary costs. Most hospitals rely on surgeon-reported booked durations
        or historical averages which are often systematically biased.

        ### Approach
        A **Random Forest regression** model was trained on ~35,000 historical surgical cases
        from a Canadian teaching hospital. The pipeline combines:

        | Feature type | Details |
        |---|---|
        | Surgical priority | Numeric urgency score (1–5) |
        | Patient / room / specialty | One-hot encoded categorical features |
        | Procedure description | TF-IDF bigrams compressed to 150 SVD components |

        ### Target
        Actual OR time in minutes, **log-transformed** during training to reduce right-skew.
        Predictions are exponentiated back to minutes.

        ### Key Findings
        - The model outperforms booked duration benchmarks on test data.
        - Most outlier cases are **under-predicted** (surgery ran longer than expected).
        - Procedure description text (SVD embeddings) is the strongest predictor group.

        ---
        **Author:** Gatik Gola · University of Waterloo
        """
    )
