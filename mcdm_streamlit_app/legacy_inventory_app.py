import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(
    page_title="MCDM Inventory Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = APP_DIR / "inventory_data.csv"
SCORED_DATA_PATH = APP_DIR / "inventory_data_scored_abc.csv"
RANDOM_STATE = 42


CSS = """
<style>
    :root {
        --bg: #f7f8fb;
        --panel: #ffffff;
        --ink: #1f2937;
        --muted: #6b7280;
        --line: #d9dee8;
        --accent: #0f766e;
        --accent-2: #2563eb;
        --warn: #b45309;
    }

    .stApp {
        background: var(--bg);
        color: var(--ink);
    }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
        color: #000000;
    }

    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] {
        color: #000000 !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="slider"] * {
        color: #000000 !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label,
    section[data-testid="stSidebar"] div[role="radiogroup"] span {
        color: #000000 !important;
    }

    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--ink);
    }

    .hero {
        padding: 1.2rem 0 0.8rem 0;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2.15rem;
        line-height: 1.1;
    }

    .hero p {
        color: var(--muted);
        font-size: 1rem;
        max-width: 920px;
        margin: 0;
    }

    .metric-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        min-height: 104px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.82rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: var(--ink);
        font-size: 1.7rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 0.35rem;
    }

    .callout {
        background: #eef6f4;
        border-left: 4px solid var(--accent);
        padding: 0.85rem 1rem;
        border-radius: 6px;
        color: #164e48;
        margin: 0.75rem 0 1rem 0;
    }

    .small-note {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.45;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        border-bottom: 1px solid var(--line);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 0.65rem 0.9rem;
    }
</style>
"""


st.markdown(CSS, unsafe_allow_html=True)


RISK_MAP = {"High": 0.47, "Normal": 0.35, "Low": 0.18}
DEMAND_FLUCTUATION_MAP = {
    "Increasing": 0.36,
    "Stable": 0.28,
    "Unknown": 0.20,
    "Decreasing": 0.16,
    "Ending": 0.00,
}
CONSIGNMENT_MAP = {"No": 0.80, "Yes": 0.20}
UNIT_SIZE_MAP = {"Large": 0.53, "Medium": 0.31, "Small": 0.13}


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def minmax(series):
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / span


@st.cache_data
def load_raw_data():
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data
def build_pipeline_outputs(abc_a_share=0.20, abc_b_share=0.30, weights=(1 / 3, 1 / 3, 1 / 3)):
    raw = load_raw_data()
    data = raw.copy()

    data["Risk_score"] = data["Risk"].map(RISK_MAP)
    data["Demand_fluctuation_score"] = data["Demand fluctuation"].map(DEMAND_FLUCTUATION_MAP)
    data["Consignment_score"] = data["Consignment stock"].map(CONSIGNMENT_MAP)
    data["Unit_size_score"] = data["Unit size"].map(UNIT_SIZE_MAP)

    data["Daily_usage_norm"] = minmax(data["Daily usage"])
    data["Average_stock_norm"] = minmax(data["Average stock"])
    data["Lead_time_norm"] = minmax(data["Lead time"])
    data["Unit_cost_norm"] = minmax(data["Unit cost"])

    data["Criticality"] = 0.78 * data["Risk_score"] + 0.22 * data["Demand_fluctuation_score"]
    data["Demand"] = 0.71 * data["Daily_usage_norm"] + 0.29 * data["Average_stock_norm"]
    data["Supply"] = 0.75 * data["Lead_time_norm"] + 0.25 * data["Consignment_score"]

    criteria = ["Criticality", "Demand", "Supply"]
    global_weights = np.array(weights, dtype=float)
    global_weights = global_weights / global_weights.sum()
    matrix = data[criteria].to_numpy(dtype=float)
    norms = np.sqrt((matrix**2).sum(axis=0))
    weighted = (matrix / norms) * global_weights
    ideal_positive = weighted.max(axis=0)
    ideal_negative = weighted.min(axis=0)
    d_positive = np.sqrt(((weighted - ideal_positive) ** 2).sum(axis=1))
    d_negative = np.sqrt(((weighted - ideal_negative) ** 2).sum(axis=1))
    data["TOPSIS_score"] = d_negative / (d_positive + d_negative)
    data["TOPSIS_rank"] = data["TOPSIS_score"].rank(ascending=False, method="first").astype(int)
    data["Article_ID"] = np.arange(1, len(data) + 1)

    ranking = data.sort_values("TOPSIS_score", ascending=False).reset_index(drop=True)
    limit_a = int(np.ceil(abc_a_share * len(ranking)))
    limit_b = int(np.ceil((abc_a_share + abc_b_share) * len(ranking)))
    ranking["Classe"] = np.where(
        ranking.index < limit_a,
        "A",
        np.where(ranking.index < limit_b, "B", "C"),
    )
    data["Classe"] = data["Article_ID"].map(ranking.set_index("Article_ID")["Classe"])

    weighted_df = pd.DataFrame(weighted, columns=[f"{c}_weighted_norm" for c in criteria])
    return data, ranking, weighted_df, global_weights, ideal_positive, ideal_negative


@st.cache_data
def train_models(data):
    feature_columns = [
        "Risk_score",
        "Demand_fluctuation_score",
        "Average_stock_norm",
        "Daily_usage_norm",
        "Unit_cost_norm",
        "Lead_time_norm",
        "Consignment_score",
        "Unit_size_score",
        "Criticality",
        "Demand",
        "Supply",
    ]
    X = data[feature_columns]
    y = data["Classe"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "SVM RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=10,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    rows = []
    trained = {}
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="macro", zero_division=0
        )
        rows.append(
            {
                "Modele": name,
                "Accuracy test": accuracy_score(y_test, pred),
                "Precision macro": precision,
                "Recall macro": recall,
                "F1 macro": f1,
                "F1 macro CV 5-fold": cross_val_score(
                    model, X, y, cv=5, scoring="f1_macro"
                ).mean(),
            }
        )
        trained[name] = model
        predictions[name] = pred

    results = pd.DataFrame(rows).sort_values("F1 macro", ascending=False)
    best_name = results.iloc[0]["Modele"]
    cm = pd.DataFrame(
        confusion_matrix(y_test, predictions[best_name], labels=["A", "B", "C"]),
        index=["Reel A", "Reel B", "Reel C"],
        columns=["Pred A", "Pred B", "Pred C"],
    )
    return results, cm, best_name, feature_columns


def fig_layout(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1f2937"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    return fig


def jev_request_for_item(data, article_id):
    row = data.loc[data["Article_ID"] == article_id].iloc[0]
    state = {
        "task": "Classify this inventory item into ABC class using the TOPSIS-based ABC rule.",
        "abc_rule": {
            "A": "Top 20% most important items by TOPSIS rank; high priority.",
            "B": "Next 30% by TOPSIS rank; medium priority.",
            "C": "Remaining 50% by TOPSIS rank; low priority.",
        },
        "thresholds_by_rank": {"A": "1 to 140", "B": "141 to 350", "C": "351 to 700"},
        "item": {
            "Article_ID": int(row["Article_ID"]),
            "Risk": row["Risk"],
            "Demand fluctuation": row["Demand fluctuation"],
            "Average stock": round(float(row["Average stock"]), 4),
            "Daily usage": round(float(row["Daily usage"]), 4),
            "Unit cost": round(float(row["Unit cost"]), 4),
            "Lead time": int(row["Lead time"]),
            "Consignment stock": row["Consignment stock"],
            "Unit size": row["Unit size"],
            "Criticality": round(float(row["Criticality"]), 4),
            "Demand": round(float(row["Demand"]), 4),
            "Supply": round(float(row["Supply"]), 4),
            "TOPSIS_score": round(float(row["TOPSIS_score"]), 4),
            "TOPSIS_rank": int(row["TOPSIS_rank"]),
        },
        "expected_class_from_app": row["Classe"],
    }
    return {
        "model": "typesafe/jev-1.13",
        "state": state,
        "questions": {
            "abc_class": {
                "type": "choice",
                "instructions": "Which ABC class should this item belong to under the given TOPSIS-based ABC rule?",
                "criteria": {
                    "A": "TOPSIS rank is between 1 and 140.",
                    "B": "TOPSIS rank is between 141 and 350.",
                    "C": "TOPSIS rank is between 351 and 700.",
                },
            },
            "high_priority": {
                "type": "noul",
                "instructions": "Is this a high-priority A-class item?",
                "criteria": {
                    "true": "The TOPSIS rank is between 1 and 140.",
                    "false": "The TOPSIS rank is greater than 140.",
                },
            },
        },
    }


with st.sidebar:
    st.title("MCDM Workflow")
    st.caption("Inventory classification from raw data to ABC labels")
    st.divider()
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Data",
            "Scoring",
            "Criteria",
            "TOPSIS",
            "ABC",
            "Machine Learning",
            "Fuzzy + Jev",
        ],
    )
    st.divider()
    st.subheader("Parameters")
    w_criticality = st.slider("Criticality weight", 0.05, 0.90, 0.333, 0.001)
    w_demand = st.slider("Demand weight", 0.05, 0.90, 0.333, 0.001)
    w_supply = st.slider("Supply weight", 0.05, 0.90, 0.334, 0.001)
    st.caption("Weights are normalized automatically.")
    a_share = st.slider("Class A share", 0.05, 0.40, 0.20, 0.01)
    b_share = st.slider("Class B share", 0.10, 0.60, 0.30, 0.01)

if a_share + b_share >= 0.95:
    st.sidebar.warning("A + B is high. Class C should keep a meaningful remainder.")

data, ranking, weighted_df, weights, ideal_positive, ideal_negative = build_pipeline_outputs(
    a_share, b_share, (w_criticality, w_demand, w_supply)
)

st.markdown(
    """
    <div class="hero">
        <h1>Inventory MCDM Analysis</h1>
        <p>From raw inventory data to normalized scores, aggregated criteria, TOPSIS ranking, ABC classes, machine learning and a compact Jev classification test.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Overview":
    cols = st.columns(4)
    with cols[0]:
        metric_card("Articles", f"{len(data):,}", "Rows in inventory_data.csv")
    with cols[1]:
        metric_card("Variables", f"{len(load_raw_data().columns)}", "Raw attributes")
    with cols[2]:
        metric_card("Best TOPSIS", f"{data['TOPSIS_score'].max():.3f}", "Closest to ideal")
    with cols[3]:
        metric_card("Class A", f"{(data['Classe'] == 'A').sum()}", "Highest priority")

    st.markdown('<div class="callout">The workflow converts qualitative variables into normalized scores, aggregates them into decision criteria, ranks items with TOPSIS, then turns the ranking into ABC labels.</div>', unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])
    with left:
        fig = px.scatter(
            data,
            x="TOPSIS_rank",
            y="TOPSIS_score",
            color="Classe",
            color_discrete_map={"A": "#0f766e", "B": "#2563eb", "C": "#b45309"},
            hover_data=["Article_ID", "Criticality", "Demand", "Supply"],
            title="TOPSIS score by rank",
        )
        st.plotly_chart(fig_layout(fig, 410), use_container_width=True)
    with right:
        counts = data["Classe"].value_counts().reindex(["A", "B", "C"]).reset_index()
        counts.columns = ["Classe", "Count"]
        fig = px.bar(
            counts,
            x="Classe",
            y="Count",
            color="Classe",
            color_discrete_map={"A": "#0f766e", "B": "#2563eb", "C": "#b45309"},
            title="ABC distribution",
        )
        st.plotly_chart(fig_layout(fig, 410), use_container_width=True)

    st.subheader("Top ranked articles")
    st.dataframe(
        ranking[
            ["Article_ID", "Risk", "Demand fluctuation", "Criticality", "Demand", "Supply", "TOPSIS_score", "TOPSIS_rank", "Classe"]
        ].head(12),
        use_container_width=True,
        hide_index=True,
    )

elif page == "Data":
    st.subheader("Available variables")
    raw = load_raw_data()
    variable_table = pd.DataFrame(
        {
            "Variable": raw.columns,
            "Nature": [
                "Qualitative ordinale",
                "Qualitative ordinale",
                "Quantitative continue",
                "Quantitative continue",
                "Quantitative continue",
                "Quantitative discrete",
                "Qualitative binaire",
                "Qualitative ordinale",
            ],
            "Type pandas": raw.dtypes.astype(str).values,
        }
    )
    st.dataframe(variable_table, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(raw, x="Daily usage", nbins=30, title="Daily usage distribution")
        st.plotly_chart(fig_layout(fig), use_container_width=True)
    with c2:
        fig = px.histogram(raw, x="Lead time", nbins=30, title="Lead time distribution")
        st.plotly_chart(fig_layout(fig), use_container_width=True)

    st.subheader("Raw data preview")
    st.dataframe(raw.head(30), use_container_width=True, hide_index=True)

elif page == "Scoring":
    st.subheader("Qualitative conversion rules")
    maps = [
        ("Risk", RISK_MAP),
        ("Demand fluctuation", DEMAND_FLUCTUATION_MAP),
        ("Consignment stock", CONSIGNMENT_MAP),
        ("Unit size", UNIT_SIZE_MAP),
    ]
    cols = st.columns(4)
    for col, (name, mapping) in zip(cols, maps):
        with col:
            st.markdown(f"**{name}**")
            st.dataframe(
                pd.DataFrame(mapping.items(), columns=["Modality", "Normalized score"]),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown('<div class="callout">This transformation lets verbal attributes enter numerical aggregation, distance calculations and machine learning models.</div>', unsafe_allow_html=True)
    st.subheader("Converted score preview")
    st.dataframe(
        data[
            [
                "Risk",
                "Risk_score",
                "Demand fluctuation",
                "Demand_fluctuation_score",
                "Consignment stock",
                "Consignment_score",
                "Unit size",
                "Unit_size_score",
            ]
        ].head(25),
        use_container_width=True,
        hide_index=True,
    )

elif page == "Criteria":
    st.subheader("Aggregated decision criteria")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Criticality", "0.78 Risk + 0.22 Fluctuation", "Risk exposure")
    with c2:
        metric_card("Demand", "0.71 Usage + 0.29 Stock", "Consumption pressure")
    with c3:
        metric_card("Supply", "0.75 Lead time + 0.25 Consignment", "Supply sensitivity")

    fig = px.scatter_3d(
        data,
        x="Criticality",
        y="Demand",
        z="Supply",
        color="Classe",
        color_discrete_map={"A": "#0f766e", "B": "#2563eb", "C": "#b45309"},
        hover_data=["Article_ID", "TOPSIS_score"],
        title="Items in the aggregated criteria space",
    )
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        data[["Article_ID", "Criticality", "Demand", "Supply", "Classe"]]
        .sort_values("Article_ID")
        .head(30),
        use_container_width=True,
        hide_index=True,
    )

elif page == "TOPSIS":
    st.subheader("TOPSIS decision matrix")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Criticality weight", f"{weights[0]:.3f}", "Normalized global weight")
    with c2:
        metric_card("Demand weight", f"{weights[1]:.3f}", "Normalized global weight")
    with c3:
        metric_card("Supply weight", f"{weights[2]:.3f}", "Normalized global weight")

    st.markdown('<div class="small-note">TOPSIS uses vector normalization, weighted normalized values, a positive ideal solution, a negative ideal solution and a closeness coefficient.</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        ideal_df = pd.DataFrame(
            {
                "Criterion": ["Criticality", "Demand", "Supply"],
                "Positive ideal": ideal_positive,
                "Negative ideal": ideal_negative,
            }
        )
        st.dataframe(ideal_df, use_container_width=True, hide_index=True)
    with right:
        fig = px.box(data, y="TOPSIS_score", color="Classe", title="TOPSIS score spread by class")
        st.plotly_chart(fig_layout(fig), use_container_width=True)

    st.subheader("Ranking")
    st.dataframe(
        ranking[
            ["Article_ID", "Criticality", "Demand", "Supply", "TOPSIS_score", "TOPSIS_rank", "Classe"]
        ].head(50),
        use_container_width=True,
        hide_index=True,
    )

elif page == "ABC":
    st.subheader("ABC classification")
    c1, c2, c3 = st.columns(3)
    class_counts = data["Classe"].value_counts().reindex(["A", "B", "C"])
    with c1:
        metric_card("A", int(class_counts["A"]), f"Top {a_share:.0%}")
    with c2:
        metric_card("B", int(class_counts["B"]), f"Next {b_share:.0%}")
    with c3:
        metric_card("C", int(class_counts["C"]), f"Remaining {1 - a_share - b_share:.0%}")

    fig = px.histogram(
        data,
        x="TOPSIS_score",
        color="Classe",
        nbins=36,
        color_discrete_map={"A": "#0f766e", "B": "#2563eb", "C": "#b45309"},
        title="TOPSIS score distribution by ABC class",
    )
    st.plotly_chart(fig_layout(fig, 430), use_container_width=True)

    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scored ABC data",
        csv,
        "inventory_data_scored_abc_from_app.csv",
        "text/csv",
    )
    st.dataframe(
        data.sort_values("TOPSIS_rank")[
            ["Article_ID", "TOPSIS_score", "TOPSIS_rank", "Classe", "Risk", "Demand fluctuation", "Lead time"]
        ],
        use_container_width=True,
        hide_index=True,
    )

elif page == "Machine Learning":
    st.subheader("Automatic ABC classification")
    results, cm, best_name, feature_columns = train_models(data)
    c1, c2, c3 = st.columns(3)
    best_row = results.iloc[0]
    with c1:
        metric_card("Best model", best_name, "Highest test F1 macro")
    with c2:
        metric_card("Accuracy", f"{best_row['Accuracy test']:.3f}", "Test split")
    with c3:
        metric_card("F1 macro", f"{best_row['F1 macro']:.3f}", "Class-balanced view")

    fig = px.bar(
        results,
        x="Modele",
        y=["Accuracy test", "F1 macro", "F1 macro CV 5-fold"],
        barmode="group",
        title="Model performance comparison",
    )
    st.plotly_chart(fig_layout(fig, 430), use_container_width=True)

    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.dataframe(results, use_container_width=True, hide_index=True)
    with c2:
        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Confusion matrix: {best_name}",
        )
        st.plotly_chart(fig_layout(fig, 360), use_container_width=True)

    st.markdown('<div class="callout">Because the labels are generated from TOPSIS + ABC, the models are learning to reproduce a decision rule. These scores do not prove external business truth.</div>', unsafe_allow_html=True)
    st.caption("Features used")
    st.write(", ".join(feature_columns))

elif page == "Fuzzy + Jev":
    st.subheader("Fuzzy extension")
    st.markdown(
        """
        A fuzzy version replaces crisp linguistic scores with triangular fuzzy numbers `(l, m, u)`.
        The workflow becomes: fuzzy conversion, fuzzy aggregation, fuzzy TOPSIS, then defuzzification.
        """
    )
    fuzzy_risk = pd.DataFrame(
        {
            "Risk": ["High", "Normal", "Low"],
            "l": [0.40, 0.28, 0.10],
            "m": [0.47, 0.35, 0.18],
            "u": [0.55, 0.42, 0.25],
        }
    )
    st.dataframe(fuzzy_risk, use_container_width=True, hide_index=True)

    st.subheader("Jev item classification test")
    article_id = st.selectbox(
        "Choose an article",
        options=data.sort_values("TOPSIS_rank")["Article_ID"].tolist(),
        index=0,
    )
    request_body = jev_request_for_item(data, int(article_id))
    item = request_body["state"]["item"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Article", item["Article_ID"], "Selected item")
    with c2:
        metric_card("TOPSIS rank", item["TOPSIS_rank"], "Lower is better")
    with c3:
        metric_card("TOPSIS score", f"{item['TOPSIS_score']:.3f}", "Closeness coefficient")
    with c4:
        metric_card("Expected class", request_body["state"]["expected_class_from_app"], "From app rule")

    with st.expander("Show Decisions API payload", expanded=False):
        st.json(request_body)

    api_key = st.text_input(
        "OpenRouter API key",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
        help="Optional. Leave empty to only preview the request.",
    )
    if st.button("Run Jev classification"):
        if not api_key:
            st.warning("No API key provided. The request preview above is ready to send.")
        else:
            try:
                response = requests.post(
                    "https://openrouter.ai/api/alpha/decisions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_body,
                    timeout=30,
                )
                response.raise_for_status()
                st.success("Jev response received.")
                st.json(response.json())
            except requests.RequestException as exc:
                st.error(f"Jev request failed: {exc}")
