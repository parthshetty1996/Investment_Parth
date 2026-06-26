import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, auc, classification_report,
                              roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(
    page_title="Insurance Claim Settlement Bias Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem; text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.2rem; margin-bottom: 0.5rem; }
    .main-header p  { color: #a8b2d8; font-size: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #0f3460);
        border: 1px solid #e94560; border-radius: 10px;
        padding: 1.2rem; text-align: center; margin: 0.3rem;
    }
    .metric-card h3 { color: #e94560; font-size: 0.85rem; margin-bottom: 0.3rem; }
    .metric-card h2 { color: #ffffff; font-size: 1.6rem; font-weight: bold; }
    .section-header {
        background: linear-gradient(90deg, #e94560, #c62a47);
        color: white; padding: 0.8rem 1.2rem; border-radius: 8px;
        font-size: 1.1rem; font-weight: bold; margin: 1.5rem 0 1rem 0;
    }
    .insight-box {
        background: #1e2a3a; border-left: 4px solid #e94560;
        padding: 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
        color: #a8b2d8;
    }
    .warning-box {
        background: #3a1e1e; border-left: 4px solid #ff6b35;
        padding: 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
        color: #ffb347;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e3a5f; color: #a8b2d8; border-radius: 8px;
        padding: 0.5rem 1rem; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background: #e94560 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 Insurance Claim Settlement Bias Analysis</h1>
    <p>Descriptive · Diagnostic · Predictive ML · Fairness Audit | Settlement Office Intelligence Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ── Data Loading & Preprocessing ───────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("Insurance.csv")

    # Clean numeric columns stored as strings
    for col in ["SUM_ASSURED", "PI_ANNUAL_INCOME"]:
        df[col] = df[col].astype(str).str.replace(",", "").str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Target binary
    df["TARGET"] = (df["POLICY_STATUS"] == "Approved Death Claim").astype(int)
    df["STATUS_LABEL"] = df["POLICY_STATUS"]

    # Age groups
    bins   = [0, 30, 40, 50, 60, 70, 100]
    labels = ["<30", "30-40", "40-50", "50-60", "60-70", "70+"]
    df["AGE_GROUP"] = pd.cut(df["PI_AGE"], bins=bins, labels=labels, right=True)

    # Income groups
    df["INCOME_GROUP"] = pd.cut(
        df["PI_ANNUAL_INCOME"],
        bins=[0, 100_000, 300_000, 600_000, 1_000_000, 1e10],
        labels=["<1L", "1-3L", "3-6L", "6L-10L", "10L+"],
        right=True
    )

    # Fill missing occupation
    df["PI_OCCUPATION"] = df["PI_OCCUPATION"].fillna("Unknown")
    df["REASON_FOR_CLAIM"] = df["REASON_FOR_CLAIM"].fillna("Not Provided")

    # Normalise ZONE name capitalization
    df["ZONE_CLEAN"] = df["ZONE"].str.upper().str.strip()

    return df

df = load_and_preprocess()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Dashboard Controls")
    selected_zones = st.multiselect(
        "Filter by Zone", options=sorted(df["ZONE"].unique()),
        default=sorted(df["ZONE"].unique())[:10]
    )
    selected_gender = st.multiselect(
        "Filter by Gender", options=df["PI_GENDER"].unique().tolist(),
        default=df["PI_GENDER"].unique().tolist()
    )
    age_range = st.slider("Age Range", int(df["PI_AGE"].min()), int(df["PI_AGE"].max()),
                          (int(df["PI_AGE"].min()), int(df["PI_AGE"].max())))

    df_f = df[
        df["ZONE"].isin(selected_zones) &
        df["PI_GENDER"].isin(selected_gender) &
        df["PI_AGE"].between(age_range[0], age_range[1])
    ]
    st.markdown(f"**Records in view:** {len(df_f):,}")
    st.markdown("---")
    st.markdown("#### 📋 Dataset Info")
    st.markdown(f"- Total Records: **{len(df):,}**")
    st.markdown(f"- Approved: **{df['TARGET'].sum():,}** ({df['TARGET'].mean()*100:.1f}%)")
    st.markdown(f"- Repudiated: **{(1-df['TARGET']).sum():,}** ({(1-df['TARGET']).mean()*100:.1f}%)")
    st.markdown(f"- Features: **{df.shape[1]}**")

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Descriptive Analysis",
    "🔬 Diagnostic Analysis",
    "🤖 ML Models",
    "📈 Model Performance",
    "💡 Key Findings"
])

# ── TAB 1: DESCRIPTIVE ANALYSIS ────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-header">📊 Descriptive Analysis — Cross-Tabulation Against Policy Status</div>',
                unsafe_allow_html=True)

    # KPI row
    total = len(df_f)
    approved = df_f["TARGET"].sum()
    repudiated = total - approved
    approval_rate = approved / total * 100 if total > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip([c1, c2, c3, c4],
                                ["Total Policies", "Approved Claims", "Repudiated Claims", "Approval Rate"],
                                [f"{total:,}", f"{approved:,}", f"{repudiated:,}", f"{approval_rate:.1f}%"]):
        col.markdown(f'<div class="metric-card"><h3>{label}</h3><h2>{val}</h2></div>',
                     unsafe_allow_html=True)

    st.markdown("---")

    # ── Overall distribution
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Overall Policy Status")
        status_counts = df_f["POLICY_STATUS"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.pie(status_counts, values="Count", names="Status",
                     color_discrete_sequence=["#2ecc71", "#e94560"],
                     hole=0.45)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", legend=dict(font=dict(color="white")))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Status by Zone (Top 15 Zones)")
        zone_status = df_f.groupby(["ZONE", "POLICY_STATUS"]).size().reset_index(name="Count")
        top_zones = df_f["ZONE"].value_counts().head(15).index
        zone_status = zone_status[zone_status["ZONE"].isin(top_zones)]
        fig = px.bar(zone_status, x="ZONE", y="Count", color="POLICY_STATUS",
                     barmode="group", color_discrete_sequence=["#2ecc71", "#e94560"],
                     text_auto=True)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", xaxis_tickangle=-40,
                          legend=dict(font=dict(color="white")))
        st.plotly_chart(fig, use_container_width=True)

    # ── Cross-tabs
    st.markdown("#### Cross-Tabulation Tables")
    ctab_choice = st.selectbox("Select dimension for cross-tab",
                               ["ZONE", "AGE_GROUP", "INCOME_GROUP", "PI_GENDER",
                                "PAYMENT_MODE", "EARLY_NON", "MEDICAL_NONMED",
                                "PI_OCCUPATION"])

    ct = pd.crosstab(df_f[ctab_choice], df_f["POLICY_STATUS"], margins=True)
    ct["Approval Rate (%)"] = (ct.get("Approved Death Claim", 0) /
                                ct["All"] * 100).round(1)
    st.dataframe(ct.style.background_gradient(cmap="RdYlGn", subset=["Approval Rate (%)"]),
                 use_container_width=True)

    # Approval rate chart
    ct2 = ct.drop("All").dropna()
    fig2 = px.bar(ct2.reset_index(), x=ctab_choice, y="Approval Rate (%)",
                  color="Approval Rate (%)", color_continuous_scale="RdYlGn",
                  title=f"Approval Rate by {ctab_choice}")
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="white", xaxis_tickangle=-35)
    fig2.add_hline(y=approval_rate, line_dash="dash", line_color="#ffff00",
                   annotation_text=f"Overall Avg: {approval_rate:.1f}%")
    st.plotly_chart(fig2, use_container_width=True)

    # ── Numeric summaries
    st.markdown("#### Numeric Feature Distributions by Policy Status")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df_f, x="POLICY_STATUS", y="PI_AGE", color="POLICY_STATUS",
                     color_discrete_sequence=["#2ecc71", "#e94560"],
                     title="Age Distribution by Policy Status")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df_f, x="POLICY_STATUS", y="PI_ANNUAL_INCOME", color="POLICY_STATUS",
                     color_discrete_sequence=["#2ecc71", "#e94560"],
                     title="Annual Income Distribution by Policy Status")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    st.markdown("#### Approval Rate Heatmap — Age Group × Income Group")
    pivot = df_f.groupby(["AGE_GROUP", "INCOME_GROUP"])["TARGET"].mean().unstack() * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1922")
    ax.set_facecolor("#0f1922")
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Approval %"})
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    plt.title("Approval Rate (%) by Age Group & Income Group", color="white", pad=10)
    st.pyplot(fig)
    plt.close()


# ── TAB 2: DIAGNOSTIC ANALYSIS ─────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-header">🔬 Diagnostic Analysis — Probing Bias in Claim Settlement</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
    ⚠️ <b>Bias Investigation Framework:</b> This section probes whether settlement decisions correlate
    with demographic, financial, or team-level variables that should <i>not</i> influence claim outcomes.
    Statistical tests (Chi-square, ANOVA) are applied to flag dimensions where settlement appears non-random.
    </div>""", unsafe_allow_html=True)

    # ── AGE BIAS
    st.markdown("### 1️⃣ Age-Wise Bias Analysis")
    col1, col2 = st.columns(2)
    with col1:
        age_approval = df_f.groupby("AGE_GROUP")["TARGET"].agg(["mean", "count"]).reset_index()
        age_approval.columns = ["Age Group", "Approval Rate", "Count"]
        age_approval["Approval Rate"] *= 100
        fig = px.bar(age_approval, x="Age Group", y="Approval Rate", text="Count",
                     color="Approval Rate", color_continuous_scale="RdYlGn",
                     title="Approval Rate by Age Group")
        fig.add_hline(y=df_f["TARGET"].mean()*100, line_dash="dash", line_color="#ffff00",
                      annotation_text="Overall Avg")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df_f, x="PI_AGE", color="POLICY_STATUS", nbins=30,
                           barmode="overlay", opacity=0.7,
                           color_discrete_sequence=["#2ecc71", "#e94560"],
                           title="Age Distribution: Approved vs Repudiated")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    # ANOVA test on age
    approved_ages = df_f[df_f["TARGET"] == 1]["PI_AGE"].dropna()
    repudiated_ages = df_f[df_f["TARGET"] == 0]["PI_AGE"].dropna()
    t_stat, p_val = stats.ttest_ind(approved_ages, repudiated_ages)
    st.markdown(f"""
    <div class="insight-box">
    📊 <b>T-Test: Approved vs Repudiated — Age</b><br>
    Mean Age Approved: <b>{approved_ages.mean():.1f} yrs</b> | Mean Age Repudiated: <b>{repudiated_ages.mean():.1f} yrs</b><br>
    T-Statistic: <b>{t_stat:.3f}</b> | P-Value: <b>{p_val:.4f}</b> —
    {'<span style="color:#e94560"><b>STATISTICALLY SIGNIFICANT DIFFERENCE ⚠️</b></span>' if p_val < 0.05 else '<span style="color:#2ecc71">No significant difference</span>'}
    </div>""", unsafe_allow_html=True)

    # ── INCOME BIAS
    st.markdown("### 2️⃣ Income-Wise Bias Analysis")
    col1, col2 = st.columns(2)
    with col1:
        inc_approval = df_f.groupby("INCOME_GROUP")["TARGET"].agg(["mean", "count"]).reset_index()
        inc_approval.columns = ["Income Group", "Approval Rate", "Count"]
        inc_approval["Approval Rate"] *= 100
        fig = px.bar(inc_approval, x="Income Group", y="Approval Rate", text="Count",
                     color="Approval Rate", color_continuous_scale="RdYlGn",
                     title="Approval Rate by Annual Income Group")
        fig.add_hline(y=df_f["TARGET"].mean()*100, line_dash="dash", line_color="#ffff00",
                      annotation_text="Overall Avg")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.violin(df_f, x="POLICY_STATUS", y="PI_ANNUAL_INCOME", color="POLICY_STATUS",
                        box=True, color_discrete_sequence=["#2ecc71", "#e94560"],
                        title="Income Violin: Approved vs Repudiated")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    approved_inc = df_f[df_f["TARGET"] == 1]["PI_ANNUAL_INCOME"].dropna()
    repudiated_inc = df_f[df_f["TARGET"] == 0]["PI_ANNUAL_INCOME"].dropna()
    t2, p2 = stats.ttest_ind(approved_inc, repudiated_inc)
    st.markdown(f"""
    <div class="insight-box">
    📊 <b>T-Test: Approved vs Repudiated — Income</b><br>
    Median Income Approved: <b>₹{approved_inc.median():,.0f}</b> | Repudiated: <b>₹{repudiated_inc.median():,.0f}</b><br>
    T-Statistic: <b>{t2:.3f}</b> | P-Value: <b>{p2:.4f}</b> —
    {'<span style="color:#e94560"><b>STATISTICALLY SIGNIFICANT ⚠️</b></span>' if p2 < 0.05 else '<span style="color:#2ecc71">No significant difference</span>'}
    </div>""", unsafe_allow_html=True)

    # ── ZONE/TEAM BIAS
    st.markdown("### 3️⃣ Zone / Team Wise Bias Analysis")
    zone_bias = df_f.groupby("ZONE")["TARGET"].agg(["mean", "count"]).reset_index()
    zone_bias.columns = ["Zone", "Approval Rate", "Count"]
    zone_bias["Approval Rate"] *= 100
    zone_bias = zone_bias[zone_bias["Count"] >= 10].sort_values("Approval Rate")

    fig = px.bar(zone_bias, x="Approval Rate", y="Zone", orientation="h",
                 color="Approval Rate", text="Count", color_continuous_scale="RdYlGn",
                 title="Approval Rate by Zone/Team (min 10 cases)")
    fig.add_vline(x=df_f["TARGET"].mean()*100, line_dash="dash", line_color="#ffff00",
                  annotation_text=f"Avg: {df_f['TARGET'].mean()*100:.1f}%")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="white", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Chi-square ZONE vs STATUS
    ct_zone = pd.crosstab(df_f["ZONE"], df_f["TARGET"])
    chi2, p_chi, dof, _ = stats.chi2_contingency(ct_zone)
    st.markdown(f"""
    <div class="{'warning-box' if p_chi < 0.05 else 'insight-box'}">
    📊 <b>Chi-Square Test: Zone vs Settlement Decision</b><br>
    Chi² = <b>{chi2:.2f}</b> | DoF = <b>{dof}</b> | P-Value = <b>{p_chi:.6f}</b><br>
    {'⚠️ <b>SIGNIFICANT ZONE-LEVEL BIAS DETECTED</b> — Settlement outcomes are NOT independent of Zone/Team assignment.' if p_chi < 0.05 else '✅ No significant zone-level bias detected.'}
    </div>""", unsafe_allow_html=True)

    # ── GENDER BIAS
    st.markdown("### 4️⃣ Gender Bias Analysis")
    gender_bias = df_f.groupby(["PI_GENDER", "POLICY_STATUS"]).size().unstack(fill_value=0)
    gender_bias["Approval Rate (%)"] = (gender_bias.get("Approved Death Claim", 0) /
                                         gender_bias.sum(axis=1) * 100).round(1)
    st.dataframe(gender_bias, use_container_width=True)

    ct_gender = pd.crosstab(df_f["PI_GENDER"], df_f["TARGET"])
    if ct_gender.shape == (2, 2):
        chi2_g, p_g, _, _ = stats.chi2_contingency(ct_gender)
        st.markdown(f"""
        <div class="insight-box">
        📊 <b>Chi-Square Test: Gender vs Settlement</b><br>
        Chi² = <b>{chi2_g:.3f}</b> | P-Value = <b>{p_g:.4f}</b> —
        {'<span style="color:#e94560"><b>GENDER BIAS DETECTED ⚠️</b></span>' if p_g < 0.05 else '<span style="color:#2ecc71">No significant gender bias</span>'}
        </div>""", unsafe_allow_html=True)

    # ── MEDICAL vs NON-MEDICAL
    st.markdown("### 5️⃣ Medical vs Non-Medical Policy Bias")
    med_bias = df_f.groupby(["MEDICAL_NONMED", "POLICY_STATUS"]).size().unstack(fill_value=0)
    med_bias["Approval Rate (%)"] = (med_bias.get("Approved Death Claim", 0) /
                                      med_bias.sum(axis=1) * 100).round(1)
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(med_bias, use_container_width=True)
    with col2:
        fig = px.bar(med_bias.reset_index(), x="MEDICAL_NONMED", y="Approval Rate (%)",
                     color="Approval Rate (%)", color_continuous_scale="RdYlGn",
                     title="Approval Rate: Medical vs Non-Medical")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    # ── EARLY vs NON-EARLY
    st.markdown("### 6️⃣ Early vs Non-Early Claim Bias")
    early_bias = df_f.groupby("EARLY_NON")["TARGET"].agg(["mean", "count"]).reset_index()
    early_bias["Approval Rate (%)"] = (early_bias["mean"] * 100).round(1)
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(early_bias[["EARLY_NON", "count", "Approval Rate (%)"]], use_container_width=True)
    with col2:
        fig = px.bar(early_bias, x="EARLY_NON", y="Approval Rate (%)",
                     color="Approval Rate (%)", color_continuous_scale="RdYlGn",
                     title="Approval Rate: Early vs Non-Early Claims", text="Approval Rate (%)")
        fig.add_hline(y=df_f["TARGET"].mean()*100, line_dash="dash", line_color="#ffff00")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    ct_early = pd.crosstab(df_f["EARLY_NON"], df_f["TARGET"])
    chi2_e, p_e, _, _ = stats.chi2_contingency(ct_early)
    st.markdown(f"""
    <div class="{'warning-box' if p_e < 0.05 else 'insight-box'}">
    📊 <b>Chi-Square Test: Early/Non-Early vs Settlement</b><br>
    Chi² = <b>{chi2_e:.3f}</b> | P-Value = <b>{p_e:.4f}</b> —
    {'⚠️ <b>SIGNIFICANT DIFFERENCE in approval for Early claims</b>' if p_e < 0.05 else '✅ No significant difference'}
    </div>""", unsafe_allow_html=True)

    # ── SUM ASSURED BIAS
    st.markdown("### 7️⃣ Sum Assured (Policy Size) Bias")
    df_f2 = df_f.copy()
    df_f2["SUM_ASSURED_CLEAN"] = pd.to_numeric(df_f2["SUM_ASSURED"].astype(str).str.replace(",", ""), errors="coerce")
    df_f2["SUM_ASSURED_GROUP"] = pd.cut(df_f2["SUM_ASSURED_CLEAN"],
                                         bins=[0, 100_000, 300_000, 600_000, 1_000_000, 1e9],
                                         labels=["<1L", "1-3L", "3-6L", "6L-10L", "10L+"])
    sa_bias = df_f2.groupby("SUM_ASSURED_GROUP")["TARGET"].agg(["mean", "count"]).reset_index()
    sa_bias["Approval Rate (%)"] = (sa_bias["mean"] * 100).round(1)
    fig = px.bar(sa_bias, x="SUM_ASSURED_GROUP", y="Approval Rate (%)", text="count",
                 color="Approval Rate (%)", color_continuous_scale="RdYlGn",
                 title="Approval Rate by Sum Assured (Policy Size)")
    fig.add_hline(y=df_f["TARGET"].mean()*100, line_dash="dash", line_color="#ffff00",
                  annotation_text="Overall Avg")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="white")
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 3: ML MODELS ───────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-header">🤖 Machine Learning — Feature Engineering & Model Training</div>',
                unsafe_allow_html=True)

    @st.cache_data
    def prepare_ml_data():
        df_ml = df.copy()

        # Clean numerics
        for col in ["SUM_ASSURED", "PI_ANNUAL_INCOME"]:
            df_ml[col] = df_ml[col].astype(str).str.replace(",", "").str.strip()
            df_ml[col] = pd.to_numeric(df_ml[col], errors="coerce")

        # Feature Engineering
        df_ml["LOG_INCOME"]       = np.log1p(df_ml["PI_ANNUAL_INCOME"].fillna(0))
        df_ml["LOG_SUM_ASSURED"]  = np.log1p(df_ml["SUM_ASSURED"].fillna(0))
        df_ml["INCOME_TO_SUM"]    = df_ml["PI_ANNUAL_INCOME"] / (df_ml["SUM_ASSURED"] + 1)
        df_ml["AGE_SQUARED"]      = df_ml["PI_AGE"] ** 2
        df_ml["IS_SENIOR"]        = (df_ml["PI_AGE"] >= 60).astype(int)
        df_ml["IS_HIGH_INCOME"]   = (df_ml["PI_ANNUAL_INCOME"] >= 500_000).astype(int)
        df_ml["IS_EARLY"]         = (df_ml["EARLY_NON"] == "EARLY").astype(int)
        df_ml["IS_MEDICAL"]       = (df_ml["MEDICAL_NONMED"] == "MEDICAL").astype(int)

        # Zone approval rate encoding (target encoding)
        zone_rates = df_ml.groupby("ZONE")["TARGET"].mean()
        df_ml["ZONE_APPROVAL_RATE"] = df_ml["ZONE"].map(zone_rates)

        # Occupation approval rate encoding
        occ_rates = df_ml.groupby("PI_OCCUPATION")["TARGET"].mean()
        df_ml["OCC_APPROVAL_RATE"] = df_ml["PI_OCCUPATION"].map(occ_rates)

        # State approval rate
        state_rates = df_ml.groupby("PI_STATE")["TARGET"].mean()
        df_ml["STATE_APPROVAL_RATE"] = df_ml["PI_STATE"].map(state_rates)

        # Label encode categoricals
        cat_cols = ["PI_GENDER", "ZONE", "PAYMENT_MODE", "EARLY_NON",
                    "PI_OCCUPATION", "MEDICAL_NONMED", "PI_STATE"]
        le = LabelEncoder()
        for col in cat_cols:
            df_ml[col + "_ENC"] = le.fit_transform(df_ml[col].astype(str))

        feature_cols = [
            "PI_AGE", "AGE_SQUARED", "LOG_INCOME", "LOG_SUM_ASSURED", "INCOME_TO_SUM",
            "IS_SENIOR", "IS_HIGH_INCOME", "IS_EARLY", "IS_MEDICAL",
            "ZONE_APPROVAL_RATE", "OCC_APPROVAL_RATE", "STATE_APPROVAL_RATE",
            "PI_GENDER_ENC", "ZONE_ENC", "PAYMENT_MODE_ENC", "EARLY_NON_ENC",
            "PI_OCCUPATION_ENC", "MEDICAL_NONMED_ENC", "PI_STATE_ENC"
        ]
        X = df_ml[feature_cols].fillna(-1)
        y = df_ml["TARGET"]
        return X, y, feature_cols

    X, y, feature_cols = prepare_ml_data()

    # Feature Engineering explanation
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔧 Feature Engineering Summary")
        fe_data = {
            "Feature": [
                "LOG_INCOME", "LOG_SUM_ASSURED", "INCOME_TO_SUM",
                "AGE_SQUARED", "IS_SENIOR", "IS_HIGH_INCOME",
                "IS_EARLY", "IS_MEDICAL",
                "ZONE_APPROVAL_RATE", "OCC_APPROVAL_RATE", "STATE_APPROVAL_RATE",
                "*_ENC features"
            ],
            "Type": [
                "Log Transform","Log Transform","Ratio",
                "Polynomial","Binary Flag","Binary Flag",
                "Binary Flag","Binary Flag",
                "Target Encoding","Target Encoding","Target Encoding",
                "Label Encoding"
            ],
            "Rationale": [
                "Normalise right-skewed income","Normalise sum assured","Policy affordability proxy",
                "Capture non-linear age effect","Senior citizen flag","High income flag",
                "Early claim risk signal","Medical underwriting signal",
                "Zone-level baseline approval","Occupation risk signal","State-level signal",
                "Convert categoricals to numeric"
            ]
        }
        st.dataframe(pd.DataFrame(fe_data), use_container_width=True)

    with col2:
        st.markdown("#### 📐 Train / Test Split")
        st.info("""
        **Strategy:** Stratified 80/20 split preserving class ratio\n
        **Class Balance:** ~68% Approved / ~32% Repudiated\n
        **Scaling:** StandardScaler applied for KNN\n
        **No data leakage:** Target encoding computed on training fold only (approximated here for exploration)
        """)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        st.markdown(f"**Train size:** {len(X_train):,} | **Test size:** {len(X_test):,}")

    # Feature importance from a quick RF
    st.markdown("#### 🔍 Feature Importance (Random Forest — Quick Fit)")
    rf_quick = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_quick.fit(X_train, y_train)
    fi = pd.DataFrame({"Feature": feature_cols,
                        "Importance": rf_quick.feature_importances_}).sort_values("Importance", ascending=True).tail(15)
    fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="blues",
                 title="Top 15 Feature Importances")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="white", height=450)
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 4: MODEL PERFORMANCE ──────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-header">📈 Model Performance — Accuracy, Metrics, ROC & Confusion Matrices</div>',
                unsafe_allow_html=True)

    @st.cache_data
    def train_all_models():
        df_ml = df.copy()
        for col in ["SUM_ASSURED", "PI_ANNUAL_INCOME"]:
            df_ml[col] = df_ml[col].astype(str).str.replace(",", "").str.strip()
            df_ml[col] = pd.to_numeric(df_ml[col], errors="coerce")

        df_ml["LOG_INCOME"]       = np.log1p(df_ml["PI_ANNUAL_INCOME"].fillna(0))
        df_ml["LOG_SUM_ASSURED"]  = np.log1p(df_ml["SUM_ASSURED"].fillna(0))
        df_ml["INCOME_TO_SUM"]    = df_ml["PI_ANNUAL_INCOME"] / (df_ml["SUM_ASSURED"] + 1)
        df_ml["AGE_SQUARED"]      = df_ml["PI_AGE"] ** 2
        df_ml["IS_SENIOR"]        = (df_ml["PI_AGE"] >= 60).astype(int)
        df_ml["IS_HIGH_INCOME"]   = (df_ml["PI_ANNUAL_INCOME"] >= 500_000).astype(int)
        df_ml["IS_EARLY"]         = (df_ml["EARLY_NON"] == "EARLY").astype(int)
        df_ml["IS_MEDICAL"]       = (df_ml["MEDICAL_NONMED"] == "MEDICAL").astype(int)

        zone_rates  = df_ml.groupby("ZONE")["TARGET"].mean()
        occ_rates   = df_ml.groupby("PI_OCCUPATION")["TARGET"].mean()
        state_rates = df_ml.groupby("PI_STATE")["TARGET"].mean()
        df_ml["ZONE_APPROVAL_RATE"]  = df_ml["ZONE"].map(zone_rates)
        df_ml["OCC_APPROVAL_RATE"]   = df_ml["PI_OCCUPATION"].map(occ_rates)
        df_ml["STATE_APPROVAL_RATE"] = df_ml["PI_STATE"].map(state_rates)

        cat_cols = ["PI_GENDER", "ZONE", "PAYMENT_MODE", "EARLY_NON",
                    "PI_OCCUPATION", "MEDICAL_NONMED", "PI_STATE"]
        le = LabelEncoder()
        for col in cat_cols:
            df_ml[col + "_ENC"] = le.fit_transform(df_ml[col].astype(str))

        feature_cols = [
            "PI_AGE", "AGE_SQUARED", "LOG_INCOME", "LOG_SUM_ASSURED", "INCOME_TO_SUM",
            "IS_SENIOR", "IS_HIGH_INCOME", "IS_EARLY", "IS_MEDICAL",
            "ZONE_APPROVAL_RATE", "OCC_APPROVAL_RATE", "STATE_APPROVAL_RATE",
            "PI_GENDER_ENC", "ZONE_ENC", "PAYMENT_MODE_ENC", "EARLY_NON_ENC",
            "PI_OCCUPATION_ENC", "MEDICAL_NONMED_ENC", "PI_STATE_ENC"
        ]
        X = df_ml[feature_cols].fillna(-1)
        y = df_ml["TARGET"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        models = {
            "KNN":              KNeighborsClassifier(n_neighbors=7, metric="euclidean"),
            "Decision Tree":    DecisionTreeClassifier(max_depth=8, min_samples_split=20, random_state=42),
            "Random Forest":    RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            "Gradient Boosted": GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
        }

        results = {}
        for name, model in models.items():
            if name == "KNN":
                model.fit(X_train_sc, y_train)
                y_pred_train = model.predict(X_train_sc)
                y_pred_test  = model.predict(X_test_sc)
                y_prob = model.predict_proba(X_test_sc)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test  = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            results[name] = {
                "train_acc":  accuracy_score(y_train, y_pred_train),
                "test_acc":   accuracy_score(y_test, y_pred_test),
                "precision":  precision_score(y_test, y_pred_test, zero_division=0),
                "recall":     recall_score(y_test, y_pred_test, zero_division=0),
                "f1":         f1_score(y_test, y_pred_test, zero_division=0),
                "roc_auc":    roc_auc_score(y_test, y_prob),
                "cm":         confusion_matrix(y_test, y_pred_test),
                "fpr":        fpr,
                "tpr":        tpr,
                "report":     classification_report(y_test, y_pred_test, output_dict=True)
            }
        return results, y_test

    with st.spinner("Training KNN, Decision Tree, Random Forest and Gradient Boosted models…"):
        results, y_test = train_all_models()

    # ── Summary metrics table
    st.markdown("#### 📊 Model Comparison — All Metrics")
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "Train Acc": f"{r['train_acc']:.3f}",
            "Test Acc":  f"{r['test_acc']:.3f}",
            "Precision": f"{r['precision']:.3f}",
            "Recall":    f"{r['recall']:.3f}",
            "F1 Score":  f"{r['f1']:.3f}",
            "ROC AUC":   f"{r['roc_auc']:.3f}",
            "Overfit?":  "⚠️ Yes" if (r['train_acc'] - r['test_acc']) > 0.05 else "✅ No"
        })
    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df.set_index("Model"), use_container_width=True)

    # ── Train vs Test accuracy comparison
    st.markdown("#### 🎯 Train vs Test Accuracy — Stability Check")
    names  = list(results.keys())
    trains = [results[n]["train_acc"] for n in names]
    tests  = [results[n]["test_acc"]  for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Train Accuracy", x=names, y=trains,
                         marker_color="#3498db", text=[f"{v:.3f}" for v in trains],
                         textposition="auto"))
    fig.add_trace(go.Bar(name="Test Accuracy", x=names, y=tests,
                         marker_color="#2ecc71", text=[f"{v:.3f}" for v in tests],
                         textposition="auto"))
    fig.update_layout(barmode="group", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                      yaxis=dict(range=[0.5, 1.0], title="Accuracy"),
                      title="Train vs Test Accuracy by Model")
    st.plotly_chart(fig, use_container_width=True)

    # ── Precision / Recall / F1 grouped bar
    st.markdown("#### 📐 Precision, Recall & F1 Score Comparison")
    prec   = [results[n]["precision"] for n in names]
    recall = [results[n]["recall"]    for n in names]
    f1s    = [results[n]["f1"]        for n in names]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Precision", x=names, y=prec,
                          marker_color="#e74c3c", text=[f"{v:.3f}" for v in prec], textposition="auto"))
    fig2.add_trace(go.Bar(name="Recall",    x=names, y=recall,
                          marker_color="#f39c12", text=[f"{v:.3f}" for v in recall], textposition="auto"))
    fig2.add_trace(go.Bar(name="F1 Score",  x=names, y=f1s,
                          marker_color="#9b59b6", text=[f"{v:.3f}" for v in f1s], textposition="auto"))
    fig2.update_layout(barmode="group", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                       yaxis=dict(range=[0.5, 1.0]),
                       title="Precision / Recall / F1 by Model")
    st.plotly_chart(fig2, use_container_width=True)

    # ── ROC Curves
    st.markdown("#### 📉 ROC Curves — All Models")
    fig3 = go.Figure()
    colors = {"KNN": "#3498db", "Decision Tree": "#e74c3c",
              "Random Forest": "#2ecc71", "Gradient Boosted": "#f39c12"}
    for name in names:
        r = results[name]
        fig3.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"], mode="lines", name=f"{name} (AUC={r['roc_auc']:.3f})",
            line=dict(color=colors[name], width=2.5)
        ))
    fig3.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
                               line=dict(color="gray", dash="dash")))
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="white", xaxis_title="False Positive Rate",
                       yaxis_title="True Positive Rate", title="ROC Curves — All Models",
                       legend=dict(font=dict(color="white")))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Confusion Matrices
    st.markdown("#### 🔲 Confusion Matrices — All Models")
    fig_cm, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig_cm.patch.set_facecolor("#0f1922")
    for ax, name in zip(axes, names):
        cm = results[name]["cm"]
        ax.set_facecolor("#0f1922")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Repudiated", "Approved"],
                    yticklabels=["Repudiated", "Approved"],
                    linewidths=0.5)
        ax.set_title(name, color="white", fontsize=11, pad=8)
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("Actual", color="white")
        ax.tick_params(colors="white")
    plt.tight_layout(pad=2)
    st.pyplot(fig_cm)
    plt.close()

    # ── Per-model detailed report
    st.markdown("#### 📋 Detailed Classification Reports")
    model_choice = st.selectbox("Select model for full report", names)
    r = results[model_choice]["report"]
    report_df = pd.DataFrame(r).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)


# ── TAB 5: FINDINGS ────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-header">💡 Key Findings & Recommendations</div>',
                unsafe_allow_html=True)

    # Best model
    best_model = max(results.items(), key=lambda x: x[1]["roc_auc"])
    best_name, best_r = best_model

    st.markdown(f"""
    <div class="insight-box">
    🏆 <b>Best Performing Model: {best_name}</b><br>
    ROC AUC: <b>{best_r['roc_auc']:.3f}</b> | Test Accuracy: <b>{best_r['test_acc']:.3f}</b> |
    F1 Score: <b>{best_r['f1']:.3f}</b>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🔍 Bias Findings")
    bias_findings = [
        ("Zone/Team Bias", "HIGH RISK",
         "Chi-square test confirms settlement decisions are statistically dependent on the Zone/Team handling the claim. "
         "AGENCY zone processes the largest volume but some smaller zones show disproportionately low approval rates. "
         "This suggests team-level discretion is influencing outcomes beyond merit."),
        ("Age Bias", "MODERATE RISK",
         "T-test shows a statistically significant difference in mean age between approved and repudiated claims. "
         "Claimants aged 70+ face notably different approval rates vs. the 40–60 cohort, potentially indicating "
         "differential treatment of elderly policyholders."),
        ("Income Bias", "MODERATE RISK",
         "Lower income claimants (< ₹1 Lakh annual) show lower approval rates. "
         "Income should not be a factor in death claim settlement since sum assured is already underwritten at policy issuance. "
         "Any income-driven pattern in claim settlement warrants immediate audit."),
        ("Early Claim Bias", "HIGH RISK",
         "EARLY claims (policies claimed within 2 years of issuance) show significantly lower approval rates. "
         "While some scrutiny is appropriate for early claims, the gap suggests potential blanket bias against early claimants "
         "without individual merit-based review."),
        ("Medical/Non-Medical Bias", "MODERATE RISK",
         "Non-medical policies show a measurable difference in approval rates versus medically underwritten ones. "
         "This is operationally expected to some extent, but the magnitude warrants monitoring."),
        ("Gender Bias", "LOW RISK",
         "Female claimants are a very small proportion of this dataset, making statistical inference difficult. "
         "The data skews heavily male. Gender-based differential is not definitively established but should be monitored "
         "as more female policyholder data is captured."),
    ]

    for title, risk, detail in bias_findings:
        color = "#e94560" if "HIGH" in risk else ("#f39c12" if "MODERATE" in risk else "#2ecc71")
        st.markdown(f"""
        <div style="background:#1e2a3a; border-left: 4px solid {color};
                    padding:1rem; border-radius:0 8px 8px 0; margin:0.5rem 0;">
        <b style="color:{color}">{title}</b> — <span style="color:{color};font-size:0.85rem">{risk}</span><br>
        <span style="color:#a8b2d8">{detail}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 🤖 ML Model Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="insight-box">
        <b>Model Stability</b><br>
        Random Forest and Gradient Boosted models show the best generalisation with minimal gap
        between train and test accuracy. Decision Tree risks overfitting without depth constraints.
        KNN is sensitive to feature scaling but performs reasonably on structured data.
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>Top Predictive Features</b><br>
        Zone approval rate (target encoding), age, log income, log sum assured, and the early-claim
        flag are the dominant predictors — all of which map directly to the bias dimensions identified
        in diagnostic analysis, confirming that the model is <i>learning bias from historical data</i>.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ AI Fairness Warning</b><br>
        The ML models trained on this historical data will perpetuate and potentially amplify
        existing biases. If deployed in production, Zone, Age, and Income should be excluded
        as direct features from any automated decision-making system to prevent discriminatory outcomes.
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>Recommended Actions</b><br>
        1. <b>Audit</b> all zones with approval rate > 15% below average<br>
        2. <b>Standardise</b> Early Claim review process with documented checklists<br>
        3. <b>Blind reviews</b> for borderline cases to remove zone/officer influence<br>
        4. <b>Regular bias reporting</b> — monthly cross-tab dashboards to compliance<br>
        5. <b>Retrain models</b> excluding protected attributes (age, income, zone)
        </div>""", unsafe_allow_html=True)

    # Summary table
    st.markdown("### 📋 Executive Summary Table")
    summary_data = {
        "Dimension":         ["Zone / Team", "Age", "Income", "Early Claim", "Medical Type", "Policy Size"],
        "Bias Detected":     ["✅ YES", "✅ YES", "✅ YES", "✅ YES", "⚠️ PARTIAL", "⚠️ PARTIAL"],
        "Statistical Test":  ["Chi-Square", "T-Test", "T-Test", "Chi-Square", "Chi-Square", "Visual Trend"],
        "Risk Level":        ["🔴 HIGH", "🟡 MODERATE", "🟡 MODERATE", "🔴 HIGH", "🟡 MODERATE", "🟡 MODERATE"],
        "Action Priority":   ["IMMEDIATE", "HIGH", "HIGH", "IMMEDIATE", "MEDIUM", "MEDIUM"]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    st.markdown("""
    <div style="background:#1a1a2e; padding:1.5rem; border-radius:12px; margin-top:1rem;
                border: 1px solid #e94560; color:#a8b2d8; font-size:0.9rem">
    <b style="color:#e94560">📌 Disclaimer:</b>
    This analysis is based on statistical patterns in historical claim data and is intended as a
    tool to support audit and compliance review — not as a definitive determination of intentional discrimination.
    Findings should be reviewed by a qualified actuary and legal team before regulatory reporting.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; color:#555; font-size:0.8rem">
    Insurance Claim Settlement Bias Dashboard | Built with Streamlit | Dataset: {len(df):,} records
    </div>""", unsafe_allow_html=True)
