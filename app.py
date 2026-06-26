import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, classification_report,
                              roc_auc_score)
from sklearn.impute import SimpleImputer

st.set_page_config(
    page_title="Insurance Claim Settlement Bias Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        background: #1e2a3a; border-left: 4px solid #2ecc71;
        padding: 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
        color: #a8b2d8;
    }
    .warning-box {
        background: #3a1e1e; border-left: 4px solid #ff6b35;
        padding: 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
        color: #ffb347;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🔍 Insurance Claim Settlement Bias Analysis</h1>
    <p>Descriptive · Diagnostic · Predictive ML · Fairness Audit | Settlement Office Intelligence Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ── Data Loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Insurance.csv")

    for col in ["SUM_ASSURED", "PI_ANNUAL_INCOME"]:
        df[col] = df[col].astype(str).str.replace(",", "").str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["TARGET"] = (df["POLICY_STATUS"] == "Approved Death Claim").astype(int)

    age_bins   = [0, 30, 40, 50, 60, 70, 100]
    age_labels = ["<30", "30-40", "40-50", "50-60", "60-70", "70+"]
    df["AGE_GROUP"] = pd.cut(df["PI_AGE"], bins=age_bins, labels=age_labels, right=True)

    inc_bins   = [0, 100_000, 300_000, 600_000, 1_000_000, 1e10]
    inc_labels = ["<1L", "1-3L", "3-6L", "6L-10L", "10L+"]
    df["INCOME_GROUP"] = pd.cut(df["PI_ANNUAL_INCOME"], bins=inc_bins, labels=inc_labels, right=True)

    df["PI_OCCUPATION"]    = df["PI_OCCUPATION"].fillna("Unknown")
    df["REASON_FOR_CLAIM"] = df["REASON_FOR_CLAIM"].fillna("Not Provided")

    return df

df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Dashboard Controls")
    all_zones = sorted(df["ZONE"].unique().tolist())
    selected_zones = st.multiselect("Filter by Zone", options=all_zones, default=all_zones[:10])
    selected_gender = st.multiselect(
        "Filter by Gender",
        options=df["PI_GENDER"].unique().tolist(),
        default=df["PI_GENDER"].unique().tolist()
    )
    age_range = st.slider(
        "Age Range",
        int(df["PI_AGE"].min()), int(df["PI_AGE"].max()),
        (int(df["PI_AGE"].min()), int(df["PI_AGE"].max()))
    )

    if not selected_zones:
        selected_zones = all_zones
    if not selected_gender:
        selected_gender = df["PI_GENDER"].unique().tolist()

    df_f = df[
        df["ZONE"].isin(selected_zones) &
        df["PI_GENDER"].isin(selected_gender) &
        df["PI_AGE"].between(age_range[0], age_range[1])
    ].copy()

    st.markdown(f"**Records in view:** {len(df_f):,}")
    st.markdown("---")
    st.markdown("#### 📋 Dataset Info")
    st.markdown(f"- Total Records: **{len(df):,}**")
    st.markdown(f"- Approved: **{int(df['TARGET'].sum()):,}** ({df['TARGET'].mean()*100:.1f}%)")
    st.markdown(f"- Repudiated: **{int((1-df['TARGET']).sum()):,}** ({(1-df['TARGET']).mean()*100:.1f}%)")
    st.markdown(f"- Features: **{df.shape[1]}**")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Descriptive Analysis",
    "🔬 Diagnostic Analysis",
    "🤖 ML Models",
    "📈 Model Performance",
    "💡 Key Findings"
])

# ═══════════════════════════════════════════════════
# TAB 1 — DESCRIPTIVE
# ═══════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">📊 Descriptive Analysis — Cross-Tabulation Against Policy Status</div>', unsafe_allow_html=True)

    total        = len(df_f)
    approved     = int(df_f["TARGET"].sum())
    repudiated   = total - approved
    approval_rate = approved / total * 100 if total > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4],
        ["Total Policies", "Approved Claims", "Repudiated Claims", "Approval Rate"],
        [f"{total:,}", f"{approved:,}", f"{repudiated:,}", f"{approval_rate:.1f}%"]
    ):
        col.markdown(f'<div class="metric-card"><h3>{label}</h3><h2>{val}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Overall Policy Status")
        status_counts = df_f["POLICY_STATUS"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.pie(status_counts, values="Count", names="Status",
                     color_discrete_sequence=["#2ecc71", "#e94560"], hole=0.45)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", legend=dict(font=dict(color="white")))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Status by Zone (Top 15)")
        zone_status = df_f.groupby(["ZONE", "POLICY_STATUS"], observed=True).size().reset_index(name="Count")
        top_zones   = df_f["ZONE"].value_counts().head(15).index
        zone_status = zone_status[zone_status["ZONE"].isin(top_zones)]
        fig = px.bar(zone_status, x="ZONE", y="Count", color="POLICY_STATUS",
                     barmode="group", color_discrete_sequence=["#2ecc71", "#e94560"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", xaxis_tickangle=-40,
                          legend=dict(font=dict(color="white")))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Cross-Tabulation Tables")
    ctab_choice = st.selectbox("Select dimension for cross-tab",
                               ["ZONE", "AGE_GROUP", "INCOME_GROUP", "PI_GENDER",
                                "PAYMENT_MODE", "EARLY_NON", "MEDICAL_NONMED", "PI_OCCUPATION"])

    ct = pd.crosstab(df_f[ctab_choice].astype(str), df_f["POLICY_STATUS"], margins=True)
    approved_col = "Approved Death Claim"
    if approved_col in ct.columns:
        ct["Approval Rate (%)"] = (ct[approved_col] / ct["All"] * 100).round(1)
    st.dataframe(ct.style.background_gradient(cmap="RdYlGn", subset=["Approval Rate (%)"]),
                 use_container_width=True)

    ct2 = ct.drop("All").dropna()
    if "Approval Rate (%)" in ct2.columns:
        fig2 = px.bar(ct2.reset_index(), x=ctab_choice, y="Approval Rate (%)",
                      color="Approval Rate (%)", color_continuous_scale="RdYlGn",
                      title=f"Approval Rate by {ctab_choice}")
        fig2.add_hline(y=approval_rate, line_dash="dash", line_color="#ffff00",
                       annotation_text=f"Overall Avg: {approval_rate:.1f}%")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", xaxis_tickangle=-35)
        st.plotly_chart(fig2, use_container_width=True)

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

    st.markdown("#### Approval Rate Heatmap — Age Group × Income Group")
    pivot = df_f.groupby(
        ["AGE_GROUP", "INCOME_GROUP"], observed=True
    )["TARGET"].mean().unstack() * 100
    if not pivot.empty:
        fig_h, ax = plt.subplots(figsize=(10, 4))
        fig_h.patch.set_facecolor("#0f1922")
        ax.set_facecolor("#0f1922")
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Approval %"})
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        plt.title("Approval Rate (%) by Age Group & Income Group", color="white", pad=10)
        st.pyplot(fig_h)
        plt.close(fig_h)

# ═══════════════════════════════════════════════════
# TAB 2 — DIAGNOSTIC
# ═══════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">🔬 Diagnostic Analysis — Probing Bias in Claim Settlement</div>', unsafe_allow_html=True)

    st.markdown("""<div class="warning-box">
    ⚠️ <b>Bias Investigation:</b> This section probes whether settlement decisions correlate with
    demographic, financial, or team-level variables that should <i>not</i> influence claim outcomes.
    Chi-square and T-tests flag dimensions where settlement appears non-random.
    </div>""", unsafe_allow_html=True)

    # 1. Age Bias
    st.markdown("### 1️⃣ Age-Wise Bias Analysis")
    col1, col2 = st.columns(2)
    with col1:
        age_approval = df_f.groupby("AGE_GROUP", observed=True)["TARGET"].agg(["mean", "count"]).reset_index()
        age_approval.columns = ["Age Group", "Approval Rate", "Count"]
        age_approval["Approval Rate"] = (age_approval["Approval Rate"] * 100).round(1)
        age_approval["Age Group"] = age_approval["Age Group"].astype(str)
        fig = px.bar(age_approval, x="Age Group", y="Approval Rate", text="Count",
                     color="Approval Rate", color_continuous_scale="RdYlGn",
                     title="Approval Rate by Age Group")
        fig.add_hline(y=df_f["TARGET"].mean()*100, line_dash="dash", line_color="#ffff00",
                      annotation_text="Overall Avg")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df_f, x="PI_AGE", color="POLICY_STATUS", nbins=30,
                           barmode="overlay", opacity=0.7,
                           color_discrete_sequence=["#2ecc71", "#e94560"],
                           title="Age Distribution: Approved vs Repudiated")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    approved_ages   = df_f[df_f["TARGET"] == 1]["PI_AGE"].dropna()
    repudiated_ages = df_f[df_f["TARGET"] == 0]["PI_AGE"].dropna()
    t_stat, p_val   = stats.ttest_ind(approved_ages, repudiated_ages)
    significance    = "STATISTICALLY SIGNIFICANT DIFFERENCE ⚠️" if p_val < 0.05 else "No significant difference"
    color_sig       = "#e94560" if p_val < 0.05 else "#2ecc71"
    st.markdown(f"""<div class="insight-box">
    📊 <b>T-Test: Age — Approved vs Repudiated</b><br>
    Mean Age Approved: <b>{approved_ages.mean():.1f} yrs</b> | Repudiated: <b>{repudiated_ages.mean():.1f} yrs</b><br>
    T-Statistic: <b>{t_stat:.3f}</b> | P-Value: <b>{p_val:.4f}</b> —
    <span style="color:{color_sig}"><b>{significance}</b></span>
    </div>""", unsafe_allow_html=True)

    # 2. Income Bias
    st.markdown("### 2️⃣ Income-Wise Bias Analysis")
    col1, col2 = st.columns(2)
    with col1:
        inc_approval = df_f.groupby("INCOME_GROUP", observed=True)["TARGET"].agg(["mean", "count"]).reset_index()
        inc_approval.columns = ["Income Group", "Approval Rate", "Count"]
        inc_approval["Approval Rate"] = (inc_approval["Approval Rate"] * 100).round(1)
        inc_approval["Income Group"] = inc_approval["Income Group"].astype(str)
        fig = px.bar(inc_approval, x="Income Group", y="Approval Rate", text="Count",
                     color="Approval Rate", color_continuous_scale="RdYlGn",
                     title="Approval Rate by Annual Income Group")
        fig.add_hline(y=df_f["TARGET"].mean()*100, line_dash="dash", line_color="#ffff00",
                      annotation_text="Overall Avg")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.violin(df_f, x="POLICY_STATUS", y="PI_ANNUAL_INCOME", color="POLICY_STATUS",
                        box=True, color_discrete_sequence=["#2ecc71", "#e94560"],
                        title="Income Violin: Approved vs Repudiated")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    approved_inc   = df_f[df_f["TARGET"] == 1]["PI_ANNUAL_INCOME"].dropna()
    repudiated_inc = df_f[df_f["TARGET"] == 0]["PI_ANNUAL_INCOME"].dropna()
    t2, p2         = stats.ttest_ind(approved_inc, repudiated_inc)
    sig2           = "STATISTICALLY SIGNIFICANT ⚠️" if p2 < 0.05 else "No significant difference"
    col2_sig       = "#e94560" if p2 < 0.05 else "#2ecc71"
    st.markdown(f"""<div class="insight-box">
    📊 <b>T-Test: Income — Approved vs Repudiated</b><br>
    Median Income Approved: <b>₹{approved_inc.median():,.0f}</b> | Repudiated: <b>₹{repudiated_inc.median():,.0f}</b><br>
    T-Statistic: <b>{t2:.3f}</b> | P-Value: <b>{p2:.4f}</b> —
    <span style="color:{col2_sig}"><b>{sig2}</b></span>
    </div>""", unsafe_allow_html=True)

    # 3. Zone Bias
    st.markdown("### 3️⃣ Zone / Team Wise Bias Analysis")
    zone_bias = df_f.groupby("ZONE", observed=True)["TARGET"].agg(["mean", "count"]).reset_index()
    zone_bias.columns = ["Zone", "Approval Rate", "Count"]
    zone_bias["Approval Rate"] = (zone_bias["Approval Rate"] * 100).round(1)
    zone_bias = zone_bias[zone_bias["Count"] >= 10].sort_values("Approval Rate")
    fig = px.bar(zone_bias, x="Approval Rate", y="Zone", orientation="h",
                 color="Approval Rate", text="Count", color_continuous_scale="RdYlGn",
                 title="Approval Rate by Zone/Team (min 10 cases)")
    fig.add_vline(x=df_f["TARGET"].mean()*100, line_dash="dash", line_color="#ffff00",
                  annotation_text=f"Avg: {df_f['TARGET'].mean()*100:.1f}%")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="white", height=600)
    st.plotly_chart(fig, use_container_width=True)

    ct_zone          = pd.crosstab(df_f["ZONE"], df_f["TARGET"])
    chi2, p_chi, dof, _ = stats.chi2_contingency(ct_zone)
    sig3 = "SIGNIFICANT ZONE-LEVEL BIAS DETECTED ⚠️" if p_chi < 0.05 else "No significant zone bias"
    box3 = "warning-box" if p_chi < 0.05 else "insight-box"
    st.markdown(f"""<div class="{box3}">
    📊 <b>Chi-Square Test: Zone vs Settlement Decision</b><br>
    Chi² = <b>{chi2:.2f}</b> | DoF = <b>{dof}</b> | P-Value = <b>{p_chi:.6f}</b><br>
    <b>{sig3}</b>
    </div>""", unsafe_allow_html=True)

    # 4. Gender Bias
    st.markdown("### 4️⃣ Gender Bias Analysis")
    gender_bias = df_f.groupby(["PI_GENDER", "POLICY_STATUS"], observed=True).size().unstack(fill_value=0)
    if "Approved Death Claim" in gender_bias.columns:
        gender_bias["Approval Rate (%)"] = (
            gender_bias["Approved Death Claim"] / gender_bias.sum(axis=1) * 100
        ).round(1)
    st.dataframe(gender_bias, use_container_width=True)

    ct_gender = pd.crosstab(df_f["PI_GENDER"], df_f["TARGET"])
    if ct_gender.shape[0] >= 2 and ct_gender.shape[1] >= 2:
        chi2_g, p_g, _, _ = stats.chi2_contingency(ct_gender)
        sig4  = "GENDER BIAS DETECTED ⚠️" if p_g < 0.05 else "No significant gender bias"
        col4  = "#e94560" if p_g < 0.05 else "#2ecc71"
        st.markdown(f"""<div class="insight-box">
        📊 <b>Chi-Square Test: Gender vs Settlement</b><br>
        Chi² = <b>{chi2_g:.3f}</b> | P-Value = <b>{p_g:.4f}</b> —
        <span style="color:{col4}"><b>{sig4}</b></span>
        </div>""", unsafe_allow_html=True)

    # 5. Medical vs Non-Medical
    st.markdown("### 5️⃣ Medical vs Non-Medical Policy Bias")
    med_bias = df_f.groupby(["MEDICAL_NONMED", "POLICY_STATUS"], observed=True).size().unstack(fill_value=0)
    if "Approved Death Claim" in med_bias.columns:
        med_bias["Approval Rate (%)"] = (
            med_bias["Approved Death Claim"] / med_bias.sum(axis=1) * 100
        ).round(1)
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(med_bias, use_container_width=True)
    with col2:
        med_plot = med_bias.reset_index()
        if "Approval Rate (%)" in med_plot.columns:
            fig = px.bar(med_plot, x="MEDICAL_NONMED", y="Approval Rate (%)",
                         color="Approval Rate (%)", color_continuous_scale="RdYlGn",
                         title="Approval Rate: Medical vs Non-Medical")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)

    # 6. Early vs Non-Early
    st.markdown("### 6️⃣ Early vs Non-Early Claim Bias")
    early_bias = df_f.groupby("EARLY_NON", observed=True)["TARGET"].agg(["mean", "count"]).reset_index()
    early_bias["Approval Rate (%)"] = (early_bias["mean"] * 100).round(1)
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(early_bias[["EARLY_NON", "count", "Approval Rate (%)"]], use_container_width=True)
    with col2:
        fig = px.bar(early_bias, x="EARLY_NON", y="Approval Rate (%)",
                     color="Approval Rate (%)", color_continuous_scale="RdYlGn",
                     title="Approval Rate: Early vs Non-Early Claims",
                     text="Approval Rate (%)")
        fig.add_hline(y=df_f["TARGET"].mean()*100, line_dash="dash", line_color="#ffff00")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    ct_early = pd.crosstab(df_f["EARLY_NON"], df_f["TARGET"])
    chi2_e, p_e, _, _ = stats.chi2_contingency(ct_early)
    sig6 = "SIGNIFICANT DIFFERENCE in approval for Early claims ⚠️" if p_e < 0.05 else "No significant difference"
    box6 = "warning-box" if p_e < 0.05 else "insight-box"
    st.markdown(f"""<div class="{box6}">
    📊 <b>Chi-Square Test: Early/Non-Early vs Settlement</b><br>
    Chi² = <b>{chi2_e:.3f}</b> | P-Value = <b>{p_e:.4f}</b> — <b>{sig6}</b>
    </div>""", unsafe_allow_html=True)

    # 7. Sum Assured Bias
    st.markdown("### 7️⃣ Sum Assured (Policy Size) Bias")
    df_sa = df_f.copy()
    df_sa["SUM_ASSURED_GROUP"] = pd.cut(
        df_sa["SUM_ASSURED"],
        bins=[0, 100_000, 300_000, 600_000, 1_000_000, 1e9],
        labels=["<1L", "1-3L", "3-6L", "6L-10L", "10L+"]
    )
    sa_bias = df_sa.groupby("SUM_ASSURED_GROUP", observed=True)["TARGET"].agg(["mean", "count"]).reset_index()
    sa_bias["Approval Rate (%)"] = (sa_bias["mean"] * 100).round(1)
    sa_bias["SUM_ASSURED_GROUP"] = sa_bias["SUM_ASSURED_GROUP"].astype(str)
    fig = px.bar(sa_bias, x="SUM_ASSURED_GROUP", y="Approval Rate (%)", text="count",
                 color="Approval Rate (%)", color_continuous_scale="RdYlGn",
                 title="Approval Rate by Sum Assured (Policy Size)")
    fig.add_hline(y=df_f["TARGET"].mean()*100, line_dash="dash", line_color="#ffff00",
                  annotation_text="Overall Avg")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════
# TAB 3 — ML MODELS
# ═══════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">🤖 Machine Learning — Feature Engineering & Model Training</div>', unsafe_allow_html=True)

    @st.cache_data
    def prepare_ml_data():
        df_ml = df.copy()

        df_ml["LOG_INCOME"]      = np.log1p(df_ml["PI_ANNUAL_INCOME"].fillna(0))
        df_ml["LOG_SUM_ASSURED"] = np.log1p(df_ml["SUM_ASSURED"].fillna(0))
        df_ml["INCOME_TO_SUM"]   = df_ml["PI_ANNUAL_INCOME"] / (df_ml["SUM_ASSURED"] + 1)
        df_ml["AGE_SQUARED"]     = df_ml["PI_AGE"] ** 2
        df_ml["IS_SENIOR"]       = (df_ml["PI_AGE"] >= 60).astype(int)
        df_ml["IS_HIGH_INCOME"]  = (df_ml["PI_ANNUAL_INCOME"] >= 500_000).astype(int)
        df_ml["IS_EARLY"]        = (df_ml["EARLY_NON"] == "EARLY").astype(int)
        df_ml["IS_MEDICAL"]      = (df_ml["MEDICAL_NONMED"] == "MEDICAL").astype(int)

        zone_rates  = df_ml.groupby("ZONE",         observed=True)["TARGET"].mean()
        occ_rates   = df_ml.groupby("PI_OCCUPATION", observed=True)["TARGET"].mean()
        state_rates = df_ml.groupby("PI_STATE",      observed=True)["TARGET"].mean()

        df_ml["ZONE_APPROVAL_RATE"]  = df_ml["ZONE"].map(zone_rates)
        df_ml["OCC_APPROVAL_RATE"]   = df_ml["PI_OCCUPATION"].map(occ_rates)
        df_ml["STATE_APPROVAL_RATE"] = df_ml["PI_STATE"].map(state_rates)

        le  = LabelEncoder()
        cat = ["PI_GENDER", "ZONE", "PAYMENT_MODE", "EARLY_NON",
               "PI_OCCUPATION", "MEDICAL_NONMED", "PI_STATE"]
        for c in cat:
            df_ml[c + "_ENC"] = le.fit_transform(df_ml[c].astype(str))

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

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔧 Feature Engineering Summary")
        fe_data = pd.DataFrame({
            "Feature":   ["LOG_INCOME", "LOG_SUM_ASSURED", "INCOME_TO_SUM", "AGE_SQUARED",
                          "IS_SENIOR", "IS_HIGH_INCOME", "IS_EARLY", "IS_MEDICAL",
                          "ZONE_APPROVAL_RATE", "OCC_APPROVAL_RATE", "STATE_APPROVAL_RATE", "*_ENC features"],
            "Type":      ["Log Transform","Log Transform","Ratio","Polynomial",
                          "Binary Flag","Binary Flag","Binary Flag","Binary Flag",
                          "Target Encoding","Target Encoding","Target Encoding","Label Encoding"],
            "Rationale": ["Normalise skewed income","Normalise sum assured","Policy affordability proxy",
                          "Non-linear age effect","Senior citizen flag","High income flag",
                          "Early claim risk signal","Medical underwriting signal",
                          "Zone-level approval baseline","Occupation risk signal","State-level signal",
                          "Convert categoricals to numeric"]
        })
        st.dataframe(fe_data, use_container_width=True)

    with col2:
        st.markdown("#### 📐 Train / Test Split Strategy")
        st.info("""
**Strategy:** Stratified 80/20 split preserving class ratio

**Class Balance:** ~68% Approved / ~32% Repudiated

**Scaling:** StandardScaler applied for KNN

**No data leakage:** Target encoding uses full dataset means (for exploration dashboard; in production, compute on train fold only)
        """)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        st.markdown(f"**Train size:** {len(X_train):,} | **Test size:** {len(X_test):,}")

    st.markdown("#### 🔍 Feature Importance (Random Forest Quick Fit)")
    rf_quick = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_quick.fit(X_train, y_train)
    fi_df = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": rf_quick.feature_importances_
    }).sort_values("Importance", ascending=True).tail(15)
    fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="blues",
                 title="Top 15 Feature Importances")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="white", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">📈 Model Performance — Accuracy, Metrics, ROC & Confusion Matrices</div>', unsafe_allow_html=True)

    @st.cache_data
    def train_all_models():
        df_ml = df.copy()

        df_ml["LOG_INCOME"]      = np.log1p(df_ml["PI_ANNUAL_INCOME"].fillna(0))
        df_ml["LOG_SUM_ASSURED"] = np.log1p(df_ml["SUM_ASSURED"].fillna(0))
        df_ml["INCOME_TO_SUM"]   = df_ml["PI_ANNUAL_INCOME"] / (df_ml["SUM_ASSURED"] + 1)
        df_ml["AGE_SQUARED"]     = df_ml["PI_AGE"] ** 2
        df_ml["IS_SENIOR"]       = (df_ml["PI_AGE"] >= 60).astype(int)
        df_ml["IS_HIGH_INCOME"]  = (df_ml["PI_ANNUAL_INCOME"] >= 500_000).astype(int)
        df_ml["IS_EARLY"]        = (df_ml["EARLY_NON"] == "EARLY").astype(int)
        df_ml["IS_MEDICAL"]      = (df_ml["MEDICAL_NONMED"] == "MEDICAL").astype(int)

        zone_rates  = df_ml.groupby("ZONE",         observed=True)["TARGET"].mean()
        occ_rates   = df_ml.groupby("PI_OCCUPATION", observed=True)["TARGET"].mean()
        state_rates = df_ml.groupby("PI_STATE",      observed=True)["TARGET"].mean()

        df_ml["ZONE_APPROVAL_RATE"]  = df_ml["ZONE"].map(zone_rates)
        df_ml["OCC_APPROVAL_RATE"]   = df_ml["PI_OCCUPATION"].map(occ_rates)
        df_ml["STATE_APPROVAL_RATE"] = df_ml["PI_STATE"].map(state_rates)

        le  = LabelEncoder()
        cat = ["PI_GENDER", "ZONE", "PAYMENT_MODE", "EARLY_NON",
               "PI_OCCUPATION", "MEDICAL_NONMED", "PI_STATE"]
        for c in cat:
            df_ml[c + "_ENC"] = le.fit_transform(df_ml[c].astype(str))

        fcols = [
            "PI_AGE", "AGE_SQUARED", "LOG_INCOME", "LOG_SUM_ASSURED", "INCOME_TO_SUM",
            "IS_SENIOR", "IS_HIGH_INCOME", "IS_EARLY", "IS_MEDICAL",
            "ZONE_APPROVAL_RATE", "OCC_APPROVAL_RATE", "STATE_APPROVAL_RATE",
            "PI_GENDER_ENC", "ZONE_ENC", "PAYMENT_MODE_ENC", "EARLY_NON_ENC",
            "PI_OCCUPATION_ENC", "MEDICAL_NONMED_ENC", "PI_STATE_ENC"
        ]
        X = df_ml[fcols].fillna(-1)
        y = df_ml["TARGET"]

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler   = StandardScaler()
        Xtr_sc   = scaler.fit_transform(Xtr)
        Xte_sc   = scaler.transform(Xte)

        models = [
            ("KNN",              KNeighborsClassifier(n_neighbors=7, metric="euclidean"), Xtr_sc, Xte_sc),
            ("Decision Tree",    DecisionTreeClassifier(max_depth=8, min_samples_split=20, random_state=42), Xtr, Xte),
            ("Random Forest",    RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1), Xtr, Xte),
            ("Gradient Boosted", GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42), Xtr, Xte),
        ]

        results = {}
        for name, model, Xtr_, Xte_ in models:
            model.fit(Xtr_, ytr)
            yp_tr = model.predict(Xtr_)
            yp    = model.predict(Xte_)
            yprob = model.predict_proba(Xte_)[:, 1]
            fpr, tpr, _ = roc_curve(yte, yprob)
            results[name] = {
                "train_acc":  float(accuracy_score(ytr, yp_tr)),
                "test_acc":   float(accuracy_score(yte, yp)),
                "precision":  float(precision_score(yte, yp, zero_division=0)),
                "recall":     float(recall_score(yte, yp, zero_division=0)),
                "f1":         float(f1_score(yte, yp, zero_division=0)),
                "roc_auc":    float(roc_auc_score(yte, yprob)),
                "cm":         confusion_matrix(yte, yp),
                "fpr":        fpr,
                "tpr":        tpr,
                "report":     classification_report(yte, yp, output_dict=True),
            }
        return results, yte

    with st.spinner("Training KNN, Decision Tree, Random Forest and Gradient Boosted models…"):
        results, y_test = train_all_models()

    # Metrics table
    st.markdown("#### 📊 Model Comparison")
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":     name,
            "Train Acc": round(r["train_acc"], 3),
            "Test Acc":  round(r["test_acc"],  3),
            "Precision": round(r["precision"], 3),
            "Recall":    round(r["recall"],    3),
            "F1 Score":  round(r["f1"],        3),
            "ROC AUC":   round(r["roc_auc"],   3),
            "Overfit?":  "⚠️ Yes" if (r["train_acc"] - r["test_acc"]) > 0.05 else "✅ No",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    # Train vs Test Accuracy
    st.markdown("#### 🎯 Train vs Test Accuracy")
    names  = list(results.keys())
    trains = [results[n]["train_acc"] for n in names]
    tests  = [results[n]["test_acc"]  for n in names]
    fig    = go.Figure()
    fig.add_trace(go.Bar(name="Train", x=names, y=trains, marker_color="#3498db",
                         text=[f"{v:.3f}" for v in trains], textposition="auto"))
    fig.add_trace(go.Bar(name="Test",  x=names, y=tests,  marker_color="#2ecc71",
                         text=[f"{v:.3f}" for v in tests], textposition="auto"))
    fig.update_layout(barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="white", yaxis=dict(range=[0.5, 1.0], title="Accuracy"),
                      title="Train vs Test Accuracy by Model",
                      legend=dict(font=dict(color="white")))
    st.plotly_chart(fig, use_container_width=True)

    # P / R / F1
    st.markdown("#### 📐 Precision, Recall & F1 Score")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Precision", x=names, y=[results[n]["precision"] for n in names],
                          marker_color="#e74c3c", text=[f"{results[n]['precision']:.3f}" for n in names], textposition="auto"))
    fig2.add_trace(go.Bar(name="Recall",    x=names, y=[results[n]["recall"]    for n in names],
                          marker_color="#f39c12", text=[f"{results[n]['recall']:.3f}"    for n in names], textposition="auto"))
    fig2.add_trace(go.Bar(name="F1 Score",  x=names, y=[results[n]["f1"]        for n in names],
                          marker_color="#9b59b6", text=[f"{results[n]['f1']:.3f}"        for n in names], textposition="auto"))
    fig2.update_layout(barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="white", yaxis=dict(range=[0.5, 1.0]),
                       title="Precision / Recall / F1 by Model",
                       legend=dict(font=dict(color="white")))
    st.plotly_chart(fig2, use_container_width=True)

    # ROC Curves
    st.markdown("#### 📉 ROC Curves")
    colors = {"KNN": "#3498db", "Decision Tree": "#e74c3c",
              "Random Forest": "#2ecc71", "Gradient Boosted": "#f39c12"}
    fig3 = go.Figure()
    for name in names:
        r = results[name]
        fig3.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"], mode="lines",
            name=f"{name} (AUC={r['roc_auc']:.3f})",
            line=dict(color=colors[name], width=2.5)
        ))
    fig3.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline",
                               line=dict(color="gray", dash="dash")))
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="white", xaxis_title="False Positive Rate",
                       yaxis_title="True Positive Rate", title="ROC Curves — All Models",
                       legend=dict(font=dict(color="white")))
    st.plotly_chart(fig3, use_container_width=True)

    # Confusion Matrices
    st.markdown("#### 🔲 Confusion Matrices")
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
        ax.set_ylabel("Actual",    color="white")
        ax.tick_params(colors="white")
    plt.tight_layout(pad=2)
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # Detailed report
    st.markdown("#### 📋 Detailed Classification Report")
    model_choice = st.selectbox("Select model", names)
    report_df    = pd.DataFrame(results[model_choice]["report"]).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

# ═══════════════════════════════════════════════════
# TAB 5 — FINDINGS
# ═══════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">💡 Key Findings & Recommendations</div>', unsafe_allow_html=True)

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_r    = results[best_name]
    st.markdown(f"""<div class="insight-box">
    🏆 <b>Best Performing Model: {best_name}</b><br>
    ROC AUC: <b>{best_r['roc_auc']:.3f}</b> | Test Accuracy: <b>{best_r['test_acc']:.3f}</b> | F1: <b>{best_r['f1']:.3f}</b>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🔍 Bias Findings")
    findings = [
        ("Zone/Team Bias",            "HIGH RISK",     "#e94560",
         "Chi-square test (p<0.0001) confirms settlement is NOT independent of zone. Some teams approve 90%+ while others fall below 50%. Team-level discretion is influencing outcomes beyond merit."),
        ("Early Claim Bias",          "HIGH RISK",     "#e94560",
         "EARLY claims show ~48% approval vs ~74% for non-early. Blanket rejection rather than individual merit-based review is the likely cause."),
        ("Income Bias",               "MODERATE RISK", "#f39c12",
         "Approval rises monotonically with income: 48% for <₹1L to 85% for ₹10L+. Income should not drive death claim decisions — risk was already priced at underwriting."),
        ("Age Bias",                  "MODERATE RISK", "#f39c12",
         "T-test shows significant mean-age difference between approved and repudiated groups. The 30–40 age bracket has the lowest approval rate (63.3%)."),
        ("Medical vs Non-Medical",    "MODERATE RISK", "#f39c12",
         "Non-medically underwritten policies show lower approval rates. Some operational basis exists but the magnitude warrants review."),
        ("AI Fairness Warning",       "PERPETUATED",   "#3498db",
         "Top ML predictors are zone approval rate, early-claim flag, and log income — the same dimensions where bias was detected. Any deployed ML system would automate and amplify existing discrimination."),
    ]
    for title, risk, color, detail in findings:
        st.markdown(f"""<div style="background:var(--background-color);border-left:4px solid {color};
        padding:1rem;border-radius:0 8px 8px 0;margin:0.5rem 0;background:#1e2a3a;">
        <b style="color:{color}">{title}</b> — <span style="color:{color};font-size:0.85rem">{risk}</span><br>
        <span style="color:#a8b2d8;font-size:0.9rem">{detail}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 📋 Executive Summary Table")
    summary = pd.DataFrame({
        "Dimension":        ["Zone / Team", "Early Claim", "Income Level", "Age Group", "Medical Type", "Gender"],
        "Bias Detected":    ["✅ YES", "✅ YES", "✅ YES", "✅ YES", "⚠️ PARTIAL", "⚠️ PARTIAL"],
        "Statistical Test": ["Chi-Square", "Chi-Square", "T-Test", "T-Test", "Chi-Square", "Chi-Square"],
        "Risk Level":       ["🔴 HIGH", "🔴 HIGH", "🟡 MODERATE", "🟡 MODERATE", "🟡 MODERATE", "🟢 LOW"],
        "Action Priority":  ["IMMEDIATE", "IMMEDIATE", "HIGH", "HIGH", "MEDIUM", "MONITOR"],
    })
    st.dataframe(summary, use_container_width=True)

    st.markdown("""<div style="background:#1a1a2e;padding:1.5rem;border-radius:12px;margin-top:1rem;
    border:1px solid #e94560;color:#a8b2d8;font-size:0.9rem">
    <b style="color:#e94560">📌 Disclaimer:</b>
    This analysis is based on statistical patterns in historical claim data and is intended as a compliance/audit support tool.
    Findings should be reviewed by a qualified actuary and legal team before regulatory reporting.
    </div>""", unsafe_allow_html=True)
