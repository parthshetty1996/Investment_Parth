# 🔍 Insurance Claim Settlement Bias Analysis Dashboard

A Streamlit dashboard for analysing bias in insurance claim settlements across demographic, financial, and team-level dimensions — with full ML classification suite.

---

## 📋 Features

| Tab | Content |
|-----|---------|
| 📊 Descriptive Analysis | Cross-tabulation, KPI cards, heatmaps, distributions |
| 🔬 Diagnostic Analysis | Bias probing (Age, Income, Zone, Gender, Early/Medical) with Chi-square & T-tests |
| 🤖 ML Models | Feature engineering + KNN, Decision Tree, Random Forest, Gradient Boosted |
| 📈 Model Performance | Train/Test accuracy, Precision/Recall/F1, ROC curves, Confusion matrices |
| 💡 Key Findings | Bias summary, recommendations, executive table |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/insurance-bias-dashboard.git
cd insurance-bias-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place Insurance.csv in the root folder (same level as app.py)

# 4. Launch
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo (with `Insurance.csv`) to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy**

> **Note:** The CSV file must be committed to the repo for Streamlit Cloud to access it.
> If the file is sensitive, consider loading it via an uploader widget (see `app_with_uploader.py`).

---

## 🗂️ Project Structure

```
insurance-bias-dashboard/
│
├── app.py               # Main Streamlit dashboard
├── requirements.txt     # Python dependencies
├── Insurance.csv        # Dataset (commit this for Cloud deployment)
└── README.md
```

---

## 📊 Dataset Columns

| Column | Description |
|--------|-------------|
| POLICY_NO | Unique policy identifier |
| PI_NAME | Policyholder name |
| PI_GENDER | Gender (M/F) |
| SUM_ASSURED | Policy sum assured (₹) |
| ZONE | Sales/settlement zone or team |
| PAYMENT_MODE | Annual / Half-Yearly / Quarterly / Monthly |
| EARLY_NON | EARLY = claimed within 2 yrs of issuance |
| PI_OCCUPATION | Policyholder occupation |
| MEDICAL_NONMED | MEDICAL / NON MEDICAL underwriting |
| PI_STATE | Indian state |
| REASON_FOR_CLAIM | Cause of death |
| PI_AGE | Age at claim |
| PI_ANNUAL_INCOME | Annual income (₹) |
| POLICY_STATUS | **Target** — Approved Death Claim / Repudiate Death |

---

## 🤖 ML Models Used

- **KNN** (K=7, Euclidean, with StandardScaler)
- **Decision Tree** (max_depth=8)
- **Random Forest** (200 trees, max_depth=10)
- **Gradient Boosted** (200 trees, lr=0.05, max_depth=5)

Feature engineering includes log transforms, binary flags, target encoding of zone/occupation/state, and polynomial age term.

---

## ⚠️ Disclaimer

Statistical bias findings are for compliance and audit support only. Consult a qualified actuary and legal counsel before regulatory reporting.
