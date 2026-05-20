import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os, time

# ══════════════════════════════════════════════════════
# Page Config
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="DiabetesAI · Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════
# Custom CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg:        #060b14;
    --surface:   #0d1626;
    --surface2:  #121f35;
    --border:    #1e3050;
    --accent:    #3b7fff;
    --accent2:   #00d4aa;
    --danger:    #ff4d6d;
    --warn:      #ffa94d;
    --text:      #e2eaf5;
    --muted:     #6b85a8;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Main background */
.main .block-container { padding: 2rem 2.5rem; max-width: 1400px; }

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-accent { border-left: 3px solid var(--accent); }
.card-danger { border-left: 3px solid var(--danger); }
.card-success { border-left: 3px solid var(--accent2); }

/* Metric tiles */
.metric-tile {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.4rem;
}

/* Risk badge */
.badge-high {
    display: inline-block;
    background: rgba(255,77,109,0.15);
    color: var(--danger);
    border: 1px solid var(--danger);
    border-radius: 50px;
    padding: 0.4rem 1.2rem;
    font-weight: 600;
    font-size: 1rem;
}
.badge-low {
    display: inline-block;
    background: rgba(0,212,170,0.15);
    color: var(--accent2);
    border: 1px solid var(--accent2);
    border-radius: 50px;
    padding: 0.4rem 1.2rem;
    font-weight: 600;
    font-size: 1rem;
}

/* Progress bar custom */
.prob-bar-bg {
    background: var(--surface2);
    border-radius: 100px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    border: 1px solid var(--border);
}
.prob-bar-fill-high {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #ff4d6d, #ff8fa3);
    transition: width 0.6s ease;
}
.prob-bar-fill-low {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #00d4aa, #38f0cc);
    transition: width 0.6s ease;
}

/* Report section */
.report-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    font-size: 0.9rem;
    line-height: 1.8;
    white-space: pre-wrap;
}
.report-section-title {
    color: var(--accent);
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.4rem;
}

/* Streamlit overrides */
.stSlider > label { color: var(--text) !important; font-size: 0.85rem !important; }
div[data-testid="stMetric"] { background: var(--surface2); border-radius: 12px; padding: 1rem; border: 1px solid var(--border); }
div[data-testid="stMetric"] label { color: var(--muted) !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Space Mono', monospace; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-radius: 12px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--muted) !important; border-radius: 8px !important; }
.stTabs [aria-selected="true"] { background: var(--accent) !important; color: white !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #3b7fff, #1d6aff) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(59,127,255,0.4) !important; }

/* Header */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #3b7fff 0%, #00d4aa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.3rem;
}
.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* Matplotlib */
.stPlot > div { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# Load Model
# ══════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    path = os.path.join(base, "model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

model_data  = load_model()
scaler      = model_data["scaler"]
centers     = model_data["centers"]
sigma       = model_data["sigma"]
lr          = model_data["lr"]
threshold   = model_data["threshold"]
MODEL_ACC   = model_data["accuracy"]


# ══════════════════════════════════════════════════════
# RBF Inference
# ══════════════════════════════════════════════════════
def rbf_features(X, centers, sigma):
    out = np.zeros((X.shape[0], len(centers)))
    for i, c in enumerate(centers):
        out[:, i] = np.exp(-np.sum((X - c) ** 2, axis=1) / (2 * sigma ** 2))
    return out

def predict(vals):
    x = np.array([vals])
    xs = scaler.transform(x)
    H  = rbf_features(xs, centers, sigma)
    p  = lr.predict_proba(H)[0]
    pred = int(p[1] >= threshold)
    return pred, p[0], p[1]


# ══════════════════════════════════════════════════════
# Medical Report Generator (Rule-Based)
# ══════════════════════════════════════════════════════
def generate_report(preg, gluc, bp, skin, ins, bmi, dpf, age, pred, p_diab):
    risk = "HIGH" if pred == 1 else "LOW"
    findings = []
    recs = []

    if gluc >= 126:
        findings.append(f"• Glucose: {gluc} mg/dL — DIABETIC range (≥126). Fasting hyperglycemia detected.")
        recs.append("• Consult endocrinologist for HbA1c testing immediately.")
    elif gluc >= 100:
        findings.append(f"• Glucose: {gluc} mg/dL — PRE-DIABETIC range (100–125). Monitor closely.")
        recs.append("• Reduce sugar intake and schedule follow-up fasting glucose test.")
    else:
        findings.append(f"• Glucose: {gluc} mg/dL — Normal range. ✓")

    if bmi >= 30:
        findings.append(f"• BMI: {bmi:.1f} — OBESE (≥30). Strong diabetes risk factor.")
        recs.append("• Structured weight-loss program: target 5–10% reduction in 6 months.")
    elif bmi >= 25:
        findings.append(f"• BMI: {bmi:.1f} — OVERWEIGHT (25–30). Moderate risk elevation.")
        recs.append("• Increase physical activity to 150 min/week of moderate exercise.")
    else:
        findings.append(f"• BMI: {bmi:.1f} — Healthy range. ✓")

    if bp >= 90:
        findings.append(f"• Blood Pressure: {bp} mmHg — HIGH (≥90 diastolic). Hypertension detected.")
        recs.append("• Blood pressure monitoring daily; low-sodium diet advised.")
    elif bp >= 80:
        findings.append(f"• Blood Pressure: {bp} mmHg — Elevated. Watch for hypertension.")
    else:
        findings.append(f"• Blood Pressure: {bp} mmHg — Normal. ✓")

    if ins > 200:
        findings.append(f"• Insulin: {ins} μU/mL — ELEVATED (>200). Possible insulin resistance.")
        recs.append("• Insulin resistance workup recommended (HOMA-IR test).")
    elif ins < 16 and ins > 0:
        findings.append(f"• Insulin: {ins} μU/mL — Below normal. Possible insufficient secretion.")
    else:
        findings.append(f"• Insulin: {ins} μU/mL — Within reference range. ✓")

    if dpf > 1.0:
        findings.append(f"• Diabetes Pedigree: {dpf:.3f} — HIGH genetic influence (>1.0).")
        recs.append("• Genetic counseling and regular family screening recommended.")
    elif dpf > 0.5:
        findings.append(f"• Diabetes Pedigree: {dpf:.3f} — Moderate genetic susceptibility.")
    else:
        findings.append(f"• Diabetes Pedigree: {dpf:.3f} — Low hereditary risk. ✓")

    if age > 45:
        findings.append(f"• Age: {age} years — Age >45 is an independent risk factor.")
        recs.append("• Annual diabetes screening (HbA1c) recommended for age >45.")
    else:
        findings.append(f"• Age: {age} years — Age is not a major risk factor currently. ✓")

    if preg > 5:
        findings.append(f"• Pregnancies: {preg} — History of multiple pregnancies elevates GDM risk.")

    if not recs:
        recs.append("• Maintain current healthy lifestyle and schedule annual check-ups.")
        recs.append("• Continue balanced diet and regular physical activity.")
        recs.append("• Annual fasting glucose screening recommended.")

    findings_str = "\n".join(findings)
    recs_str     = "\n".join(recs)

    reasoning = (
        f"The RBF model assessed {len(centers)} learned cluster centers from the "
        f"Pima Indians training dataset. Using Gaussian radial activations, "
        f"the patient's feature vector was mapped to a {len(centers)}-dimensional "
        f"representation, then classified by Logistic Regression at a calibrated "
        f"threshold of {threshold:.2f} (F1-optimized). "
        f"The prediction confidence is {p_diab:.1%} for diabetes."
    )

    report = f"""RISK LEVEL: {risk} ({p_diab:.1%} diabetes probability)
{'─' * 52}

CLINICAL FINDINGS
{findings_str}

{'─' * 52}
HEALTH RECOMMENDATIONS
{recs_str}

{'─' * 52}
MODEL REASONING
{reasoning}

{'─' * 52}
DISCLAIMER: This prediction is generated by an AI model
trained on the Pima Indians Diabetes Dataset. It is NOT
a substitute for professional medical advice. Please
consult a qualified healthcare provider for diagnosis.
"""
    return report


# ══════════════════════════════════════════════════════
# Plots (dark theme)
# ══════════════════════════════════════════════════════
BG   = "#060b14"
SRF  = "#0d1626"
BDR  = "#1e3050"
BLUE = "#3b7fff"
TEAL = "#00d4aa"
RED  = "#ff4d6d"
GOLD = "#ffa94d"
TXT  = "#e2eaf5"
MUT  = "#6b85a8"

def dark_fig(w=8, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SRF)
    for spine in ax.spines.values():
        spine.set_edgecolor(BDR)
    ax.tick_params(colors=MUT, labelsize=9)
    ax.xaxis.label.set_color(MUT)
    ax.yaxis.label.set_color(MUT)
    ax.title.set_color(TXT)
    ax.grid(color=BDR, linestyle='--', alpha=0.5)
    return fig, ax

def plot_gauge(prob):
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SRF)

    theta = np.linspace(np.pi, 0, 300)
    ax.plot(theta, [1]*300, color=BDR, linewidth=8)

    fill = int(prob * 300)
    color = RED if prob > 0.5 else TEAL
    ax.plot(theta[:fill], [1]*fill, color=color, linewidth=8)

    needle = np.pi - prob * np.pi
    ax.annotate('', xy=(needle, 0.9), xytext=(needle, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=2.5))

    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(0, -0.2, f"{prob:.1%}", ha='center', va='center',
            fontsize=22, fontweight='bold', color=color,
            fontfamily='monospace', transform=ax.transData)
    ax.text(0, -0.45, "Diabetes Probability", ha='center', va='center',
            fontsize=9, color=MUT, transform=ax.transData)

    plt.tight_layout()
    return fig

def plot_feature_radar(vals):
    labels = ['Pregnancies','Glucose','BP','Skin','Insulin','BMI','DPF','Age']
    norms  = [v/mx for v, mx in zip(vals, [17,200,122,99,846,67,2.5,81])]
    N = len(labels)
    angles = [n/N*2*np.pi for n in range(N)] + [0]
    norms  = norms + [norms[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SRF)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=MUT, size=8)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["25%","50%","75%"], color=MUT, size=7)
    ax.spines['polar'].set_color(BDR)
    ax.grid(color=BDR, alpha=0.5)

    ax.plot(angles, norms, color=BLUE, linewidth=2)
    ax.fill(angles, norms, color=BLUE, alpha=0.2)
    ax.set_title("Feature Profile", color=TXT, pad=15, fontsize=11)
    plt.tight_layout()
    return fig

def plot_prob_bar(p_diab, p_no):
    fig, ax = dark_fig(6, 2.5)
    bars = ax.barh(['Diabetic','Non-Diabetic'], [p_diab, p_no],
                   color=[RED, TEAL], edgecolor=BG, height=0.45)
    for bar, val in zip(bars, [p_diab, p_no]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va='center', color=TXT, fontsize=12,
                fontfamily='monospace', fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.set_title("Prediction Confidence", color=TXT, fontsize=11)
    ax.axvline(threshold, color=GOLD, linestyle='--', lw=1.5,
               label=f'Threshold={threshold:.2f}')
    ax.legend(fontsize=8, labelcolor=MUT, framealpha=0)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <span style='font-size:2.5rem'>🩺</span>
        <div style='font-family: Space Mono, monospace; font-size:1.1rem;
                    font-weight:700; color:#3b7fff; margin-top:0.3rem;'>
            DiabetesAI
        </div>
        <div style='color:#6b85a8; font-size:0.75rem; margin-top:0.2rem;'>
            RBF Neural Network · v1.0
        </div>
    </div>
    <hr style='border-color:#1e3050; margin:1rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Patient Input")

    preg  = st.slider("Pregnancies",              0,   17,  3,   1)
    gluc  = st.slider("Glucose (mg/dL)",          44, 200, 117,  1)
    bp    = st.slider("Blood Pressure (mmHg)",    24, 122,  72,  1)
    skin  = st.slider("Skin Thickness (mm)",       7,  99,  23,  1)
    ins   = st.slider("Insulin (μU/mL)",           14, 846,  30,  1)
    bmi   = st.slider("BMI",                      18.0, 67.0, 32.0, 0.1)
    dpf   = st.slider("Diabetes Pedigree Function",0.08, 2.42, 0.47, 0.01)
    age   = st.slider("Age (years)",               21,  81,  33,  1)

    st.markdown("<hr style='border-color:#1e3050;'>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Run Prediction")

    st.markdown("""
    <hr style='border-color:#1e3050;'>
    <div style='color:#6b85a8; font-size:0.72rem; line-height:1.6;'>
    <b style='color:#3b7fff;'>Model Info</b><br>
    Architecture: RBF Network<br>
    Centers (K): 10<br>
    Training set: 768 records<br>
    Balancing: SMOTE<br>
    Threshold: F1-Optimized
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════
st.markdown("""
<div class='hero-title'>Diabetes Risk Predictor</div>
<div class='hero-sub'>Radial Basis Function Neural Network · Pima Indians Dataset · Explainable AI</div>
<hr class='divider'>
""", unsafe_allow_html=True)

# Model stats bar
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Model", "RBF Network")
with c2:
    st.metric("RBF Centers", "K = 10")
with c3:
    st.metric("Training Samples", "768")
with c4:
    st.metric("Threshold", f"{threshold:.2f}")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── On predict ───────────────────────────────────────
if predict_btn:
    vals = [preg, gluc, bp, skin, ins, bmi, dpf, age]

    with st.spinner("Running RBF inference..."):
        time.sleep(0.4)
        pred, p_no, p_diab = predict(vals)

    report = generate_report(preg, gluc, bp, skin, ins, bmi, dpf, age, pred, p_diab)

    # ── Result Header ──────────────────────────────────
    if pred == 1:
        badge = f"<span class='badge-high'>⚠️ HIGH RISK — Diabetic</span>"
        bc    = "#ff4d6d"
    else:
        badge = f"<span class='badge-low'>✅ LOW RISK — Non-Diabetic</span>"
        bc    = "#00d4aa"

    st.markdown(f"""
    <div class='card' style='border-left:3px solid {bc}; text-align:center; padding:2rem;'>
        {badge}
        <div style='color:#6b85a8; font-size:0.85rem; margin-top:0.8rem;'>
            Diabetes Probability: <b style='color:{bc}; font-family:Space Mono,monospace;
            font-size:1.1rem;'>{p_diab:.1%}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊  Prediction Analysis", "📈  Feature Profile", "📋  Medical Report"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Probability Gauge")
            st.pyplot(plot_gauge(p_diab), use_container_width=True)
        with col2:
            st.markdown("#### Confidence Breakdown")
            st.pyplot(plot_prob_bar(p_diab, p_no), use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🔴 Diabetes Prob.", f"{p_diab:.1%}")
            with col_b:
                st.metric("🟢 No Diabetes Prob.", f"{p_no:.1%}")

    with tab2:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Patient Radar Chart")
            st.pyplot(plot_feature_radar(vals), use_container_width=True)
        with col2:
            st.markdown("#### Input Summary")
            labels = ['Pregnancies','Glucose','Blood Pressure','Skin Thickness',
                      'Insulin','BMI','Diabetes Pedigree','Age']
            norms_ref = [17, 200, 122, 99, 846, 67, 2.5, 81]
            for label, val, maxv in zip(labels, vals, norms_ref):
                pct = min(val / maxv, 1.0)
                color = "#ff4d6d" if pct > 0.7 else "#ffa94d" if pct > 0.4 else "#00d4aa"
                st.markdown(f"""
                <div style='margin-bottom:0.6rem;'>
                    <div style='display:flex; justify-content:space-between;
                                color:#e2eaf5; font-size:0.82rem; margin-bottom:3px;'>
                        <span>{label}</span>
                        <span style='font-family:Space Mono,monospace;
                                     color:{color};'>{val}</span>
                    </div>
                    <div class='prob-bar-bg'>
                        <div style='width:{pct*100:.1f}%; height:100%; border-radius:100px;
                                    background:{color};'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        st.markdown("#### AI-Generated Medical Report")
        st.markdown(f"""
        <div class='report-box'>{report}</div>
        """, unsafe_allow_html=True)

        st.download_button(
            label="⬇️  Download Report",
            data=report,
            file_name="diabetes_risk_report.txt",
            mime="text/plain"
        )

else:
    # ── Welcome State ──────────────────────────────────
    st.markdown("""
    <div class='card card-accent' style='text-align:center; padding:3rem;'>
        <div style='font-size:3rem; margin-bottom:1rem;'>🩺</div>
        <div style='font-size:1.2rem; font-weight:600; margin-bottom:0.5rem;'>
            Ready for Prediction
        </div>
        <div style='color:#6b85a8; font-size:0.9rem;'>
            Adjust the patient parameters in the sidebar,<br>
            then click <b style='color:#3b7fff;'>Run Prediction</b> to get a full risk assessment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='card'>
            <div style='font-size:1.5rem;'>🧠</div>
            <div style='font-weight:600; margin:0.5rem 0 0.3rem;'>RBF Network</div>
            <div style='color:#6b85a8; font-size:0.82rem;'>
                Manually implemented Radial Basis Function neural network with
                K-Means derived centers and Gaussian activations.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='card'>
            <div style='font-size:1.5rem;'>📊</div>
            <div style='font-weight:600; margin:0.5rem 0 0.3rem;'>Explainable AI</div>
            <div style='color:#6b85a8; font-size:0.82rem;'>
                Rule-based clinical report engine analyzing each biomarker
                against established medical reference ranges.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='card'>
            <div style='font-size:1.5rem;'>⚙️</div>
            <div style='font-weight:600; margin:0.5rem 0 0.3rem;'>F1-Optimized</div>
            <div style='color:#6b85a8; font-size:0.82rem;'>
                Decision threshold tuned via F1-Score optimization across
                0.1–0.9 range for balanced precision and recall.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────
st.markdown("""
<hr class='divider'>
<div style='text-align:center; color:#6b85a8; font-size:0.75rem; padding-bottom:1rem;'>
    Built with RBF Neural Network · Pima Indians Dataset · Streamlit<br>
    <span style='color:#1e3050;'>─────</span>
    <span style='color:#3b7fff;'> For educational & portfolio purposes only </span>
    <span style='color:#1e3050;'>─────</span>
</div>
""", unsafe_allow_html=True)
