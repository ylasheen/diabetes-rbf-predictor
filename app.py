import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os, time

st.set_page_config(
    page_title="DiabetesAI · Neural Risk System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;600&display=swap');

:root {
    --bg:         #030712;
    --bg2:        #070f1f;
    --glass:      rgba(255,255,255,0.03);
    --glass2:     rgba(255,255,255,0.06);
    --border:     rgba(255,255,255,0.07);
    --border2:    rgba(59,130,246,0.3);
    --blue:       #3b82f6;
    --cyan:       #06b6d4;
    --teal:       #14b8a6;
    --red:        #f43f5e;
    --amber:      #f59e0b;
    --purple:     #8b5cf6;
    --text:       #f1f5f9;
    --muted:      #64748b;
    --glow-blue:  0 0 40px rgba(59,130,246,0.15);
    --glow-cyan:  0 0 40px rgba(6,182,212,0.15);
}

*, body, html { font-family: 'Outfit', sans-serif !important; }

/* ── Global background ── */
.stApp, .main {
    background: radial-gradient(ellipse at 20% 0%, rgba(59,130,246,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 100%, rgba(6,182,212,0.06) 0%, transparent 60%),
                var(--bg) !important;
}
.main .block-container { padding: 2rem 2.5rem; max-width: 1440px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--blue) !important;
    border: 2px solid rgba(59,130,246,0.5) !important;
    box-shadow: 0 0 12px rgba(59,130,246,0.5) !important;
}
.stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] { color: var(--cyan) !important; font-family: 'JetBrains Mono', monospace !important; }
.stSlider label { color: var(--muted) !important; font-size: 0.8rem !important; letter-spacing: 0.5px !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
    border: none !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(6,182,212,0.2)) !important;
    color: var(--cyan) !important;
    border: 1px solid rgba(6,182,212,0.3) !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #0e7490) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 24px rgba(59,130,246,0.25) !important;
    transition: all 0.3s !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(59,130,246,0.4) !important;
}

/* ── Metrics ── */
div[data-testid="stMetric"] {
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1rem 1.2rem !important;
    backdrop-filter: blur(10px) !important;
}
div[data-testid="stMetric"] label { color: var(--muted) !important; font-size: 0.75rem !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 1.4rem !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background: var(--glass2) !important;
    color: var(--cyan) !important;
    border: 1px solid rgba(6,182,212,0.3) !important;
    border-radius: 10px !important;
    font-size: 0.85rem !important;
}

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ══ Load Model ══════════════════════════════════════════
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

m = load_model()
scaler, centers, sigma, lr, threshold = m["scaler"], m["centers"], m["sigma"], m["lr"], m["threshold"]

def rbf_features(X):
    out = np.zeros((X.shape[0], len(centers)))
    for i, c in enumerate(centers):
        out[:, i] = np.exp(-np.sum((X - c)**2, axis=1) / (2 * sigma**2))
    return out

def predict(vals):
    xs = scaler.transform(np.array([vals]))
    p  = lr.predict_proba(rbf_features(xs))[0]
    return int(p[1] >= threshold), p[0], p[1]


# ══ Medical Report ═══════════════════════════════════════
def generate_report(preg, gluc, bp, skin, ins, bmi, dpf, age, pred, p_diab):
    findings, recs = [], []
    if gluc >= 126:
        findings.append(f"• Glucose {gluc} mg/dL — Diabetic range (≥126). Fasting hyperglycemia confirmed.")
        recs.append("• Immediate endocrinologist referral. HbA1c test required.")
    elif gluc >= 100:
        findings.append(f"• Glucose {gluc} mg/dL — Pre-diabetic range (100–125). Closely monitor.")
        recs.append("• Reduce refined sugar intake. Schedule fasting glucose retest in 3 months.")
    else:
        findings.append(f"• Glucose {gluc} mg/dL — Normal. ✓")
    if bmi >= 30:
        findings.append(f"• BMI {bmi:.1f} — Obese (≥30). Primary modifiable risk factor.")
        recs.append("• Structured weight-loss program: 5–10% body weight reduction target.")
    elif bmi >= 25:
        findings.append(f"• BMI {bmi:.1f} — Overweight (25–30). Moderate risk elevation.")
        recs.append("• 150 min/week moderate aerobic exercise recommended.")
    else:
        findings.append(f"• BMI {bmi:.1f} — Healthy range. ✓")
    if bp >= 90:
        findings.append(f"• Blood Pressure {bp} mmHg — Hypertensive. Compounding diabetes risk.")
        recs.append("• Daily BP monitoring. Low-sodium diet (<2300 mg/day).")
    else:
        findings.append(f"• Blood Pressure {bp} mmHg — Normal. ✓")
    if ins > 200:
        findings.append(f"• Insulin {ins} μU/mL — Elevated. Insulin resistance suspected.")
        recs.append("• HOMA-IR test recommended to assess insulin resistance.")
    else:
        findings.append(f"• Insulin {ins} μU/mL — Within range. ✓")
    if dpf > 1.0:
        findings.append(f"• Diabetes Pedigree {dpf:.3f} — High genetic susceptibility (>1.0).")
        recs.append("• Screen first-degree relatives. Annual diabetes check mandatory.")
    elif dpf > 0.5:
        findings.append(f"• Diabetes Pedigree {dpf:.3f} — Moderate hereditary risk.")
    else:
        findings.append(f"• Diabetes Pedigree {dpf:.3f} — Low hereditary risk. ✓")
    if age > 45:
        findings.append(f"• Age {age} yrs — Age >45 is an independent risk factor.")
        recs.append("• Annual HbA1c screening recommended for all patients >45.")
    else:
        findings.append(f"• Age {age} yrs — Age not a major factor currently. ✓")
    if not recs:
        recs = ["• Maintain current healthy lifestyle.", "• Annual fasting glucose check.", "• Regular physical activity ≥150 min/week."]
    return (
        f"RISK CLASSIFICATION: {'HIGH RISK' if pred==1 else 'LOW RISK'} · {p_diab:.1%} probability\n"
        f"{'─'*54}\n\nCLINICAL FINDINGS\n" + "\n".join(findings) +
        f"\n\n{'─'*54}\nHEALTH RECOMMENDATIONS\n" + "\n".join(recs) +
        f"\n\n{'─'*54}\nMODEL REASONING\nRBF Network with {len(centers)} K-Means centers. Gaussian activations "
        f"mapped the 8-feature vector to a {len(centers)}-dimensional space. Logistic Regression "
        f"classified at F1-optimized threshold {threshold:.2f}. Confidence: {p_diab:.1%}.\n\n{'─'*54}\n"
        f"⚠ DISCLAIMER: AI-generated assessment. Not a substitute for medical diagnosis."
    )


# ══ Dark Plots ═══════════════════════════════════════════
BG, SRF, BDR = "#030712", "#0d1b2e", "#1e293b"
BLUE, CYAN, RED, TEAL, MUT = "#3b82f6", "#06b6d4", "#f43f5e", "#14b8a6", "#475569"

def fig_base(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SRF)
    for s in ax.spines.values(): s.set_color(BDR)
    ax.tick_params(colors=MUT, labelsize=9)
    ax.xaxis.label.set_color(MUT); ax.yaxis.label.set_color(MUT)
    ax.title.set_color("#e2eaf5")
    ax.grid(color=BDR, linestyle='--', alpha=0.6, linewidth=0.8)
    return fig, ax

def plot_gauge(prob):
    fig, ax = plt.subplots(figsize=(5, 3.2), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    theta = np.linspace(np.pi, 0, 500)
    # BG track
    ax.plot(theta, [1]*500, color='#1e293b', linewidth=14, solid_capstyle='round')
    # zones
    z1 = int(0.4*500); z2 = int(0.65*500)
    ax.plot(theta[:z1],    [1]*z1,    color="#14532d", linewidth=14, alpha=0.6, solid_capstyle='round')
    ax.plot(theta[z1:z2],  [1]*(z2-z1),  color="#713f12", linewidth=14, alpha=0.6, solid_capstyle='round')
    ax.plot(theta[z2:],    [1]*(500-z2), color="#4c0519", linewidth=14, alpha=0.6, solid_capstyle='round')
    # fill
    fill_n = int(prob * 500)
    color = TEAL if prob < 0.4 else ("#f59e0b" if prob < 0.65 else RED)
    if fill_n > 0:
        ax.plot(theta[:fill_n], [1]*fill_n, color=color, linewidth=14,
                solid_capstyle='round', alpha=0.9)
    # glow dot at tip
    if fill_n > 0:
        ax.plot([theta[fill_n-1]], [1], 'o', color=color, markersize=12,
                markerfacecolor=color, markeredgewidth=0,
                zorder=5)

    # needle
    needle_angle = np.pi - prob * np.pi
    ax.annotate('', xy=(needle_angle, 0.82), xytext=(needle_angle, 0.05),
                arrowprops=dict(arrowstyle='->', color='white', lw=2.5,
                                connectionstyle='arc3,rad=0'))
    ax.plot([needle_angle], [0.05], 'o', color='white', markersize=7, zorder=10)

    ax.set_ylim(0, 1.3); ax.set_yticks([]); ax.set_xticks([])
    for s in ax.spines.values(): s.set_visible(False)

    # labels
    ax.text(np.pi*0.88, 1.22, 'LOW', ha='center', color=TEAL, fontsize=8,
            fontfamily='monospace', fontweight='bold')
    ax.text(np.pi*0.5,  1.25, 'MED', ha='center', color='#f59e0b', fontsize=8,
            fontfamily='monospace', fontweight='bold')
    ax.text(np.pi*0.12, 1.22, 'HIGH', ha='center', color=RED, fontsize=8,
            fontfamily='monospace', fontweight='bold')

    ax.text(0, -0.18, f"{prob:.1%}", ha='center', va='center', fontsize=26,
            fontweight='bold', color=color, fontfamily='monospace',
            transform=ax.transData)
    ax.text(0, -0.42, "Diabetes Probability", ha='center', va='center',
            fontsize=9, color=MUT, transform=ax.transData)
    plt.tight_layout(pad=0)
    return fig

def plot_confidence(p_diab, p_no):
    fig, ax = fig_base(6, 2.8)
    colors = [RED, TEAL]
    vals   = [p_diab, p_no]
    labels = ['Diabetic', 'Non-Diabetic']
    bars = ax.barh(labels, vals, color=colors, height=0.4,
                   edgecolor='none', alpha=0.85)
    # glow
    for bar, c in zip(bars, colors):
        ax.barh([bar.get_y()+bar.get_height()/2], [bar.get_width()],
                height=0.55, color=c, alpha=0.1, edgecolor='none')
    for bar, v in zip(bars, vals):
        ax.text(v + 0.015, bar.get_y()+bar.get_height()/2,
                f"{v:.1%}", va='center', color='white',
                fontsize=13, fontfamily='monospace', fontweight='bold')
    ax.axvline(threshold, color='#f59e0b', linestyle='--', lw=1.5, alpha=0.8)
    ax.text(threshold+0.01, 1.15, f'threshold\n{threshold:.2f}',
            color='#f59e0b', fontsize=7.5, fontfamily='monospace')
    ax.set_xlim(0, 1.18)
    ax.set_title('Confidence Breakdown', fontsize=11, pad=10)
    plt.tight_layout()
    return fig

def plot_radar(vals):
    labels = ['Preg','Glucose','BP','Skin','Insulin','BMI','DPF','Age']
    maxv   = [17, 200, 122, 99, 846, 67, 2.5, 81]
    norms  = [min(v/m, 1.0) for v,m in zip(vals, maxv)]
    N = len(labels)
    angles = [n/N*2*np.pi for n in range(N)] + [0]
    norms_plot = norms + [norms[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SRF)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="#94a3b8", fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%','50%','75%','100%'], color=MUT, fontsize=7)
    ax.spines['polar'].set_color(BDR)
    ax.grid(color=BDR, alpha=0.5, linewidth=0.8)

    # gradient-like fill
    for i, alpha in [(0.03, 0.6), (0.0, 0.0)]:
        ax.fill(angles, [n+i for n in norms_plot], color=BLUE, alpha=alpha)
    ax.plot(angles, norms_plot, color=CYAN, linewidth=2, zorder=5)
    ax.fill(angles, norms_plot, color=BLUE, alpha=0.15)

    # dots
    for angle, norm, label in zip(angles[:-1], norms, labels):
        c = RED if norm > 0.75 else ("#f59e0b" if norm > 0.5 else TEAL)
        ax.plot([angle], [norm], 'o', color=c, markersize=7, zorder=10)

    ax.set_title('Feature Profile', color='#e2eaf5', pad=15, fontsize=11)
    plt.tight_layout()
    return fig


# ══ SIDEBAR ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 1.2rem 1rem; border-bottom:1px solid rgba(255,255,255,0.06);'>
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:4px;'>
            <span style='font-size:1.6rem;'>🩺</span>
            <div>
                <div style='font-family:JetBrains Mono,monospace; font-size:1rem;
                            font-weight:700; background:linear-gradient(135deg,#3b82f6,#06b6d4);
                            -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
                    DiabetesAI
                </div>
                <div style='color:#475569; font-size:0.7rem; margin-top:1px;'>
                    RBF Neural Network · v1.0
                </div>
            </div>
        </div>
    </div>
    <div style='padding:1rem 1.2rem 0.5rem;'>
        <div style='color:#64748b; font-size:0.7rem; text-transform:uppercase;
                    letter-spacing:1.5px; margin-bottom:1rem;'>
            ◈ Patient Parameters
        </div>
    </div>
    """, unsafe_allow_html=True)

    preg  = st.slider("Pregnancies",                    0,   17,  3,   1)
    gluc  = st.slider("Glucose (mg/dL)",               44,  200, 117,  1)
    bp    = st.slider("Blood Pressure (mmHg)",          24,  122,  72,  1)
    skin  = st.slider("Skin Thickness (mm)",             7,   99,  23,  1)
    ins   = st.slider("Insulin (μU/mL)",                14,  846,  30,  1)
    bmi   = st.slider("BMI",                          18.0, 67.0, 32.0, 0.1)
    dpf   = st.slider("Diabetes Pedigree Function",   0.08, 2.42, 0.47, 0.01)
    age   = st.slider("Age (years)",                    21,   81,  33,  1)

    st.markdown("<div style='padding:0 1.2rem;'>", unsafe_allow_html=True)
    run = st.button("⬡  Run Neural Analysis")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='margin:1.2rem; padding:1rem; background:rgba(255,255,255,0.02);
                border:1px solid rgba(255,255,255,0.05); border-radius:12px;
                font-family:JetBrains Mono,monospace; font-size:0.72rem; color:#475569;
                line-height:2;'>
        <span style='color:#3b82f6;'>Architecture</span> RBF Network<br>
        <span style='color:#3b82f6;'>Centers (K)</span>   {len(centers)}<br>
        <span style='color:#3b82f6;'>Dataset</span>       768 records<br>
        <span style='color:#3b82f6;'>Balancing</span>     SMOTE<br>
        <span style='color:#3b82f6;'>Threshold</span>     {threshold:.2f} (F1)
    </div>
    """, unsafe_allow_html=True)


# ══ MAIN ═════════════════════════════════════════════════

# Header
st.markdown("""
<div style='margin-bottom:2rem;'>
    <div style='font-family:Outfit,sans-serif; font-size:2.6rem; font-weight:800; line-height:1.1;
                background:linear-gradient(135deg,#e2eaf5 0%,#93c5fd 50%,#67e8f9 100%);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                background-clip:text; margin-bottom:0.4rem;'>
        Diabetes Risk Predictor
    </div>
    <div style='color:#475569; font-size:0.9rem; letter-spacing:0.5px;'>
        Radial Basis Function Neural Network &nbsp;·&nbsp; Pima Indians Dataset
        &nbsp;·&nbsp; Explainable AI Engine
    </div>
</div>
""", unsafe_allow_html=True)

# Stats bar
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Architecture", "RBF Network")
with c2: st.metric("RBF Centers", f"K = {len(centers)}")
with c3: st.metric("Training Records", "768")
with c4: st.metric("Decision Threshold", f"{threshold:.2f}")

st.markdown("<hr style='border-color:rgba(255,255,255,0.05); margin:1.5rem 0;'>", unsafe_allow_html=True)

# ── RESULT ────────────────────────────────────────────
if run:
    vals = [preg, gluc, bp, skin, ins, bmi, dpf, age]
    with st.spinner(""):
        time.sleep(0.5)
        pred, p_no, p_diab = predict(vals)

    report = generate_report(preg, gluc, bp, skin, ins, bmi, dpf, age, pred, p_diab)

    # Risk banner
    if pred == 1:
        bc, icon, label = "#f43f5e", "⚠", "HIGH RISK — Diabetic"
        bg_c = "rgba(244,63,94,0.07)"
    else:
        bc, icon, label = "#14b8a6", "✓", "LOW RISK — Non-Diabetic"
        bg_c = "rgba(20,184,166,0.07)"

    st.markdown(f"""
    <div style='background:{bg_c}; border:1px solid {bc}40;
                border-left:3px solid {bc}; border-radius:16px;
                padding:1.5rem 2rem; margin-bottom:1.5rem;
                display:flex; align-items:center; justify-content:space-between;'>
        <div style='display:flex; align-items:center; gap:16px;'>
            <div style='width:44px; height:44px; border-radius:50%;
                        background:{bc}20; border:1.5px solid {bc};
                        display:flex; align-items:center; justify-content:center;
                        font-size:1.3rem; color:{bc};'>{icon}</div>
            <div>
                <div style='font-size:1.3rem; font-weight:700; color:{bc};
                            font-family:Outfit,sans-serif;'>{label}</div>
                <div style='color:#64748b; font-size:0.82rem; margin-top:2px;'>
                    Neural analysis complete · RBF confidence score
                </div>
            </div>
        </div>
        <div style='text-align:right;'>
            <div style='font-family:JetBrains Mono,monospace; font-size:2.2rem;
                        font-weight:700; color:{bc}; line-height:1;'>{p_diab:.1%}</div>
            <div style='color:#64748b; font-size:0.75rem; margin-top:4px;'>diabetes probability</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    t1, t2, t3 = st.tabs(["  ⬡  Neural Analysis  ", "  ◈  Feature Profile  ", "  ✦  Clinical Report  "])

    with t1:
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown("<div style='color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.8rem;'>◈ PROBABILITY GAUGE</div>", unsafe_allow_html=True)
            st.pyplot(plot_gauge(p_diab), use_container_width=True)
        with col2:
            st.markdown("<div style='color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.8rem;'>◈ CONFIDENCE BREAKDOWN</div>", unsafe_allow_html=True)
            st.pyplot(plot_confidence(p_diab, p_no), use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            with m1: st.metric("🔴 Diabetic", f"{p_diab:.1%}")
            with m2: st.metric("🟢 Non-Diabetic", f"{p_no:.1%}")

    with t2:
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown("<div style='color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.8rem;'>◈ PATIENT RADAR</div>", unsafe_allow_html=True)
            st.pyplot(plot_radar(vals), use_container_width=True)
        with col2:
            st.markdown("<div style='color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:1rem;'>◈ BIOMARKER SUMMARY</div>", unsafe_allow_html=True)
            labels = ['Pregnancies','Glucose','Blood Pressure','Skin Thickness','Insulin','BMI','Diabetes Pedigree','Age']
            maxv   = [17, 200, 122, 99, 846, 67, 2.5, 81]
            for label, v, mx in zip(labels, vals, maxv):
                pct = min(v/mx, 1.0)
                color = "#f43f5e" if pct > 0.75 else ("#f59e0b" if pct > 0.45 else "#14b8a6")
                st.markdown(f"""
                <div style='margin-bottom:0.9rem;'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:5px;'>
                        <span style='color:#94a3b8; font-size:0.8rem;'>{label}</span>
                        <span style='font-family:JetBrains Mono,monospace; color:{color}; font-size:0.82rem; font-weight:600;'>{v}</span>
                    </div>
                    <div style='background:rgba(255,255,255,0.04); border-radius:100px; height:6px; border:1px solid rgba(255,255,255,0.05);'>
                        <div style='width:{pct*100:.1f}%; height:100%; border-radius:100px;
                                    background:linear-gradient(90deg, {color}99, {color});
                                    box-shadow: 0 0 8px {color}60;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with t3:
        st.markdown("<div style='color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:1rem;'>◈ AI CLINICAL REPORT</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);
                    border-radius:16px; padding:1.8rem 2rem;
                    font-family:JetBrains Mono,monospace; font-size:0.82rem;
                    line-height:2; color:#cbd5e1; white-space:pre-wrap;'>{report}</div>
        """, unsafe_allow_html=True)
        st.download_button("⬇  Download Report (.txt)", report,
                           "diabetes_report.txt", "text/plain")

else:
    # Welcome
    st.markdown("""
    <div style='background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);
                border-radius:20px; padding:3.5rem 2rem; text-align:center; margin-bottom:2rem;'>
        <div style='font-size:3rem; margin-bottom:1.2rem;'>🩺</div>
        <div style='font-size:1.4rem; font-weight:600; color:#e2eaf5; margin-bottom:0.6rem;'>
            Neural System Ready
        </div>
        <div style='color:#475569; font-size:0.9rem; max-width:400px; margin:0 auto; line-height:1.8;'>
            Configure patient biomarkers in the sidebar panel,<br>
            then click <span style='color:#06b6d4; font-weight:600;'>Run Neural Analysis</span>
            to generate a full risk assessment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    cards = [
        ("⬡", "#3b82f6", "RBF Network", "Manually implemented Radial Basis Function network with K-Means derived Gaussian centers. Zero dependency on neural network frameworks."),
        ("◈", "#06b6d4", "Explainable AI", "Rule-based clinical engine maps each biomarker to established medical reference ranges and generates actionable findings."),
        ("✦", "#8b5cf6", "F1-Optimized", "Decision threshold tuned across 0.1–0.9 range via F1-Score maximization for optimal precision-recall balance."),
    ]
    for col, (icon, color, title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);
                        border-radius:16px; padding:1.6rem; height:100%;
                        transition:all 0.3s;'>
                <div style='width:40px; height:40px; border-radius:10px;
                            background:{color}18; border:1px solid {color}40;
                            display:flex; align-items:center; justify-content:center;
                            font-size:1.2rem; color:{color}; margin-bottom:1rem;'>{icon}</div>
                <div style='font-weight:600; font-size:1rem; color:#e2eaf5;
                            margin-bottom:0.5rem;'>{title}</div>
                <div style='color:#475569; font-size:0.82rem; line-height:1.7;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style='border-color:rgba(255,255,255,0.05); margin:2rem 0 1rem;'>
<div style='text-align:center; color:#334155; font-size:0.75rem; padding-bottom:0.5rem; font-family:JetBrains Mono,monospace; letter-spacing:0.5px;'>
    DiabetesAI &nbsp;·&nbsp; RBF Neural Network &nbsp;·&nbsp; Pima Indians Dataset &nbsp;·&nbsp; Built for Portfolio
</div>
""", unsafe_allow_html=True)
