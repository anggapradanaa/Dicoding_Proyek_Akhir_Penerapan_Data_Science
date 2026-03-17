import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jaya Jaya Institut – Student Monitor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Blue Palette ──────────────────────────────────────────────────────────────
BLUE = {
    "darkest":  "#0a2342",
    "dark":     "#1a3a6b",
    "mid":      "#2563eb",
    "base":     "#3b82f6",
    "light":    "#93c5fd",
    "lighter":  "#bfdbfe",
    "lightest": "#eff6ff",
}

STATUS_COLOR = {
    "Dropout":  BLUE["dark"],
    "Enrolled": BLUE["base"],
    "Graduate": BLUE["light"],
}

COURSE_MAP = {
    33:"Biofuel Production Tech", 171:"Animation & Multimedia",
    8014:"Social Service (Eve)", 9003:"Agronomy", 9070:"Communication Design",
    9085:"Veterinary Nursing", 9119:"Informatics Engineering", 9130:"Equinculture",
    9147:"Basic Education", 9238:"Oral Hygiene", 9254:"Advertising & Mktg Mgmt",
    9500:"Journalism & Comm", 9556:"Basic Education (Eve)", 9670:"Management",
    9773:"Social Service", 9853:"Tourism", 9991:"Nursing",
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
.stApp {{ background: {BLUE["lightest"]}; }}

.main-header {{
    background: linear-gradient(135deg, {BLUE["darkest"]} 0%, {BLUE["dark"]} 50%, {BLUE["mid"]} 100%);
    border-radius: 16px; padding: 32px 40px; margin-bottom: 28px; color: white;
}}
.main-header h1 {{ font-size: 2rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.5px; }}
.main-header p  {{ font-size: 0.95rem; opacity: 0.8; margin: 0; }}

.metric-card {{
    background: white; border-radius: 14px; padding: 20px 22px;
    border-left: 5px solid {BLUE["base"]};
    box-shadow: 0 2px 12px rgba(37,99,235,0.08); height: 100%;
}}
.metric-card.dropout  {{ border-left-color: {BLUE["darkest"]}; }}
.metric-card.enrolled {{ border-left-color: {BLUE["base"]}; }}
.metric-card.graduate {{ border-left-color: {BLUE["light"]}; }}
.metric-card.warning  {{ border-left-color: #dc2626; background: #fff5f5; }}
.metric-card.caution  {{ border-left-color: {BLUE["mid"]}; background: {BLUE["lightest"]}; }}
.metric-card.safe     {{ border-left-color: {BLUE["light"]}; }}
.metric-label {{ font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.8px; color: #64748b; margin-bottom: 6px; }}
.metric-value {{ font-size: 2rem; font-weight: 800; color: {BLUE["darkest"]}; line-height: 1; }}
.metric-pct   {{ font-size: 0.82rem; color: {BLUE["mid"]}; font-weight: 600; margin-top: 4px; }}

.chart-card {{
    background: white; border-radius: 14px; padding: 22px;
    box-shadow: 0 2px 12px rgba(37,99,235,0.07); margin-bottom: 20px;
}}
.chart-title    {{ font-size: 1rem; font-weight: 700; color: {BLUE["darkest"]}; margin-bottom: 3px; }}
.chart-subtitle {{ font-size: 0.8rem; color: #94a3b8; margin-bottom: 14px; }}

.stTabs [data-baseweb="tab-list"] {{
    gap: 8px; background: white; padding: 8px; border-radius: 12px;
    box-shadow: 0 2px 8px rgba(37,99,235,0.08); margin-bottom: 24px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px; padding: 10px 20px; font-weight: 600;
    font-size: 0.86rem; color: #64748b; border: none !important;
}}
.stTabs [aria-selected="true"] {{ background: {BLUE["mid"]} !important; color: white !important; }}

.insight-box {{
    background: linear-gradient(135deg, {BLUE["lightest"]}, {BLUE["lighter"]});
    border: 1px solid {BLUE["lighter"]}; border-radius: 10px;
    padding: 14px 18px; font-size: 0.84rem; color: {BLUE["dark"]}; margin-top: 12px;
}}
.insight-box strong {{ color: {BLUE["darkest"]}; }}

.section-divider {{
    height: 1px; background: linear-gradient(90deg, {BLUE["lighter"]}, transparent);
    margin: 24px 0;
}}

section[data-testid="stSidebar"] {{
    background: white;
    border-right: 1px solid {BLUE["lighter"]};
}}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color, alpha=0.3):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def tidy_fig(fig, height=360, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=36, b=12),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Plus Jakarta Sans", size=12, color=BLUE["darkest"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)) if legend else dict(visible=False),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
        yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
    )
    return fig

STATUS_ORDER = ["Dropout", "Enrolled", "Graduate"]
PLOT_COLORS  = [STATUS_COLOR[s] for s in STATUS_ORDER]

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "data", "data.csv")
    try:
        df = pd.read_csv(path, sep=';')
        if df.shape[1] <= 1:
            df = pd.read_csv(path, sep=',')
    except Exception:
        df = pd.read_csv(path)

    df["Gender_label"]  = df["Gender"].map({1: "Male", 0: "Female"})
    df["Scholar_label"] = df["Scholarship_holder"].map({1: "Scholarship", 0: "Non-Scholarship"})
    df["Debtor_label"]  = df["Debtor"].map({1: "Debtor", 0: "Non-Debtor"})
    df["Tuition_label"] = df["Tuition_fees_up_to_date"].map({1: "Up to Date", 0: "Not Up to Date"})
    df["Course_name"]   = df["Course"].map(COURSE_MAP).fillna(df["Course"].astype(str))

    df["approval_rate_sem1"] = (
        df["Curricular_units_1st_sem_approved"] /
        df["Curricular_units_1st_sem_enrolled"].replace(0, np.nan)
    ).fillna(0)
    df["approval_rate_sem2"] = (
        df["Curricular_units_2nd_sem_approved"] /
        df["Curricular_units_2nd_sem_enrolled"].replace(0, np.nan)
    ).fillna(0)
    df["avg_grade_both_sem"] = (
        df["Curricular_units_1st_sem_grade"] + df["Curricular_units_2nd_sem_grade"]
    ) / 2
    df["total_approved"] = (
        df["Curricular_units_1st_sem_approved"] + df["Curricular_units_2nd_sem_approved"]
    )
    df["grade_trend"] = (
        df["Curricular_units_2nd_sem_grade"] - df["Curricular_units_1st_sem_grade"]
    )
    df["is_academically_active"] = (
        (df["Curricular_units_1st_sem_evaluations"] > 0) |
        (df["Curricular_units_2nd_sem_evaluations"] > 0)
    ).astype(int)
    df["Age_group"] = pd.cut(
        df["Age_at_enrollment"],
        bins=[0, 20, 23, 27, 35, 100],
        labels=["<=20", "21-23", "24-27", "28-35", ">35"],
    )
    return df

@st.cache_resource
def load_model():
    try:
        base      = os.path.dirname(os.path.abspath(__file__))
        model     = joblib.load(os.path.join(base, "model", "model.pkl"))
        le        = joblib.load(os.path.join(base, "model", "label_encoder.pkl"))
        feat_cols = joblib.load(os.path.join(base, "model", "feature_cols.pkl"))
        return model, le, feat_cols, None
    except Exception as e:
        return None, None, None, str(e)

df_raw = load_data()
model, le, feature_cols, model_err = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 Jaya Jaya Institut — Student Performance Monitor</h1>
    <p>Dashboard analitik untuk memahami pola dropout dan memonitor performa mahasiswa secara menyeluruh</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Filter ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Filter Global")
    st.caption("Filter berlaku untuk semua tab")
    st.markdown("---")

    gender_opts = ["Semua"] + sorted(df_raw["Gender_label"].dropna().unique().tolist())
    sel_gender  = st.selectbox("👤 Gender", gender_opts)

    course_opts = ["Semua"] + sorted(df_raw["Course_name"].dropna().unique().tolist())
    sel_course  = st.selectbox("📚 Jurusan", course_opts)

    st.markdown("---")
    st.markdown(f"**Total data:** `{len(df_raw):,}` mahasiswa")
    st.markdown(f"**Dropout rate:** `{(df_raw['Status']=='Dropout').mean()*100:.1f}%`")
    st.markdown("---")
    st.caption("Model: XGBoost | F1-Score: 0.7608")

# Terapkan filter
df = df_raw.copy()
if sel_gender != "Semua":
    df = df[df["Gender_label"] == sel_gender]
if sel_course != "Semua":
    df = df[df["Course_name"] == sel_course]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Overview",
    "🔍  Analisis Faktor Risiko",
    "📈  Performa Akademik",
    "🚨  Early Warning",
])

# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ╔══════════════════════════════════════════════════════════════════════════════
with tab1:
    total     = len(df)
    n_drop    = (df["Status"] == "Dropout").sum()
    n_enr     = (df["Status"] == "Enrolled").sum()
    n_grad    = (df["Status"] == "Graduate").sum()
    drop_rate = n_drop / total * 100

    c0, c1, c2, c3 = st.columns(4)
    with c0:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Mahasiswa</div>
            <div class="metric-value">{total:,}</div>
            <div class="metric-pct">Dataset aktif</div></div>""", unsafe_allow_html=True)
    with c1:
        st.markdown(f"""<div class="metric-card dropout">
            <div class="metric-label">Dropout</div>
            <div class="metric-value">{n_drop:,}</div>
            <div class="metric-pct">{n_drop/total*100:.1f}% dari total</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card enrolled">
            <div class="metric-label">Enrolled</div>
            <div class="metric-value">{n_enr:,}</div>
            <div class="metric-pct">{n_enr/total*100:.1f}% dari total</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card graduate">
            <div class="metric-label">Graduate</div>
            <div class="metric-value">{n_grad:,}</div>
            <div class="metric-pct">{n_grad/total*100:.1f}% dari total</div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Distribusi Status Mahasiswa</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-subtitle'>Proporsi keseluruhan dataset</div>", unsafe_allow_html=True)
        counts = df["Status"].value_counts().reindex(STATUS_ORDER)
        fig_donut = go.Figure(go.Pie(
            labels=counts.index, values=counts.values, hole=0.55,
            marker=dict(colors=PLOT_COLORS, line=dict(color="white", width=3)),
            textinfo="percent", textfont=dict(size=13, family="Plus Jakarta Sans"),
            hovertemplate="<b>%{label}</b><br>%{value:,} mahasiswa<br>%{percent}<extra></extra>",
        ))
        fig_donut.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="white", showlegend=True,
            legend=dict(orientation="v", x=1.05, y=0.5,
                        font=dict(size=12, family="Plus Jakarta Sans")),
            annotations=[dict(
                text=f"<b>{drop_rate:.1f}%</b><br>Dropout",
                x=0.5, y=0.5,
                font=dict(size=14, color=BLUE["darkest"], family="Plus Jakarta Sans"),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Dropout Rate per Jurusan (Top 10)</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-subtitle'>Persentase dropout per program studi — diurutkan tertinggi</div>", unsafe_allow_html=True)
        course_grp = (
            df.groupby("Course_name")["Status"]
            .apply(lambda x: (x == "Dropout").sum() / len(x) * 100)
            .reset_index(name="dropout_rate")
            .sort_values("dropout_rate", ascending=False)
            .head(10)
            .sort_values("dropout_rate", ascending=True)
        )
        fig_course = go.Figure(go.Bar(
            x=course_grp["dropout_rate"], y=course_grp["Course_name"],
            orientation="h",
            marker=dict(
                color=course_grp["dropout_rate"],
                colorscale=[[0, BLUE["lighter"]], [0.5, BLUE["base"]], [1, BLUE["darkest"]]],
                showscale=False,
            ),
            text=course_grp["dropout_rate"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Dropout Rate: %{x:.1f}%<extra></extra>",
        ))
        fig_course.update_layout(
            height=360, margin=dict(l=12, r=60, t=10, b=12),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Plus Jakarta Sans", size=11),
            xaxis=dict(title="Dropout Rate (%)", gridcolor="#f1f5f9",
                       range=[0, course_grp["dropout_rate"].max() * 1.22]),
            yaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_course, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    top_course      = course_grp.iloc[-1]["Course_name"] if len(course_grp) > 0 else "-"
    top_course_rate = course_grp.iloc[-1]["dropout_rate"] if len(course_grp) > 0 else 0
    filter_info     = f"{sel_gender} · {sel_course}" if (sel_gender != "Semua" or sel_course != "Semua") else "Semua Data"
    st.markdown(f"""
    <div class="insight-box">
        💡 <strong>Insight [{filter_info}]:</strong> Dari <strong>{total:,} mahasiswa</strong>,
        tingkat dropout mencapai <strong>{drop_rate:.1f}%</strong>.
        Jurusan dengan dropout rate tertinggi adalah <strong>{top_course} ({top_course_rate:.1f}%)</strong>
        — perlu perhatian khusus dari sisi kurikulum, dukungan finansial, maupun bimbingan akademik.
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALISIS FAKTOR RISIKO
# ╔══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Faktor Finansial & Demografis terhadap Status Mahasiswa")
    st.markdown("<p style='color:#94a3b8;font-size:0.84rem;margin-top:-8px;margin-bottom:20px'>Berdasarkan feature importance XGBoost — fitur finansial menempati 3 dari 5 posisi teratas</p>", unsafe_allow_html=True)

    def pct_bar(col_label):
        ct = (pd.crosstab(df[col_label], df["Status"], normalize="index") * 100
              ).reindex(columns=STATUS_ORDER, fill_value=0).reset_index()
        ct_melt = ct.melt(id_vars=col_label, var_name="Status", value_name="Pct")
        fig = px.bar(
            ct_melt, x=col_label, y="Pct", color="Status",
            color_discrete_map=STATUS_COLOR, barmode="stack",
            category_orders={"Status": STATUS_ORDER},
            text=ct_melt["Pct"].apply(lambda v: f"{v:.0f}%" if v > 6 else ""),
        )
        fig.update_traces(textposition="inside", textfont=dict(size=11, color="white"))
        fig = tidy_fig(fig, height=310)
        fig.update_layout(xaxis_title="", yaxis_title="Persentase (%)", yaxis_range=[0, 107])
        return fig

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Pembayaran SPP vs Status <span style='font-size:0.72rem;color:#3b82f6;background:#eff6ff;padding:2px 8px;border-radius:20px;margin-left:6px'>&#11088; Fitur #2</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-subtitle'>Mahasiswa yang tidak membayar SPP tepat waktu memiliki risiko dropout jauh lebih tinggi</div>", unsafe_allow_html=True)
        st.plotly_chart(pct_bar("Tuition_label"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r1c2:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Status Beasiswa vs Status <span style='font-size:0.72rem;color:#3b82f6;background:#eff6ff;padding:2px 8px;border-radius:20px;margin-left:6px'>&#11088; Fitur #5</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-subtitle'>Penerima beasiswa memiliki tingkat kelulusan lebih tinggi dibanding non-penerima</div>", unsafe_allow_html=True)
        st.plotly_chart(pct_bar("Scholar_label"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Status Hutang vs Status <span style='font-size:0.72rem;color:#3b82f6;background:#eff6ff;padding:2px 8px;border-radius:20px;margin-left:6px'>&#11088; Fitur #6</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-subtitle'>Mahasiswa dengan hutang menunjukkan kecenderungan dropout yang lebih signifikan</div>", unsafe_allow_html=True)
        st.plotly_chart(pct_bar("Debtor_label"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r2c2:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Gender vs Status</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-subtitle'>Distribusi status berdasarkan jenis kelamin mahasiswa</div>", unsafe_allow_html=True)
        st.plotly_chart(pct_bar("Gender_label"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title'>Distribusi Usia Pendaftaran vs Status</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-subtitle'>Mahasiswa yang mendaftar di usia lebih tua cenderung memiliki risiko dropout lebih tinggi</div>", unsafe_allow_html=True)
    age_ct = (pd.crosstab(df["Age_group"], df["Status"], normalize="index") * 100
              ).reindex(columns=STATUS_ORDER, fill_value=0).reset_index()
    age_melt = age_ct.melt(id_vars="Age_group", var_name="Status", value_name="Pct")
    fig_age = px.bar(
        age_melt, x="Age_group", y="Pct", color="Status",
        color_discrete_map=STATUS_COLOR, barmode="group",
        category_orders={"Status": STATUS_ORDER},
        text=age_melt["Pct"].apply(lambda v: f"{v:.0f}%"),
    )
    fig_age.update_traces(textposition="outside", textfont=dict(size=10))
    fig_age = tidy_fig(fig_age, height=310)
    fig_age.update_layout(xaxis_title="Kelompok Usia saat Mendaftar",
                          yaxis_title="Persentase (%)", yaxis_range=[0, 85])
    st.plotly_chart(fig_age, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Hitung data dinamis Tab 2
    dr_not_paid  = df[df["Tuition_fees_up_to_date"] == 0]["Status"].eq("Dropout").mean() * 100 if len(df[df["Tuition_fees_up_to_date"] == 0]) > 0 else 0
    dr_paid      = df[df["Tuition_fees_up_to_date"] == 1]["Status"].eq("Dropout").mean() * 100 if len(df[df["Tuition_fees_up_to_date"] == 1]) > 0 else 0
    dr_debtor    = df[df["Debtor"] == 1]["Status"].eq("Dropout").mean() * 100 if len(df[df["Debtor"] == 1]) > 0 else 0
    dr_scholar   = df[df["Scholarship_holder"] == 1]["Status"].eq("Dropout").mean() * 100 if len(df[df["Scholarship_holder"] == 1]) > 0 else 0
    dr_nonscholar= df[df["Scholarship_holder"] == 0]["Status"].eq("Dropout").mean() * 100 if len(df[df["Scholarship_holder"] == 0]) > 0 else 0
    age_high_risk= df[df["Age_at_enrollment"] > 28]["Status"].eq("Dropout").mean() * 100 if len(df[df["Age_at_enrollment"] > 28]) > 0 else 0
    st.markdown(f"""
    <div class="insight-box">
        💡 <strong>Insight [{filter_info}]:</strong>
        Mahasiswa yang <strong>tidak membayar SPP</strong> memiliki dropout rate <strong>{dr_not_paid:.1f}%</strong>
        vs <strong>{dr_paid:.1f}%</strong> yang membayar tepat waktu.
        Mahasiswa berhutang memiliki dropout rate <strong>{dr_debtor:.1f}%</strong>,
        sedangkan penerima beasiswa hanya <strong>{dr_scholar:.1f}%</strong> vs non-beasiswa <strong>{dr_nonscholar:.1f}%</strong>.
        Mahasiswa usia &gt;28 tahun memiliki dropout rate <strong>{age_high_risk:.1f}%</strong>.
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PERFORMA AKADEMIK
# ╔══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Performa Akademik & Tren Kurikuler")
    st.markdown("<p style='color:#94a3b8;font-size:0.84rem;margin-top:-8px;margin-bottom:20px'>Fitur akademik mendominasi Top-1 feature importance — approval rate semester 2 adalah sinyal terkuat prediksi dropout</p>", unsafe_allow_html=True)

    def box_chart(col, ylabel):
        data_list = []
        for status in STATUS_ORDER:
            vals = df[df["Status"] == status][col].dropna().tolist()
            data_list.append(go.Box(
                y=vals, name=status,
                marker_color=STATUS_COLOR[status],
                line_color=STATUS_COLOR[status],
                fillcolor=hex_to_rgba(STATUS_COLOR[status], 0.3),
                boxmean=True,
                hovertemplate=f"<b>{status}</b><br>%{{y:.2f}}<extra></extra>",
            ))
        fig = go.Figure(data_list)
        fig = tidy_fig(fig, height=310)
        fig.update_layout(yaxis_title=ylabel)
        return fig

    r1a, r1b = st.columns(2)
    with r1a:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Approval Rate Semester 1 <span style='font-size:0.72rem;color:#3b82f6;background:#eff6ff;padding:2px 8px;border-radius:20px;margin-left:6px'>&#11088; Fitur #7</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-subtitle'>Rasio unit disetujui terhadap unit diambil — semester 1</div>", unsafe_allow_html=True)
        st.plotly_chart(box_chart("approval_rate_sem1", "Approval Rate"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r1b:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Approval Rate Semester 2 <span style='font-size:0.72rem;color:#3b82f6;background:#eff6ff;padding:2px 8px;border-radius:20px;margin-left:6px'>&#127941; Fitur #1</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-subtitle'>Fitur terpenting model — mahasiswa dropout hampir selalu memiliki approval rate mendekati 0</div>", unsafe_allow_html=True)
        st.plotly_chart(box_chart("approval_rate_sem2", "Approval Rate"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    r2a, r2b = st.columns(2)
    with r2a:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Unit Disetujui Semester 1 <span style='font-size:0.72rem;color:#3b82f6;background:#eff6ff;padding:2px 8px;border-radius:20px;margin-left:6px'>&#11088; Fitur #4</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-subtitle'>Jumlah mata kuliah yang berhasil lulus di semester 1</div>", unsafe_allow_html=True)
        st.plotly_chart(box_chart("Curricular_units_1st_sem_approved", "Jumlah Unit"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r2b:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Unit Disetujui Semester 2 <span style='font-size:0.72rem;color:#3b82f6;background:#eff6ff;padding:2px 8px;border-radius:20px;margin-left:6px'>&#11088; Fitur #3</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-subtitle'>Jumlah mata kuliah yang berhasil lulus di semester 2</div>", unsafe_allow_html=True)
        st.plotly_chart(box_chart("Curricular_units_2nd_sem_approved", "Jumlah Unit"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    r3a, r3b = st.columns(2)
    with r3a:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Distribusi Nilai Rata-rata (Kedua Semester)</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-subtitle'>Mahasiswa dropout dominan memiliki nilai mendekati 0</div>", unsafe_allow_html=True)
        fig_hist = go.Figure()
        for status in STATUS_ORDER:
            vals = df[df["Status"] == status]["avg_grade_both_sem"].dropna()
            fig_hist.add_trace(go.Histogram(
                x=vals, name=status, marker_color=STATUS_COLOR[status],
                opacity=0.70, nbinsx=30,
                hovertemplate=f"<b>{status}</b><br>Nilai: %{{x:.1f}}<br>Jumlah: %{{y}}<extra></extra>",
            ))
        fig_hist = tidy_fig(fig_hist, height=310)
        fig_hist.update_layout(barmode="overlay",
                               xaxis_title="Nilai Rata-rata (0-20)", yaxis_title="Frekuensi")
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r3b:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Nilai Semester 1 vs Semester 2</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-subtitle'>Korelasi nilai antar semester — warna menunjukkan status akhir</div>", unsafe_allow_html=True)
        df_sample = df.sample(min(1000, len(df)), random_state=42)
        fig_scatter = px.scatter(
            df_sample,
            x="Curricular_units_1st_sem_grade",
            y="Curricular_units_2nd_sem_grade",
            color="Status", color_discrete_map=STATUS_COLOR,
            opacity=0.55, category_orders={"Status": STATUS_ORDER},
        )
        fig_scatter = tidy_fig(fig_scatter, height=310)
        fig_scatter.update_layout(xaxis_title="Nilai Semester 1 (0-20)",
                                  yaxis_title="Nilai Semester 2 (0-20)")
        fig_scatter.update_traces(marker=dict(size=5))
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Hitung data dinamis Tab 3
    avg_ar2_dropout  = df[df["Status"] == "Dropout"]["approval_rate_sem2"].mean() if len(df[df["Status"] == "Dropout"]) > 0 else 0
    avg_ar2_graduate = df[df["Status"] == "Graduate"]["approval_rate_sem2"].mean() if len(df[df["Status"] == "Graduate"]) > 0 else 0
    avg_grade_drop   = df[df["Status"] == "Dropout"]["avg_grade_both_sem"].mean() if len(df[df["Status"] == "Dropout"]) > 0 else 0
    avg_grade_grad   = df[df["Status"] == "Graduate"]["avg_grade_both_sem"].mean() if len(df[df["Status"] == "Graduate"]) > 0 else 0
    corr_val         = df["Curricular_units_1st_sem_grade"].corr(df["Curricular_units_2nd_sem_grade"]) if len(df) > 1 else 0
    st.markdown(f"""
    <div class="insight-box">
        💡 <strong>Insight [{filter_info}]:</strong>
        Rata-rata approval rate sem 2 mahasiswa <strong>Dropout: {avg_ar2_dropout:.2f}</strong>
        vs <strong>Graduate: {avg_ar2_graduate:.2f}</strong>.
        Nilai rata-rata Dropout (<strong>{avg_grade_drop:.1f}</strong>) jauh di bawah Graduate (<strong>{avg_grade_grad:.1f}</strong>).
        Korelasi nilai sem 1 dan sem 2 pada data ini: <strong>{corr_val:.2f}</strong>
        — intervensi dini di semester 1 sangat efektif untuk mencegah dropout.
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EARLY WARNING
# ╔══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### 🚨 Early Warning — Monitoring Risiko Dropout Mahasiswa Aktif")
    st.markdown("<p style='color:#94a3b8;font-size:0.84rem;margin-top:-8px;margin-bottom:20px'>Menggunakan model XGBoost untuk memprediksi probabilitas dropout mahasiswa berstatus Enrolled</p>", unsafe_allow_html=True)

    if model_err:
        st.warning(f"Model tidak dapat dimuat: {model_err}. Pastikan folder `model/` tersedia.")
        st.stop()

    # Ambil enrolled dengan filter
    enrolled_base = df_raw[df_raw["Status"] == "Enrolled"].copy()
    if sel_gender != "Semua":
        enrolled_base = enrolled_base[enrolled_base["Gender_label"] == sel_gender]
    if sel_course != "Semua":
        enrolled_base = enrolled_base[enrolled_base["Course_name"] == sel_course]

    if len(enrolled_base) == 0:
        st.info("Tidak ada mahasiswa berstatus Enrolled pada filter yang dipilih.")
        st.stop()

    # Prediksi
    X_enroll = enrolled_base[feature_cols].copy()
    proba    = model.predict_proba(X_enroll)
    classes  = list(le.classes_)
    enrolled_base = enrolled_base.copy()
    enrolled_base["Prob_Dropout"]  = proba[:, classes.index("Dropout")]
    enrolled_base["Prob_Graduate"] = proba[:, classes.index("Graduate")]
    enrolled_base["Prob_Enrolled"] = proba[:, classes.index("Enrolled")]

    # Threshold slider
    threshold = st.slider(
        "Threshold Risiko Tinggi (%)", 40, 80, 60, step=5,
        help="Mahasiswa dengan probabilitas dropout >= threshold dianggap berisiko tinggi"
    )
    thr       = threshold / 100
    n_high    = (enrolled_base["Prob_Dropout"] >= thr).sum()
    n_mid     = ((enrolled_base["Prob_Dropout"] >= 0.4) & (enrolled_base["Prob_Dropout"] < thr)).sum()
    n_low     = (enrolled_base["Prob_Dropout"] < 0.4).sum()
    total_enr = len(enrolled_base)

    # Metric cards
    mc0, mc1, mc2, mc3 = st.columns(4)
    with mc0:
        st.markdown(f"""<div class="metric-card enrolled">
            <div class="metric-label">Total Dipantau</div>
            <div class="metric-value">{total_enr:,}</div>
            <div class="metric-pct">Mahasiswa Enrolled</div></div>""", unsafe_allow_html=True)
    with mc1:
        st.markdown(f"""<div class="metric-card warning">
            <div class="metric-label">🔴 Risiko Tinggi</div>
            <div class="metric-value">{n_high:,}</div>
            <div class="metric-pct">>= {threshold}% prob. dropout</div></div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""<div class="metric-card caution">
            <div class="metric-label">🟡 Risiko Sedang</div>
            <div class="metric-value">{n_mid:,}</div>
            <div class="metric-pct">40% - {threshold}% prob. dropout</div></div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""<div class="metric-card safe">
            <div class="metric-label">🟢 Risiko Rendah</div>
            <div class="metric-value">{n_low:,}</div>
            <div class="metric-pct">< 40% prob. dropout</div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Row: Histogram + Bar Jurusan
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Distribusi Probabilitas Dropout</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-subtitle'>Sebaran probabilitas dropout seluruh mahasiswa enrolled</div>", unsafe_allow_html=True)
        fig_hist_ew = go.Figure()
        fig_hist_ew.add_trace(go.Histogram(
            x=enrolled_base["Prob_Dropout"], nbinsx=30,
            marker_color=BLUE["base"], opacity=0.75,
            hovertemplate="Prob: %{x:.2f}<br>Jumlah: %{y}<extra></extra>",
        ))
        fig_hist_ew.add_vline(
            x=thr, line_dash="dash", line_color=BLUE["darkest"], line_width=2,
            annotation_text=f"Threshold {threshold}%",
            annotation_font=dict(color=BLUE["darkest"], size=11),
        )
        fig_hist_ew.add_vline(
            x=0.4, line_dash="dot", line_color=BLUE["light"], line_width=1.5,
            annotation_text="40%",
            annotation_font=dict(color=BLUE["base"], size=10),
        )
        fig_hist_ew = tidy_fig(fig_hist_ew, height=300, legend=False)
        fig_hist_ew.update_layout(xaxis_title="Probabilitas Dropout",
                                  yaxis_title="Jumlah Mahasiswa")
        st.plotly_chart(fig_hist_ew, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ch2:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Top 10 Jurusan — Mahasiswa Risiko Tinggi</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-subtitle'>Jurusan dengan jumlah mahasiswa berisiko tinggi terbanyak</div>", unsafe_allow_html=True)
        high_risk = enrolled_base[enrolled_base["Prob_Dropout"] >= thr]
        if len(high_risk) > 0:
            jurusan_risk = (
                high_risk.groupby("Course_name").size()
                .reset_index(name="n_risiko")
                .sort_values("n_risiko", ascending=True)
                .tail(10)
            )
            fig_jr = go.Figure(go.Bar(
                x=jurusan_risk["n_risiko"], y=jurusan_risk["Course_name"],
                orientation="h",
                marker=dict(
                    color=jurusan_risk["n_risiko"],
                    colorscale=[[0, BLUE["lighter"]], [0.5, BLUE["base"]], [1, BLUE["darkest"]]],
                    showscale=False,
                ),
                text=jurusan_risk["n_risiko"], textposition="outside",
                hovertemplate="<b>%{y}</b><br>%{x} mahasiswa berisiko<extra></extra>",
            ))
            fig_jr = tidy_fig(fig_jr, height=300, legend=False)
            fig_jr.update_layout(
                xaxis_title="Jumlah Mahasiswa Risiko Tinggi",
                yaxis=dict(tickfont=dict(size=10)),
            )
            st.plotly_chart(fig_jr, use_container_width=True)
        else:
            st.info("Tidak ada mahasiswa risiko tinggi pada threshold ini.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Scatter: Approval Rate Sem2 vs Prob Dropout
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title'>Approval Rate Sem 2 vs Probabilitas Dropout <span style='font-size:0.72rem;color:#3b82f6;background:#eff6ff;padding:2px 8px;border-radius:20px;margin-left:6px'>&#127941; Fitur #1</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-subtitle'>Semakin rendah approval rate semester 2, semakin tinggi probabilitas dropout — korelasi langsung fitur terpenting model</div>", unsafe_allow_html=True)

    enrolled_plot = enrolled_base.copy()
    enrolled_plot["Risiko"] = enrolled_plot["Prob_Dropout"].apply(
        lambda p: "Tinggi" if p >= thr else ("Sedang" if p >= 0.4 else "Rendah")
    )
    risk_color_map = {"Tinggi": BLUE["darkest"], "Sedang": BLUE["base"], "Rendah": BLUE["light"]}

    fig_sc = px.scatter(
        enrolled_plot.sample(min(800, len(enrolled_plot)), random_state=42),
        x="approval_rate_sem2", y="Prob_Dropout",
        color="Risiko", color_discrete_map=risk_color_map,
        opacity=0.65,
        category_orders={"Risiko": ["Tinggi", "Sedang", "Rendah"]},
        hover_data={"Course_name": True, "approval_rate_sem1": ":.2f", "Prob_Dropout": ":.2f"},
    )
    fig_sc.add_hline(
        y=thr, line_dash="dash", line_color=BLUE["darkest"], line_width=1.5,
        annotation_text=f"Threshold {threshold}%",
        annotation_font=dict(color=BLUE["darkest"], size=11),
    )
    fig_sc = tidy_fig(fig_sc, height=340)
    fig_sc.update_layout(
        xaxis_title="Approval Rate Semester 2 (0-1)",
        yaxis_title="Probabilitas Dropout",
    )
    fig_sc.update_traces(marker=dict(size=6))
    st.plotly_chart(fig_sc, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Tabel risiko
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### Daftar Mahasiswa Berisiko")

    label_tinggi = f"Risiko Tinggi (>={threshold}%)"
    label_sedang = f"Risiko Sedang (40-{threshold}%)"
    filter_opt = st.radio("Tampilkan:", ["Semua", label_tinggi, label_sedang], horizontal=True)

    display_cols = [c for c in [
        "Course_name", "Gender_label", "Age_at_enrollment",
        "Curricular_units_1st_sem_grade", "Curricular_units_2nd_sem_grade",
        "approval_rate_sem1", "approval_rate_sem2",
        "Tuition_fees_up_to_date", "Debtor", "Scholarship_holder",
        "Prob_Dropout", "Prob_Enrolled", "Prob_Graduate",
    ] if c in enrolled_base.columns]

    tabel = enrolled_base[display_cols].sort_values("Prob_Dropout", ascending=False).copy()
    prob_raw = tabel["Prob_Dropout"].values.copy()

    tabel = tabel.rename(columns={
        "Course_name": "Jurusan", "Gender_label": "Gender",
        "Age_at_enrollment": "Usia",
        "Curricular_units_1st_sem_grade": "Nilai Sem1",
        "Curricular_units_2nd_sem_grade": "Nilai Sem2",
        "approval_rate_sem1": "Approval Sem1", "approval_rate_sem2": "Approval Sem2",
        "Tuition_fees_up_to_date": "SPP Lunas", "Debtor": "Punya Hutang",
        "Scholarship_holder": "Beasiswa",
        "Prob_Dropout": "P(Dropout)", "Prob_Enrolled": "P(Enrolled)",
        "Prob_Graduate": "P(Graduate)",
    })

    for col in ["P(Dropout)", "P(Enrolled)", "P(Graduate)", "Approval Sem1", "Approval Sem2"]:
        if col in tabel.columns:
            tabel[col] = tabel[col].map(lambda x: f"{x*100:.1f}%")
    for col in ["Nilai Sem1", "Nilai Sem2"]:
        if col in tabel.columns:
            tabel[col] = tabel[col].map(lambda x: f"{x:.1f}")
    for col in ["SPP Lunas", "Punya Hutang", "Beasiswa"]:
        if col in tabel.columns:
            tabel[col] = tabel[col].map({1: "Ya", 0: "Tidak"})

    # Badge kolom risiko
    tabel.insert(0, "Risiko", [
        "🔴 Tinggi" if p >= thr else ("🟡 Sedang" if p >= 0.4 else "🟢 Rendah")
        for p in prob_raw
    ])

    # Filter
    if filter_opt == label_tinggi:
        mask = prob_raw >= thr
    elif filter_opt == label_sedang:
        mask = (prob_raw >= 0.4) & (prob_raw < thr)
    else:
        mask = np.ones(len(prob_raw), dtype=bool)

    tabel_show = tabel[mask].reset_index(drop=True)
    tabel_show.index += 1

    def highlight_risk(row):
        p = float(row["P(Dropout)"].replace("%", "")) / 100
        if p >= thr:
            return ["background-color:#fee2e2; font-weight:600"] * len(row)
        elif p >= 0.4:
            return [f"background-color:{BLUE['lightest']}"] * len(row)
        return [""] * len(row)

    st.dataframe(
        tabel_show.style.apply(highlight_risk, axis=1),
        use_container_width=True, height=420,
    )

    csv = tabel_show.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Daftar Risiko (.csv)",
        data=csv, file_name="mahasiswa_berisiko_dropout.csv", mime="text/csv",
    )

    pct_high = n_high / total_enr * 100 if total_enr > 0 else 0
    course_counts   = high_risk.groupby("Course_name").size() if len(high_risk) > 0 else pd.Series()
    max_val         = course_counts.max() if len(course_counts) > 0 else 0
    top_risk_course = ", ".join(course_counts[course_counts == max_val].index.tolist()) if len(course_counts) > 0 else "-"
    avg_prob_high = enrolled_base[enrolled_base["Prob_Dropout"] >= thr]["Prob_Dropout"].mean() * 100 if n_high > 0 else 0
    st.markdown(f"""
    <div class="insight-box">
        💡 <strong>Early Warning [{filter_info}]:</strong>
        Dari <strong>{total_enr:,} mahasiswa enrolled</strong>, sebanyak <strong>{n_high} ({pct_high:.1f}%)</strong>
        berisiko tinggi dropout pada threshold <strong>{threshold}%</strong>.
        Jurusan dengan mahasiswa berisiko terbanyak: <strong>{top_risk_course}</strong>.
        Rata-rata probabilitas dropout kelompok risiko tinggi: <strong>{avg_prob_high:.1f}%</strong>.
        Fokus intervensi: mahasiswa dengan <strong>approval rate sem 2 rendah + SPP belum lunas + punya hutang</strong>.
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#94a3b8;font-size:0.78rem;padding:12px 0 24px'>
    Jaya Jaya Institut · Student Performance Dashboard · Model: XGBoost (F1: 0.7608) · Data: 4,424 mahasiswa
</div>
""", unsafe_allow_html=True)