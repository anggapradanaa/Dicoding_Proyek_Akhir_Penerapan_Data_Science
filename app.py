import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Dropout – Jaya Jaya Institut",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Blue Palette ─────────────────────────────────
BLUE = {
    "darkest":  "#0a2342",
    "dark":     "#1a3a6b",
    "mid":      "#2563eb",
    "base":     "#3b82f6",
    "light":    "#93c5fd",
    "lighter":  "#bfdbfe",
    "lightest": "#eff6ff",
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
.stApp {{ background: {BLUE["lightest"]}; }}

.main-header {{
    background: linear-gradient(135deg, {BLUE["darkest"]} 0%, {BLUE["dark"]} 50%, {BLUE["mid"]} 100%);
    border-radius: 16px; padding: 32px 40px; margin-bottom: 28px; color: white; text-align: center;
}}
.main-header h1 {{ font-size: 2rem; font-weight: 800; margin: 0 0 6px 0; }}
.main-header p  {{ font-size: 0.95rem; opacity: 0.8; margin: 0; }}

.section-header {{
    font-size: 0.88rem; font-weight: 700; color: {BLUE["darkest"]};
    border-left: 4px solid {BLUE["mid"]}; padding-left: 10px;
    margin-top: 1.2rem; margin-bottom: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.result-box {{
    border-radius: 14px; padding: 24px 28px; text-align: center;
    font-size: 1.6rem; font-weight: 800; margin-top: 1rem; color: white;
}}
.result-dropout  {{ background: linear-gradient(135deg, {BLUE["darkest"]}, {BLUE["dark"]}); }}
.result-graduate {{ background: linear-gradient(135deg, {BLUE["mid"]}, {BLUE["base"]}); }}
.result-enrolled {{ background: linear-gradient(135deg, {BLUE["base"]}, {BLUE["light"]}); color: {BLUE["darkest"]}; }}

.prob-card {{
    background: white; border-radius: 12px; padding: 16px;
    text-align: center; box-shadow: 0 2px 10px rgba(37,99,235,0.08);
    border-top: 4px solid {BLUE["base"]};
}}
.prob-card.dropout  {{ border-top-color: {BLUE["darkest"]}; }}
.prob-card.graduate {{ border-top-color: {BLUE["base"]}; }}
.prob-card.enrolled {{ border-top-color: {BLUE["light"]}; }}
.prob-value {{ font-size: 2rem; font-weight: 800; color: {BLUE["darkest"]}; }}
.prob-label {{ font-size: 0.78rem; color: #64748b; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }}

.insight-card {{
    background: white; border-radius: 12px; padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(37,99,235,0.07); margin-top: 16px;
    border-left: 5px solid {BLUE["mid"]};
}}
.insight-card h4 {{ color: {BLUE["darkest"]}; font-size: 0.9rem;
    font-weight: 700; margin: 0 0 12px 0; }}

.feature-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid {BLUE["lightest"]};
    font-size: 0.85rem;
}}
.feature-row:last-child {{ border-bottom: none; }}
.feature-name  {{ color: #64748b; }}
.feature-value {{ font-weight: 700; color: {BLUE["darkest"]}; }}
.feature-value.bad  {{ color: #dc2626; }}
.feature-value.good {{ color: {BLUE["mid"]}; }}

.info-box {{
    background: {BLUE["lightest"]}; border-left: 4px solid {BLUE["mid"]};
    padding: 10px 14px; border-radius: 0 8px 8px 0;
    margin-bottom: 16px; font-size: 0.85rem; color: {BLUE["dark"]};
}}

.chart-card {{
    background: white; border-radius: 14px; padding: 22px;
    box-shadow: 0 2px 12px rgba(37,99,235,0.07); margin-bottom: 20px;
}}

section[data-testid="stSidebar"] {{
    background: white; border-right: 1px solid {BLUE["lighter"]};
}}
</style>
""", unsafe_allow_html=True)

# ── Load Artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        base      = os.path.dirname(os.path.abspath(__file__))
        model     = joblib.load(os.path.join(base, "model", "model.pkl"))
        le        = joblib.load(os.path.join(base, "model", "label_encoder.pkl"))
        feat_cols = joblib.load(os.path.join(base, "model", "feature_cols.pkl"))
        return model, le, feat_cols, None
    except Exception as e:
        return None, None, None, str(e)

model, le, feature_cols, load_error = load_artifacts()

# ── Feature Engineering ───────────────────────────────────────────────────────
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['approval_rate_sem1'] = (
        df['Curricular_units_1st_sem_approved'] /
        df['Curricular_units_1st_sem_enrolled'].replace(0, np.nan)
    ).fillna(0)
    df['approval_rate_sem2'] = (
        df['Curricular_units_2nd_sem_approved'] /
        df['Curricular_units_2nd_sem_enrolled'].replace(0, np.nan)
    ).fillna(0)
    df['total_approved'] = (
        df['Curricular_units_1st_sem_approved'] +
        df['Curricular_units_2nd_sem_approved']
    )
    df['avg_grade_both_sem'] = (
        df['Curricular_units_1st_sem_grade'] +
        df['Curricular_units_2nd_sem_grade']
    ) / 2
    df['is_academically_active'] = (
        (df['Curricular_units_1st_sem_evaluations'] > 0) |
        (df['Curricular_units_2nd_sem_evaluations'] > 0)
    ).astype(int)
    df['grade_trend'] = (
        df['Curricular_units_2nd_sem_grade'] -
        df['Curricular_units_1st_sem_grade']
    )
    return df

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 Sistem Prediksi Dropout Mahasiswa</h1>
    <p>Jaya Jaya Institut — Masukkan data mahasiswa untuk memprediksi status kelulusan</p>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"Gagal memuat model: {load_error}. Pastikan folder `model/` tersedia.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 📋 Panduan")
    st.markdown("""
    1. Isi **data utama** mahasiswa di form
    2. Expand **Data Tambahan** jika perlu
    3. Klik **Prediksi Sekarang**
    4. Lihat hasil & analisis faktor risiko
    """)
    st.markdown("---")
    st.markdown("#### Status Prediksi")
    st.markdown(f"""
    - 🔴 **Dropout** — Berisiko keluar
    - 🔵 **Enrolled** — Masih aktif
    - 🟢 **Graduate** — Diprediksi lulus
    """)
    st.markdown("---")
    st.markdown("#### Faktor Paling Berpengaruh")
    st.markdown(f"""
    <div style='font-size:0.82rem;color:{BLUE["dark"]}'>
    1. Approval Rate Sem 2<br>
    2. SPP Tepat Waktu<br>
    3. Unit Disetujui Sem 2<br>
    4. Unit Disetujui Sem 1<br>
    5. Status Beasiswa<br>
    6. Status Debitur
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Model: XGBoost | F1-Score: 0.7608")

# ── Form ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="info-box">💡 Lengkapi data mahasiswa di bawah ini. Fokus pada <strong>data utama</strong> yang paling berpengaruh terhadap prediksi.</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# ── Kolom 1: Data Pribadi & Keuangan ─────────────────────────────────────────
with col1:
    st.markdown('<div class="section-header">👤 Data Pribadi</div>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", [1, 0],
        format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
    age_at_enrollment = st.slider("Usia Saat Mendaftar", 17, 70, 20)
    marital_status = st.selectbox("Status Pernikahan", [1,2,3,4,5,6],
        format_func=lambda x: {1:"Single",2:"Married",3:"Widower",
        4:"Divorced",5:"Facto Union",6:"Legally Separated"}[x])
    displaced = st.selectbox("Mahasiswa Displaced", [0,1],
        format_func=lambda x: "Ya" if x==1 else "Tidak")
    international = st.selectbox("Mahasiswa Internasional", [0,1],
        format_func=lambda x: "Ya" if x==1 else "Tidak")

    st.markdown('<div class="section-header">💳 Keuangan</div>', unsafe_allow_html=True)
    tuition_up_to_date = st.selectbox("SPP Tepat Waktu ⭐", [1, 0],
        format_func=lambda x: "Ya" if x==1 else "Tidak",
        help="Fitur #2 terpenting — sangat berpengaruh pada prediksi")
    debtor = st.selectbox("Status Debitur ⭐", [0, 1],
        format_func=lambda x: "Ya" if x==1 else "Tidak",
        help="Fitur #6 terpenting")
    scholarship_holder = st.selectbox("Penerima Beasiswa ⭐", [0, 1],
        format_func=lambda x: "Ya" if x==1 else "Tidak",
        help="Fitur #5 terpenting")

# ── Kolom 2: Data Akademik ────────────────────────────────────────────────────
with col2:
    st.markdown('<div class="section-header">📘 Semester 1</div>', unsafe_allow_html=True)
    cu1_enrolled    = st.number_input("Unit Diambil Sem 1 ⭐", 0, 20, 6,
        help="Fitur #8 terpenting")
    cu1_approved    = st.number_input("Unit Disetujui Sem 1 ⭐", 0, 20, 5,
        help="Fitur #4 terpenting")
    cu1_grade       = st.slider("Nilai Rata-rata Sem 1 (0-20)", 0.0, 20.0, 12.0, step=0.1)
    cu1_evaluations = st.number_input("Unit Evaluasi Sem 1", 0, 45, 6)
    cu1_credited    = st.number_input("Unit Credited Sem 1", 0, 20, 0)
    cu1_no_eval     = st.number_input("Unit Tanpa Evaluasi Sem 1", 0, 12, 0)

    st.markdown('<div class="section-header">📗 Semester 2</div>', unsafe_allow_html=True)
    cu2_enrolled    = st.number_input("Unit Diambil Sem 2 ⭐", 0, 20, 6,
        help="Fitur #9 terpenting")
    cu2_approved    = st.number_input("Unit Disetujui Sem 2 ⭐", 0, 20, 5,
        help="Fitur #3 terpenting")
    cu2_grade       = st.slider("Nilai Rata-rata Sem 2 (0-20)", 0.0, 20.0, 12.0, step=0.1)
    cu2_evaluations = st.number_input("Unit Evaluasi Sem 2", 0, 45, 6)
    cu2_credited    = st.number_input("Unit Credited Sem 2", 0, 20, 0)
    cu2_no_eval     = st.number_input("Unit Tanpa Evaluasi Sem 2", 0, 12, 0)

# ── Kolom 3: Data Akademik Lanjutan ──────────────────────────────────────────
with col3:
    st.markdown('<div class="section-header">🎓 Data Akademik</div>', unsafe_allow_html=True)
    course = st.selectbox("Program Studi ⭐", [33,171,8014,9003,9070,9085,9119,9130,
        9147,9238,9254,9500,9556,9670,9773,9853,9991],
        format_func=lambda x: {33:"Biofuel Production Tech",171:"Animation & Multimedia",
        8014:"Social Service (Evening)",9003:"Agronomy",9070:"Communication Design",
        9085:"Veterinary Nursing",9119:"Informatics Engineering",9130:"Equinculture",
        9147:"Management",9238:"Social Service",9254:"Tourism",9500:"Nursing",
        9556:"Oral Hygiene",9670:"Advertising & Marketing",9773:"Journalism & Comm.",
        9853:"Basic Education",9991:"Management (Evening)"}.get(x, str(x)),
        help="Fitur #10 terpenting")
    daytime_evening = st.selectbox("Waktu Kuliah", [1, 0],
        format_func=lambda x: "Pagi/Siang" if x==1 else "Malam")
    admission_grade = st.slider("Nilai Masuk (0-200)", 0.0, 200.0, 130.0, step=0.5)
    previous_qualification_grade = st.slider("Nilai Kualifikasi Sebelumnya (0-200)",
        0.0, 200.0, 130.0, step=0.5)
    application_order = st.slider("Urutan Pilihan Prodi (0=Pilihan 1)", 0, 9, 1)
    educational_special_needs = st.selectbox("Kebutuhan Khusus Pendidikan", [0,1],
        format_func=lambda x: "Ya" if x==1 else "Tidak")

    # ── Data Tambahan (disembunyikan) ─────────────────────────────────────────
    with st.expander("📂 Data Tambahan (tidak wajib diubah)"):
        st.caption("Fitur-fitur ini kurang berpengaruh pada prediksi. Nilai default sudah representatif.")
        application_mode = st.selectbox("Jalur Pendaftaran",
            [1,2,5,7,10,15,16,17,18,26,27,39,42,43,44,51,53,57],
            format_func=lambda x: {1:"1st Phase–General",2:"Ordinance 612/93",
            5:"1st Phase–Azores",7:"Other Higher Courses",10:"Ordinance 854-B/99",
            15:"International (Bachelor)",16:"1st Phase–Madeira",17:"2nd Phase–General",
            18:"3rd Phase–General",26:"Other Plan",27:"Other Institution",
            39:"Over 23 Years Old",42:"Transfer",43:"Change of Course",
            44:"Tech Specialization",51:"Change Inst./Course",
            53:"Short Cycle Diploma",57:"International Change"}.get(x, str(x)))
        previous_qualification = st.selectbox("Kualifikasi Sebelumnya",
            [1,2,3,4,5,6,9,10,12,14,15,19,38,39,40,42,43],
            format_func=lambda x: {1:"Secondary Ed",2:"Bachelor's",3:"Degree",
            4:"Master's",5:"Doctorate",6:"Freq Higher Ed",9:"12th–Not Done",
            10:"11th–Not Done",12:"Other 11th",14:"10th Year",15:"10th–Not Done",
            19:"Basic Ed 3rd Cycle",38:"Basic Ed 2nd Cycle",39:"Tech Specialization",
            40:"Higher Ed 1st Cycle",42:"Prof. Higher Tech",43:"Higher Ed Master"}.get(x, str(x)))
        nationality = st.selectbox("Kewarganegaraan",
            [1,2,6,11,13,14,41,62,100,101,103,105,108,109],
            format_func=lambda x: {1:"Portugal",2:"Jerman",6:"Spanyol",11:"Italia",
            13:"Belanda",14:"Inggris",41:"Brasil",62:"Romania",100:"Moldova",
            101:"Meksiko",103:"Ukraina",105:"Rusia",108:"Kuba",109:"Kolombia"}.get(x, str(x)))
        qual_opts = [1,2,3,4,5,6,9,10,11,12,14,18,19,22,26,27,29,30,34,35,36,37,38,39,40,41,42,43,44]
        qual_fmt  = lambda x: {1:"Secondary Ed",2:"Bachelor's",3:"Degree",4:"Master's",
            5:"Doctorate",6:"Freq Higher Ed",9:"12th–Not Done",10:"11th–Not Done",
            11:"7th Year",12:"Other 11th",14:"10th Year",18:"General Commerce",
            19:"Basic 3rd Cycle",22:"Tech-Professional",26:"7th Year Schooling",
            27:"2nd Cycle Basic",29:"9th Year–Not Done",30:"8th Year",
            34:"Unknown",35:"Can't Read/Write",36:"Can Read–No 4th Year",
            37:"Basic 1st Cycle",38:"Basic 2nd Cycle",39:"Tech Specialization",
            40:"Higher Ed 1st Cycle",41:"Higher Specialized",
            42:"Prof. Higher Tech",43:"Higher Ed 2nd Cycle",44:"Higher Ed 3rd Cycle"}.get(x, str(x))
        mothers_qualification = st.selectbox("Pendidikan Ibu", qual_opts, format_func=qual_fmt)
        fathers_qualification = st.selectbox("Pendidikan Ayah", qual_opts, format_func=qual_fmt)
        occ_opts = list(range(0,10))
        occ_fmt  = lambda x: {0:"Student",1:"Legislative/Director",2:"Intellectual/Scientific",
            3:"Intermediate Technician",4:"Administrative",5:"Services/Sellers",
            6:"Agriculture/Fisheries",7:"Industry/Construction",
            8:"Machine Operators",9:"Unskilled Workers"}.get(x, str(x))
        mothers_occupation = st.selectbox("Pekerjaan Ibu", occ_opts, format_func=occ_fmt)
        fathers_occupation = st.selectbox("Pekerjaan Ayah", occ_opts, format_func=occ_fmt)
        unemployment_rate = st.slider("Unemployment Rate (%)", 0.0, 25.0, 10.8, step=0.1)
        inflation_rate    = st.slider("Inflation Rate (%)", -1.0, 15.0, 1.4, step=0.1)
        gdp               = st.slider("GDP", -5.0, 5.0, 1.74, step=0.01)

# ── Predict Button ────────────────────────────────────────────────────────────
st.markdown("")
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_btn = st.button("🔮 Prediksi Sekarang", use_container_width=True)

# ── Prediction Logic ──────────────────────────────────────────────────────────
if predict_btn:
    raw_input = {
        'Marital_status'                              : marital_status,
        'Application_mode'                            : application_mode,
        'Application_order'                           : application_order,
        'Course'                                      : course,
        'Daytime_evening_attendance'                  : daytime_evening,
        'Previous_qualification'                      : previous_qualification,
        'Previous_qualification_grade'                : previous_qualification_grade,
        'Nacionality'                                 : nationality,
        'Mothers_qualification'                       : mothers_qualification,
        'Fathers_qualification'                       : fathers_qualification,
        'Mothers_occupation'                          : mothers_occupation,
        'Fathers_occupation'                          : fathers_occupation,
        'Admission_grade'                             : admission_grade,
        'Displaced'                                   : displaced,
        'Educational_special_needs'                   : educational_special_needs,
        'Debtor'                                      : debtor,
        'Tuition_fees_up_to_date'                     : tuition_up_to_date,
        'Gender'                                      : gender,
        'Scholarship_holder'                          : scholarship_holder,
        'Age_at_enrollment'                           : age_at_enrollment,
        'International'                               : international,
        'Curricular_units_1st_sem_credited'           : cu1_credited,
        'Curricular_units_1st_sem_enrolled'           : cu1_enrolled,
        'Curricular_units_1st_sem_evaluations'        : cu1_evaluations,
        'Curricular_units_1st_sem_approved'           : cu1_approved,
        'Curricular_units_1st_sem_grade'              : cu1_grade,
        'Curricular_units_1st_sem_without_evaluations': cu1_no_eval,
        'Curricular_units_2nd_sem_credited'           : cu2_credited,
        'Curricular_units_2nd_sem_enrolled'           : cu2_enrolled,
        'Curricular_units_2nd_sem_evaluations'        : cu2_evaluations,
        'Curricular_units_2nd_sem_approved'           : cu2_approved,
        'Curricular_units_2nd_sem_grade'              : cu2_grade,
        'Curricular_units_2nd_sem_without_evaluations': cu2_no_eval,
        'Unemployment_rate'                           : unemployment_rate,
        'Inflation_rate'                              : inflation_rate,
        'GDP'                                         : gdp,
    }

    input_df = pd.DataFrame([raw_input])
    input_df = add_engineered_features(input_df)
    input_df = input_df[feature_cols]

    pred       = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0]
    pred_label = le.inverse_transform([pred])[0]
    classes    = le.classes_

    # Engineered values untuk analisis
    ar2   = input_df['approval_rate_sem2'].values[0]
    ar1   = input_df['approval_rate_sem1'].values[0]
    grade = input_df['avg_grade_both_sem'].values[0]
    trend = input_df['grade_trend'].values[0]

    st.markdown("---")
    st.markdown("### 🎯 Hasil Prediksi")

    res_col, prob_col = st.columns([1, 1])

    # ── Hasil Utama ───────────────────────────────────────────────────────────
    with res_col:
        css_class = {"Dropout": "result-dropout", "Graduate": "result-graduate",
                     "Enrolled": "result-enrolled"}.get(pred_label, "result-enrolled")
        icon_map  = {"Dropout": "🔴", "Graduate": "🟢", "Enrolled": "🔵"}
        st.markdown(f"""
        <div class="result-box {css_class}">
            {icon_map.get(pred_label, "")} Prediksi: <b>{pred_label}</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        if pred_label == "Dropout":
            st.error("⚠️ Mahasiswa ini **berisiko tinggi dropout**. Segera lakukan bimbingan dan tinjau kondisi keuangannya.")
        elif pred_label == "Graduate":
            st.success("✅ Mahasiswa ini diprediksi akan **berhasil lulus**. Tetap pantau perkembangannya.")
        else:
            st.info("ℹ️ Mahasiswa masih **aktif berstudi**. Lakukan monitoring berkala.")

        # Probabilitas card
        st.markdown("")
        st.markdown("**Probabilitas per Status:**")
        prob_cols = st.columns(3)
        css_map   = {"Dropout": "dropout", "Enrolled": "enrolled", "Graduate": "graduate"}
        for pc, cls, prob in zip(prob_cols, classes, proba):
            with pc:
                st.markdown(f"""
                <div class="prob-card {css_map.get(cls,'')}">
                    <div class="prob-value">{prob*100:.1f}%</div>
                    <div class="prob-label">{cls}</div>
                </div>""", unsafe_allow_html=True)

    # ── Analisis Faktor Risiko ────────────────────────────────────────────────
    with prob_col:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<h4>🔍 Analisis Faktor Risiko (berdasarkan feature importance)</h4>', unsafe_allow_html=True)

        def fmt_row(label, value, is_bad):
            cls = "bad" if is_bad else "good"
            return f"""
            <div class="feature-row">
                <span class="feature-name">{label}</span>
                <span class="feature-value {cls}">{value}</span>
            </div>"""

        # Approval Rate Sem2 — fitur #1
        ar2_bad = ar2 < 0.5
        rows = fmt_row("Approval Rate Sem 2 🥇", f"{ar2:.2f}", ar2_bad)

        # Tuition — fitur #2
        tuit_bad = tuition_up_to_date == 0
        rows += fmt_row("SPP Tepat Waktu ⭐", "Tidak ❌" if tuit_bad else "Ya ✅", tuit_bad)

        # Unit Disetujui Sem2 — fitur #3
        cu2_bad = cu2_approved == 0
        rows += fmt_row("Unit Disetujui Sem 2 ⭐", str(cu2_approved), cu2_bad)

        # Unit Disetujui Sem1 — fitur #4
        cu1_bad = cu1_approved == 0
        rows += fmt_row("Unit Disetujui Sem 1 ⭐", str(cu1_approved), cu1_bad)

        # Beasiswa — fitur #5
        sch_bad = scholarship_holder == 0
        rows += fmt_row("Penerima Beasiswa ⭐", "Tidak" if sch_bad else "Ya ✅", sch_bad)

        # Debtor — fitur #6
        deb_bad = debtor == 1
        rows += fmt_row("Status Debitur ⭐", "Ya ❌" if deb_bad else "Tidak ✅", deb_bad)

        # Approval Rate Sem1 — fitur #7
        ar1_bad = ar1 < 0.5
        rows += fmt_row("Approval Rate Sem 1 ⭐", f"{ar1:.2f}", ar1_bad)

        # Grade trend
        trend_bad = trend < 0
        rows += fmt_row("Tren Nilai (Sem2−Sem1)", f"{trend:+.1f}", trend_bad)

        st.markdown(rows, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Ringkasan risiko
        n_bad = sum([ar2_bad, tuit_bad, cu2_bad, cu1_bad, sch_bad, deb_bad, ar1_bad, trend_bad])
        if n_bad >= 5:
            st.error(f"🔴 **{n_bad}/8 indikator risiko** aktif — risiko dropout sangat tinggi.")
        elif n_bad >= 3:
            st.warning(f"🟡 **{n_bad}/8 indikator risiko** aktif — perlu monitoring ketat.")
        else:
            st.success(f"🟢 **{n_bad}/8 indikator risiko** aktif — kondisi relatif baik.")

    # ── Detail Expander ───────────────────────────────────────────────────────
    with st.expander("🔧 Detail Fitur Engineering (kalkulasi otomatis)"):
        eng_data = {
            "Fitur": ["approval_rate_sem1","approval_rate_sem2","total_approved",
                      "avg_grade_both_sem","is_academically_active","grade_trend"],
            "Nilai": [f"{ar1:.3f}", f"{ar2:.3f}",
                      f"{input_df['total_approved'].values[0]:.0f}",
                      f"{grade:.2f}",
                      f"{input_df['is_academically_active'].values[0]:.0f}",
                      f"{trend:.2f}"],
            "Keterangan": [
                "Rasio lulus/ambil sem 1",
                "Rasio lulus/ambil sem 2 (fitur #1)",
                "Total unit lulus kedua semester",
                "Nilai rata-rata kedua semester",
                "1 jika ada evaluasi, 0 jika tidak",
                "Positif = nilai membaik",
            ]
        }
        st.dataframe(pd.DataFrame(eng_data), use_container_width=True, hide_index=True)

    with st.expander("📄 Lihat Semua Data Input"):
        st.dataframe(input_df.T.rename(columns={0: "Nilai"}), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:#94a3b8; font-size:0.78rem; padding: 8px 0 20px;'>
    🎓 Jaya Jaya Institut — Student Dropout Prediction System<br>
    Dibangun dengan XGBoost + SMOTE & Streamlit
</div>
""", unsafe_allow_html=True)
