# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

**Nama:** Angga Yulian Adi Pradana  
**Email:** anggayulian2004@gmail.com  
**Id Dicoding:** anggapradanaa

---

## Business Understanding

Jaya Jaya Institut merupakan institusi pendidikan perguruan tinggi yang telah berdiri sejak tahun 2000. Selama lebih dari dua dekade, institusi ini telah berhasil mencetak ribuan lulusan dengan reputasi akademik yang sangat baik di berbagai bidang, mulai dari agronomi, desain, pendidikan, keperawatan, jurnalistik, manajemen, hingga teknologi.

Namun demikian, di balik reputasi yang membanggakan tersebut, Jaya Jaya Institut menghadapi tantangan serius yang mengancam kualitas dan citra institusinya, yaitu **tingginya angka mahasiswa yang tidak menyelesaikan pendidikan mereka (dropout)**. Fenomena dropout ini bukan hanya berdampak pada mahasiswa secara individual, tetapi juga berdampak signifikan pada reputasi institusi, efisiensi operasional, serta target akreditasi dan peringkat institusi pendidikan.

Tingginya angka dropout menunjukkan adanya kesenjangan antara kebutuhan mahasiswa dan layanan yang diberikan institusi. Tanpa sistem deteksi dini yang efektif, permasalahan ini akan terus berlanjut dan semakin sulit ditangani.

### Permasalahan Bisnis

Berdasarkan pemahaman bisnis di atas, berikut adalah permasalahan bisnis yang akan diselesaikan dalam proyek ini:

1. **Tingginya Angka Dropout** – Sebanyak 32.1% dari total 4.424 mahasiswa tidak menyelesaikan studi, menjadi ancaman serius bagi reputasi dan kualitas institusi.

2. **Tidak Ada Sistem Deteksi Dini** – Jaya Jaya Institut belum memiliki sistem yang dapat mengidentifikasi mahasiswa berisiko dropout sejak dini, sehingga intervensi sering terlambat dilakukan.

3. **Keterbatasan dalam Pengambilan Keputusan Berbasis Data** – Keputusan penanganan mahasiswa masih bersifat reaktif dan belum memanfaatkan data historis secara optimal untuk membuat prediksi berbasis bukti.

4. **Kurangnya Pemahaman tentang Faktor Penyebab Dropout** – Institusi belum memiliki gambaran yang jelas tentang faktor-faktor utama yang mendorong mahasiswa untuk dropout, sehingga program intervensi tidak tepat sasaran.

### Cakupan Proyek

Proyek ini mencakup seluruh tahapan data science end-to-end, meliputi:

1. **Analisis Data Eksploratif (EDA)** – Memahami struktur data, distribusi variabel, hubungan antar fitur, dan insight dari data mahasiswa Jaya Jaya Institut.

2. **Data Preprocessing** – Encoding variabel target, feature engineering (6 fitur turunan baru), train-test split (80:20 stratified), dan penanganan class imbalance dengan SMOTE.

3. **Pemodelan Machine Learning** – Membangun dan membandingkan dua model klasifikasi (Random Forest dan XGBoost) dengan RandomizedSearchCV untuk hyperparameter tuning.

4. **Evaluasi Model** – Evaluasi performa model menggunakan accuracy, precision, recall, F1-score, confusion matrix, dan 5-Fold Stratified Cross-Validation.

5. **Analisis Feature Importance** – Mengidentifikasi faktor-faktor utama yang paling berpengaruh terhadap kemungkinan dropout mahasiswa sebagai dasar pembuatan dashboard.

6. **Business Dashboard** – Membangun dashboard interaktif 4 tab menggunakan Streamlit untuk memonitor performa dan risiko dropout mahasiswa secara menyeluruh.

7. **Prototype Sistem Machine Learning** – Membuat aplikasi web interaktif menggunakan Streamlit untuk memprediksi status mahasiswa secara real-time beserta analisis faktor risikonya.

### Persiapan

**Sumber data:**  
Dataset berasal dari Jaya Jaya Institut yang berisi informasi akademik, demografis, dan sosial-ekonomi mahasiswa. Dataset memiliki **4.424 entri** dengan **37 kolom** dan tidak memiliki missing values.

- Referensi: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict students' dropout and academic success*. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

**Setup environment:**

```bash
# 1. Clone atau unduh repository proyek

# 2. Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependensi
pip install -r requirements.txt

# 4. Jalankan notebook untuk melatih model
jupyter notebook student_dropout_analysis.ipynb

# 5. Jalankan Business Dashboard
streamlit run dashboard.py

# 6. Jalankan Prototype ML
streamlit run app.py
```

**requirements.txt:**
```
imbalanced-learn
joblib
matplotlib
numpy
pandas
plotly
scikit-learn
seaborn
streamlit
xgboost
```

---

## Business Dashboard

Dashboard monitoring performa mahasiswa dibangun menggunakan **Streamlit** dengan tema warna gradasi biru yang konsisten. Dashboard terdiri dari **4 tab** utama:

### Tab 1 — 📊 Overview
Menampilkan gambaran umum kondisi mahasiswa meliputi:
- **4 metric card** : Total mahasiswa, jumlah Dropout, Enrolled, dan Graduate
- **Donut chart** distribusi status mahasiswa dengan persentase dropout di tengah
- **Bar chart horizontal** dropout rate per jurusan (Top 10) dengan gradasi warna berdasarkan tingkat keparahan

### Tab 2 — 🔍 Analisis Faktor Risiko
Visualisasi faktor finansial dan demografis berdasarkan **feature importance XGBoost**:
- Pembayaran SPP vs Status *(Fitur #2)*
- Status Beasiswa vs Status *(Fitur #5)*
- Status Hutang (Debtor) vs Status *(Fitur #6)*
- Gender vs Status
- Distribusi kelompok usia saat mendaftar vs Status

### Tab 3 — 📈 Performa Akademik
Visualisasi faktor akademik berdasarkan **feature importance XGBoost**:
- Approval Rate Semester 1 & 2 *(Fitur #7 & #1 — terpenting)*
- Unit Disetujui Semester 1 & 2 *(Fitur #4 & #3)*
- Distribusi nilai rata-rata kedua semester
- Scatter plot korelasi nilai Semester 1 vs Semester 2

### Tab 4 — 🚨 Early Warning
Sistem peringatan dini menggunakan **model XGBoost** yang sudah dilatih:
- **4 metric card** risiko: Total dipantau, Risiko Tinggi, Sedang, dan Rendah
- **Threshold slider** (40%–80%) untuk menyesuaikan sensitivitas deteksi
- **Histogram** distribusi probabilitas dropout seluruh mahasiswa Enrolled
- **Bar chart** Top 10 jurusan dengan mahasiswa berisiko tinggi terbanyak
- **Scatter plot** Approval Rate Sem 2 vs Probabilitas Dropout
- **Tabel interaktif** mahasiswa berisiko lengkap dengan badge 🔴🟡🟢, highlight warna, filter, dan tombol **download CSV**

Dashboard juga dilengkapi **sidebar filter global** (Gender & Jurusan) yang berlaku untuk semua tab, serta **insight dinamis** di setiap tab yang menyesuaikan dengan filter yang dipilih.

> 🔗 **Link Dashboard:** *(akan ditambahkan setelah deployment ke Streamlit Community Cloud)*

---

## Menjalankan Sistem Machine Learning

Prototype sistem machine learning dibangun menggunakan **Streamlit** dan dapat digunakan untuk memprediksi status mahasiswa (Dropout, Enrolled, Graduate) secara real-time lengkap dengan analisis faktor risiko berbasis feature importance.

### Prasyarat

Pastikan model sudah dilatih terlebih dahulu melalui notebook sehingga file berikut tersedia:
```
model/
├── model.pkl           ← Model XGBoost terbaik
├── label_encoder.pkl   ← Label encoder untuk dekoding prediksi
└── feature_cols.pkl    ← Daftar fitur yang digunakan model
```

### Cara Menjalankan Secara Lokal

```bash
# 1. Pastikan berada di direktori proyek
cd path/to/ProyekAkhir_Penerapan_Data_Science

# 2. Install dependensi (jika belum)
pip install -r requirements.txt

# 3. Jalankan aplikasi Streamlit
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser pada alamat: `http://localhost:8501`

### Fitur Prototype

| Fitur | Deskripsi |
|-------|-----------|
| **Form Input Terstruktur** | Input data mahasiswa dibagi menjadi Data Pribadi, Keuangan, Semester 1 & 2, dan Data Akademik. Fitur tidak penting disembunyikan dalam expander agar tidak overwhelming |
| **Label ⭐ Feature Importance** | Setiap input fitur penting diberi label bintang dan nomor urut feature importance agar user tahu fitur mana yang paling berpengaruh |
| **Prediksi Status** | Memprediksi apakah mahasiswa akan Dropout 🔴, Enrolled 🔵, atau Graduate 🟢 |
| **Probabilitas per Kelas** | Menampilkan probabilitas untuk setiap kelas dalam bentuk metric card bergradasi biru |
| **Analisis Faktor Risiko** | Panel khusus menampilkan 8 indikator risiko berdasarkan feature importance — merah jika buruk, biru jika baik — beserta ringkasan "n/8 indikator risiko aktif" |
| **Detail Engineering** | Expander menampilkan nilai 6 fitur hasil feature engineering beserta keterangannya |

> 🔗 **Link Prototype Online:** *(akan ditambahkan setelah deployment ke Streamlit Community Cloud)*

---

## Conclusion

Proyek ini berhasil membangun sistem prediksi dropout mahasiswa untuk Jaya Jaya Institut menggunakan pendekatan machine learning end-to-end. Berikut adalah kesimpulan utama:

1. **Dataset** terdiri dari 4.424 mahasiswa dengan 36 fitur asli + 6 fitur hasil engineering, dan 3 kelas target: Dropout (32.1%), Graduate (49.9%), dan Enrolled (17.9%).

2. **Faktor utama penyebab dropout** berdasarkan feature importance XGBoost adalah:
   - **Approval Rate Semester 2** (0.1632) — sinyal terkuat, mahasiswa dropout hampir selalu memiliki approval rate mendekati 0
   - **Tuition Fees Up to Date** (0.0981) — mahasiswa yang tidak membayar SPP tepat waktu memiliki dropout rate 86.6% vs 24.7% yang membayar
   - **Unit Disetujui Semester 2 & 1** (0.0546 & 0.0403) — jumlah mata kuliah lulus sangat membedakan antar status
   - **Scholarship Holder** (0.0400) — penerima beasiswa memiliki dropout rate hanya 12.2% vs 38.7% non-penerima
   - **Debtor** (0.0296) — mahasiswa berhutang memiliki dropout rate 62.0% vs 28.3% non-debtor

3. **Model terbaik** adalah **XGBoost** dengan performa:
   - Accuracy: **0.7627**
   - F1-Score: **0.7608**
   - Precision: **0.7602**
   - Recall: **0.7627**
   - Mean CV F1-Score: **0.8596** (±0.0154) — stabil dan konsisten lintas fold

4. **Business Dashboard** 4 tab berhasil dibangun dengan sistem Early Warning yang mengintegrasikan model XGBoost untuk memonitor mahasiswa Enrolled secara real-time.

5. **Prototype ML** berhasil dibangun dengan antarmuka yang intuitif, menampilkan prediksi beserta analisis faktor risiko berbasis feature importance untuk mendukung keputusan akademik.

### Rekomendasi Action Items

- **Action Item 1 — Implementasi Sistem Early Warning Berbasis ML:** Integrasikan prototype sistem machine learning ke dalam sistem informasi akademik institusi. Gunakan Tab Early Warning pada dashboard untuk mengidentifikasi mahasiswa berisiko tinggi di awal semester sehingga intervensi dapat dilakukan jauh sebelum dropout terjadi.

- **Action Item 2 — Program Intervensi Keuangan Proaktif:** Data menunjukkan SPP tidak terbayar dan status debtor sangat berkorelasi dengan dropout. Institusi perlu membuat program cicilan, skema beasiswa darurat, atau program kerja paruh waktu di kampus untuk membantu mahasiswa dengan kesulitan keuangan sebelum mereka memutuskan untuk dropout.

- **Action Item 3 — Penguatan Bimbingan Akademik di Semester Awal:** Approval rate dan jumlah unit disetujui semester 1 & 2 adalah prediktor terkuat dropout. Institusi harus memperkuat program tutoring dan mentoring khususnya di semester pertama. Mahasiswa dengan nilai rendah atau jumlah unit disetujui yang sedikit harus segera diidentifikasi untuk pendampingan intensif.

- **Action Item 4 — Kebijakan Khusus untuk Mahasiswa Non-Tradisional:** Mahasiswa yang mendaftar di usia lebih tua (>25 tahun) memiliki risiko dropout lebih tinggi. Institusi perlu mengembangkan layanan khusus seperti jadwal kuliah yang lebih fleksibel, konseling karir, dan dukungan keseimbangan kerja-studi untuk kelompok ini.

- **Action Item 5 — Monitoring Berbasis Jurusan:** Dashboard menunjukkan variasi dropout rate yang signifikan antar jurusan. Institusi perlu melakukan audit kurikulum dan beban studi secara khusus pada jurusan-jurusan dengan dropout rate tertinggi, kemudian merancang program dukungan yang spesifik per jurusan.
