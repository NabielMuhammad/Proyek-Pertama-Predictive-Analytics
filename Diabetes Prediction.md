## Laporan Proyek Machine Learning - Nabiel Muhammad Imjauzanansyah

## Domain Proyek
Masalah diabetes menjadi isu kesehatan global yang memerlukan perhatian khusus, terutama dalam hal deteksi dini. Berdasarkan laporan WHO (2023), lebih dari 400 juta orang hidup dengan diabetes di seluruh dunia, dan angka ini terus meningkat setiap tahunnya. Deteksi dini melalui analisis data kesehatan dapat membantu mencegah komplikasi serius, menurunkan biaya pengobatan, dan meningkatkan kualitas hidup pasien.

Proyek ini bertujuan membangun model klasifikasi berbasis machine learning untuk memprediksi apakah seseorang menderita diabetes atau tidak berdasarkan data klinis, seperti kadar glukosa, tekanan darah, dan indeks massa tubuh (BMI). Dataset yang digunakan adalah Pima Indians Diabetes Dataset.

Mengapa masalah ini penting untuk diselesaikan?
Deteksi dini diabetes dapat menyelamatkan nyawa. Menggunakan model prediktif yang akurat membantu institusi kesehatan memberikan perawatan lebih cepat dan tepat sasaran.

> Referensi:
World Health Organization. (2023). Diabetes. Retrieved from https://www.who.int/news-room/fact-sheets/detail/diabetes
Smith, J., Wang, T., & Lee, K. (2020). Machine Learning for Early Diabetes Detection. Journal of Medical Systems, 44(5), 101. https://doi.org/10.1007/s10916-020-1534-0

## Business Understanding
Problem Statements
- Bagaimana membangun model machine learning untuk memprediksi apakah pasien memiliki diabetes berdasarkan fitur medis?
- Bagaimana performa model KNN dalam mendeteksi kasus diabetes menggunakan data yang tersedia??

Goals
- Mengembangkan model klasifikasi yang mampu memprediksi dengan akurat kemungkinan diabetes berdasarkan data pasien.
- Mengevaluasi performa model KNN menggunakan metrik akurasi, precision, recall, dan F1-score.

Solution Statements
- Melakukan proses standardisasi fitur untuk meningkatkan akurasi KNN.
- Melakukan hyperparameter tuning untuk meningkatkan akurasi model.
- Menggunakan metrik akurasi dan classification report (precision, recall, F1-score) untuk - mengevaluasi performa model.

## Data Understanding
Dataset yang digunakan adalah diabetes.csv, yang berisi 768 observasi dengan 8 fitur prediktor dan 1 target label (Outcome).
Dataset yang digunakan: Pima Indians Diabetes Dataset
Sumber: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Fitur dalam dataset:
- Pregnancies: Jumlah kehamilan
- Glucose: Konsentrasi glukosa dalam plasma
- BloodPressure: Tekanan darah diastolik (mm Hg)
- SkinThickness: Ketebalan lipatan kulit triceps (mm)
- Insulin: Kadar insulin serum dua jam (mu U/ml)
- BMI: Indeks massa tubuh (berat dalam kg/(tinggi dalam m)^2)
- DiabetesPedigreeFunction: Fungsi riwayat diabetes keluarga
- Age: Usia dalam tahun
- Outcome: 1 jika mengidap diabetes, 0 jika tidak

Visualisasi distribusi kelas target dilakukan untuk melihat apakah data seimbang atau tidak, yang penting dalam pemilihan metrik evaluasi.

## Data Preparation
Tahapan pemrosesan data yang dilakukan adalah sebagai berikut:

Handling Missing Value
- Beberapa fitur seperti Glucose, BloodPressure, SkinThickness, Insulin, dan BMI memiliki nilai 0, yang secara medis tidak valid. Nilai 0 pada kolom-kolom ini dianggap sebagai missing value dan ditangani dengan pendekatan pengisian:
~~~
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
df[cols_with_zero] = df[cols_with_zero].fillna(df[cols_with_zero].median())
~~~

- Mengganti nilai 0 dengan median dari masing-masing kolom.

Standardisasi Fitur
- Karena algoritma KNN sensitif terhadap skala fitur, dilakukan standardisasi menggunakan StandardScaler dari Scikit-learn untuk semua fitur numerik. Hal ini penting untuk memastikan perhitungan jarak (Euclidean) menjadi relevan antar fitur.

Split Data
- Dataset dibagi menjadi 80% data latih dan 20% data uji menggunakan fungsi train_test_split dari Scikit-learn dengan parameter random_state=42 untuk memastikan reprodusibilitas.

> Tidak dilakukan: encoding, balancing, atau penghapusan duplikat karena tidak diperlukan pada dataset ini setelah pemeriksaan eksploratif.

## Modeling
K-Nearest Neighbors (KNN) merupakan algoritma non-parametrik yang tidak membangun model secara eksplisit, tetapi menyimpan data latih dan menggunakan prinsip kedekatan (distance-based voting) untuk memprediksi label.
- Menghitung jarak Euclidean antara data uji dan seluruh data latih.
- Memilih k tetangga terdekat (nilai k ditentukan dari eksperimen).
- Melakukan voting mayoritas berdasarkan label dari tetangga tersebut.

Parameter
- n_neighbors=5 (default), kemudian dieksplorasi nilai 3 hingga 15.
- metric='minkowski' (default), yang ekuivalen dengan Euclidean distance jika p=2.
- weights='uniform' (default), semua tetangga memiliki bobot yang sama.

Kelebihan
- Sederhana dan mudah diimplementasikan.
- Bekerja baik pada dataset yang seimbang.

Kekurangan
- Sensitif terhadap skala dan outlier.
- Boros memori karena menyimpan semua data latih.
- Waktu prediksi lambat untuk dataset besar.

## Evaluation
Evaluation
Metrik Evaluasi 
- Accuracy: Proporsi prediksi benar dibandingkan total prediksi.
- Precision: Kemampuan model menghindari false positive.
- Recall: Kemampuan model mendeteksi kasus positif.
- F1-Score: Harmonik rata-rata precision dan recall.

Contoh hasil evaluasi (diambil dari hasil notebook):
- Akurasi: 0.73
- Presisi: 0.61
- Recall: 0.69
- F1-score: 0.65

Interpretasi:
 Interpretasi
Model KNN menunjukkan performa yang cukup baik dengan akurasi 73%. F1-score sebesar 0.65 menunjukkan keseimbangan antara presisi dan recall, yang penting dalam konteks deteksi penyakit seperti diabetes.

🔗 Hubungan ke Business Understanding
Apakah menjawab problem statement?
Ya, model KNN dibangun dan mampu memprediksi diabetes berdasarkan fitur klinis.

Apakah mencapai goals?
Ya, model menunjukkan performa prediktif yang baik.

Apakah solusi berdampak?
Ya. Model prediktif ini bisa digunakan oleh institusi kesehatan untuk mendeteksi pasien berisiko lebih awal, yang memungkinkan intervensi medis dilakukan lebih cepat.


