# 💧 Water Quality Classification using Machine Learning

Bu proje, su örneklerinin **güvenli (Safe)** veya **güvensiz (Unsafe)** olup olmadığını tahmin etmek amacıyla geliştirilmiş bir **Makine Öğrenmesi sınıflandırma sistemidir**.

Projede üç farklı makine öğrenmesi algoritması eğitilmiş, performansları karşılaştırılmış ve sonuçlar grafiksel olarak analiz edilmiştir.

# 📌 Proje Amacı

Temiz içme suyunun belirlenmesi günümüzde önemli bir halk sağlığı problemidir.

Bu proje ile;

- Su örneklerinin güvenli olup olmadığını tahmin etmek
- Farklı makine öğrenmesi algoritmalarını karşılaştırmak
- Sınıflandırma performanslarını analiz etmek
- En başarılı modeli belirlemek
- Sonuçları grafiksel olarak sunmak

amaçlanmıştır.

---

# 🚀 Özellikler

- Su kalitesi sınıflandırması
- Binary Classification (Safe / Unsafe)
- Veri ön işleme
- Eksik veri temizleme
- Sınıf dengesizliği yönetimi
- Üç farklı makine öğrenmesi modeli
- ROC eğrileri
- Karışıklık matrisi
- Feature Importance analizi
- Korelasyon Heatmap
- Tkinter tabanlı grafik arayüzü

---

# 📊 Veri Seti

**Dosya Adı**

```
waterQuality1.csv
```

### Veri Seti Özellikleri

| Özellik | Değer |
|---------|-------|
| Toplam Veri | 7.996 |
| Feature Sayısı | 20 |
| Hedef Değişken | is_safe |
| Problem Türü | Binary Classification |

### Sınıf Dağılımı

| Sınıf | Veri Sayısı |
|-------|------------|
| Unsafe (0) | 7.084 |
| Safe (1) | 912 |

Veri seti ciddi bir **Class Imbalance** içermektedir.

Bu nedenle modellerde sınıf ağırlıkları kullanılmıştır.

---

# 🧠 Kullanılan Makine Öğrenmesi Algoritmaları

- Logistic Regression
- Random Forest
- XGBoost

---

# ⚙️ Kullanılan Teknolojiler

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib
- Tkinter

---

# 📈 Model Performansları

| Model | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
|--------|-----------|----------|-----------|---------|-----------|
| Logistic Regression | 82.06% | 0.8440 | 0.35 | 0.69 | 0.47 |
| Random Forest | 95.94% | 0.9802 | 0.95 | 0.68 | 0.79 |
| XGBoost | ~96% | ~0.98 | ~0.94 | ~0.70 | ~0.80 |

---

# 🏆 En Başarılı Model

Proje sonunda en başarılı model **XGBoost** olmuştur.

Başlıca nedenleri;

- Class Imbalance yönetimi
- Yüksek Accuracy
- Yüksek ROC-AUC
- Güçlü Genelleme Yeteneği
- Gradient Boosting yapısı
- Düzenlileştirme (Regularization)

sayesinde diğer modellere göre daha başarılı sonuçlar vermesidir.

---

# 📊 Kullanılan Performans Metrikleri

Projede aşağıdaki değerlendirme metrikleri kullanılmıştır.

- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curve
- ROC-AUC
- Confusion Matrix
- Feature Importance

---

# 📉 Görselleştirmeler

Uygulama aşağıdaki analizleri grafik olarak sunmaktadır.

- Accuracy Karşılaştırması
- ROC Eğrileri
- Karışıklık Matrisi
- Feature Importance
- Korelasyon Heatmap
- Sınıf Dağılımı

---

# 🧪 Veri Ön İşleme

Model eğitiminden önce aşağıdaki işlemler uygulanmıştır.

- Eksik verilerin temizlenmesi
- "#NUM!" değerlerinin kaldırılması
- Sayısal veri dönüşümleri
- Train/Test Split (%80 - %20)
- Stratified Sampling
- Class Weight dengelenmesi
- scale_pos_weight kullanımı (XGBoost)

---

# 📂 Proje Yapısı

```
Water-Quality-Classification/
│
├── waterQuality1.csv
├── su_kalitesi.py
├── README.md
├── requirements.txt
└── images/
```

---

# ▶️ Kurulum

Projeyi bilgisayarınıza klonlayın.

```bash
git clone https://github.com/kullaniciadi/Water-Quality-Classification.git
```

Klasöre girin.

```bash
cd Water-Quality-Classification
```

Gerekli kütüphaneleri yükleyin.

```bash
pip install -r requirements.txt
```

Programı çalıştırın.

```bash
python su_kalitesi.py
```

---

# 📷 Uygulama

Program çalıştırıldığında;

- Modeller eğitilir.
- Accuracy hesaplanır.
- ROC eğrileri oluşturulur.
- Karışıklık matrisleri gösterilir.
- Feature Importance analizi yapılır.
- Grafik arayüzü üzerinden tüm sonuçlar görüntülenebilir.

---

# 🔬 Kullanılan Özellikler

Veri setinde bulunan bazı önemli özellikler:

- Aluminium
- Ammonia
- Arsenic
- Cadmium
- Chloramine
- Chromium
- Copper
- Bacteria
- Viruses
- Lead
- Mercury
- Nitrates
- Nitrites
- Perchlorate
- Radium
- Uranium

Toplam **20 farklı su kalite parametresi** modele giriş olarak kullanılmaktadır.

---

# 📚 Akademik Amaç

Bu proje;

- Makine Öğrenmesi
- Veri Ön İşleme
- Binary Classification
- Ensemble Learning
- Gradient Boosting
- Model Karşılaştırma
- Veri Görselleştirme

konularını uygulamalı olarak göstermektedir.

---

# 👨‍💻 Geliştirici

**Yavuz Berke Pektaş**

Bilgisayar Mühendisliği

Niğde Ömer Halisdemir Üniversitesi

---

# 📄 Lisans

Bu proje eğitim ve akademik çalışmalar kapsamında geliştirilmiştir.
