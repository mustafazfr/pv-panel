# 🌞 PV Panel Defect Classification (Güneş Paneli Hata Tespiti)

Bu proje, insansız hava araçları veya kameralar ile çekilen fotoğraflardan güneş panellerindeki 6 farklı sınıfı (**Temiz, Kuş Pisliği, Tozlu, Elektriksel Hasar, Fiziksel Hasar, Kar Kaplı**) tespit etmeyi amaçlayan bir Derin Öğrenme (Deep Learning) bilgisayarlı görü projesidir. 

**Projenin En Temel Disiplini:** Projede hiçbir önceden eğitilmiş ağırlık (Transfer Learning) veya İnce Ayar (Fine-Tuning) kullanılmamıştır. Tüm modeller ImageNet bilgisi olmadan "Sıfırdan (From Scratch)" eğitilmiştir. Optimizasyonlar sonucu geliştirilen Özel Mimari (**CustomCNN V3**), literatürdeki VGG ve MobileNet gibi modelleri geride bırakmış ve sadece bu göreve spesifik tasarlanmış olmanın avantajıyla %81+ başarı yakalamıştır (Lider modelimiz ise %89.47 başarıyla ResNet-Scratch olmuştur).

Projenin bilimsel serüveni, teknik zorlukları ve mimari matematik detayları için lütfen **[PROJECT_REPORT.md](PROJECT_REPORT.md)** dosyasına göz atınız.

---

## 📂 Proje Klasör ve Dosya Yapısı

```text
pv-panel/
│
├── dataset/                    # Veri setinin bulunduğu ana klasör
│   ├── train/                  # Eğitim seti
│   ├── val/                    # Doğrulama (Validation) seti
│   └── test/                   # Test seti (Görülmemiş nihai test verileri)
│       └── [6 Hata Sınıfı Klasörü: Bird-drop, Clean, Dusty, ...]
│
├── outputs_benchmark/          # Modellerin en güncel ve temiz benchmark çıktıları
│   ├── checkpoints/            # Eğitilen modellerin en iyi ağırlık dosyaları (.pth)
│   ├── plots/                  # Görselleştirme çıktıları
│   │   ├── eda/                # Keşifsel Veri Analizi grafikleri
│   │   ├── training/           # Epoch bazlı eğitim ve doğrulama Loss/Accuracy eğrileri
│   │   └── evaluation/         # Confusion Matrix (Karmaşıklık Matrisi), Grad-CAM ısı haritaları
│   └── results/                # Kapsamlı performans karşılaştırmaları (benchmark_results.csv vb.)
│
├── src/                        # Projenin kaynak kod modülleri (Core)
│   ├── models/                 # Model mimarilerinin tanımlandığı klasör (Custom ve SOTA modeller)
│   ├── config.py               # Merkezi ayarlar (Sabitler, Hyperparametreler, LR, Patience)
│   ├── dataset.py              # Albumentations veri artırımı ve PyTorch DataLoader servisi
│   ├── train.py                # Eğitim motoru (Loss func, SGD, CosineScheduler, MixUp, CutMix)
│   ├── visualize_eda.py        # Veri seti analizi görselleştirmeleri
│   ├── visualize_results.py    # Değerlendirme ve test raporlama servisi
│   └── gradcam.py              # XAI - Modellerin nereye odaklandığını gösteren (Isı Haritası) algoritma
│
├── run_benchmark_all.py        # Tüm 7 modeli (Custom ve SOTA) sıfırdan eğiten MASTER çalıştırıcı
├── requirements.txt            # Gerekli Python kütüphaneleri listesi
├── PROJECT_REPORT.md           # Rapor: Başarısızlıklar, model karşılaştırmaları ve bilimsel sonuçlar
└── README.md                   # Proje haritası (Şu an okuduğunuz dosya)
```

---

## 🚀 Nasıl Çalıştırılır?

1. **Bağımlılıkları Yükleyin:**  
   Proje için gerekli olan kütüphaneleri (`torch`, `albumentations` vb.) yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

2. **Veri Seti Hazırlığı:**  
   `dataset/` klasörü altına resim verilerinizin `train`, `val` ve `test` formatında ve altlarında etiket klasörleri olacak şekilde bulunduğundan emin olun.

3. **Master Benchmark: Tüm Modelleri Sıfırdan Eğitme ve Karşılaştırma**  
   Tüm SOTA modelleri ve CustomCNN mimarilerini tek bir seferde sıfırdan eğitip kıyaslamak için:
   ```bash
   python run_benchmark_all.py
   ```

Tüm eğitim ilerleyişleri, metrikler, Grad-CAM görselleştirmeleri ve hata matrisleri (confusion matrix) otomatik olarak `outputs_benchmark/` klasörüne (grafikler ve `.csv` dosyaları şeklinde) kaydedilecektir.
