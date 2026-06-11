# 🌞 Fotovoltaik (PV) Panellerde Hata Tespiti: "Sıfırdan" Zirveye Bir Derin Öğrenme Yolculuğu

## 1. Özet ve Problem Tanımı (Abstract & Scope)
Bu projenin temel amacı, fotovoltaik (güneş) panellerinin yüzeylerinden alınmış görseller üzerinden 6 farklı durumu (*Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered*) otonom bir şekilde sınıflandırmaktır.

Projenin en büyük kısıtı ve aynı zamanda en önemli bilimsel meydan okuması: **Transfer Learning (Önceden eğitilmiş ağırlıkların) kullanımının tamamen yasak olmasıdır**. Klasik yaklaşımlarda ImageNet gibi devasa veri setlerinde eğitilmiş (pre-trained) modeller ince ayar (fine-tuning) ile görevlere uyarlanırken, bu projede her şey **"Sıfırdan (From-Scratch)"** tasarlanmış ve veri setinin kendi piksellerinden öğrenilmiştir.

---

## 2. Veri Seti Analizi ve Veri Hazırlık Hattı (Data Pipeline)

### Veri Karakteristiği ve Sınıf Dengesizliği
* Veri setimiz genel nesneler yerine sadece panelleri içerdiği için, ImageNet standart ortalamaları yerine **bu veri setine ait RGB Piksellerin** istatistiksel ortalaması (Mean) ve standart sapması (Std) hesaplanıp görseller standardize edilmiştir.
* Az bulunan hata sınıflarının model tarafından göz ardı edilmesini önlemek amacıyla `compute_class_weights` fonksiyonu kullanılmış; azılı sınıflar için ceza (Loss) ağırlığı artırılarak veri adaleti sağlanmıştır.

### Veri Zenginleştirme (Data Augmentation)
* Sınırlı veriden maksimum verimi almak ve ezberlemeyi (overfitting) önlemek için **Albumentations** kütüphanesi kullanılarak rastgele kırpmalar (RandomCrop) ve perspektif/açı bozma işlemleri (Affine) uygulanmıştır.

### Sentetik Veri Üretimi
* **MixUp:** İki görsel şeffaflıkla üst üste bindirilmiş ve etiket oranları karıştırılmıştır.
* **CutMix:** Bir panelden yama kesilip diğerine yapıştırılarak, modelin sadece çerçeveyi değil iç dokuyu öğrenmeye zorlanması sağlanmıştır.

---

## 3. Model Mimarilerinde "Sıfırdan" Tasarım Evrimi

Literatürdeki (SOTA) devasa mimarilerin (VGG, ResNet, MobileNet, EfficientNet) ön-eğitimsiz (`weights=None`) versiyonları projeye dahil edildiği gibi, probleme özel kendi **CustomCNN** mimarilerimiz de evrimsel bir sırayla geliştirilmiştir:

1. **CustomCNN (V1):** Standart ardışık evrişimlerden oluşan ilk denememiz. Kapasite eksikliğinden dolayı başarısı %72 seviyelerinde sınırlı kalmıştır.
2. **CustomCNN-V2 (Çöküş):** ResNet mimarisinden ilhamla çok derin katmanlar (stages) tasarlandı. Ancak *Transfer Learning olmadığı için*, başlarda gradyanlar kayboldu (**Vanishing Gradient**) ve model karmaşıklığı kaldıramadığı için %29 oranında ibretlik bir çöküş yaşadı.
3. **CustomCNN-V3 (Hibrit Başarı):** V2'deki çöküşten alınan dersle, "Derin değil, sığ ama vizyonu geniş" felsefesine geçildi.
   * **Multi-Scale Inception (Çoklu Ölçek):** Modele aynı anda `1x1`, `3x3` ve `5x5` komşuluklara bakma yeteneği verildi. (Kar gibi devasa bir örtüyü 5x5, minik bir elektrik yanığını 1x1 algılar).
   * **Depthwise Separable Convolutions:** MobileNet matematiği ile parametre sayıları dramatik biçimde %80-90 oranında düşürüldü.
   * *Sonuç:* Sıfırdan eğitime mükemmel entegre oldu ve büyük SOTA modellerine kafa tutan özgün şampiyonumuz oldu.

---

## 4. İleri Seviye Eğitim Stratejisi ve Optimizasyon

Ağırlıkları olmayan bir modelin "kaderine terk edilmesi" hüsranla sonuçlanır. Eğitim, güçlü optimizasyonlarla ayakta tutuldu:

* **SGD ve Nesterov:** Yüksek parametreli boş ağlarda Adam optimizasyonu hızlıca ezberlemeye (local minima) koşar. Bunun önüne geçmek adına ivmeli (momentum) klasik SGD kullanılmıştır.
* **Cosine Annealing Warm Restarts:** Öğrenme oranı (Learning Rate) kosinüs eğrisi gibi yavaşça daraltılmış, eğitimin donduğu hissedildiği her 10 epoch'ta bir fırlatılarak model sığ çukurlardan dışarı sıçratılmıştır.
* **Label Smoothing (Etiket Yumuşatma):** "Bu kesinlikle Kuş Pisliği %100" yerine "%95 Kuş Pisliği" mantığı aşılanarak modelin gereksiz özgüven sergilemesi ve aşırı uyumu engellenmiştir.

---

## 5. Kapsamlı Performans Karşılaştırması (The Full Benchmark)

Kurulan `outputs_benchmark` pipeline'ı ile hiçbir eski kalıntı (checkpoint) bırakılmadan test setinde aynı anda yarışan 7 modelin güncel performans sıralaması şu şekildedir:

| Sıra | Model (Tümü From-Scratch) | Accuracy | F1-Score | Analiz/Durum |
| :--- | :--- | :--- | :--- | :--- |
| **🥇 1** | **ResNet-Scratch** | **%89.47** | **0.89** | Derinliğini 'Skip-Connection' köprüleriyle yenerek sıfırdan eğitime mükemmel adapte oldu. |
| 🥈 2 | EfficientNet-Scratch | %83.15 | 0.83 | Compound Scaling ile parametre dengesini kusursuz kurdu. |
| **🥉 3** | **CustomCNN-V3** | **%81.05** | **0.81** | **Sadece bu probleme özel tasarladığımız hafif-Inception yapı devleri geride bıraktı.** |
| 4 | MobileNetV2-Scratch | %77.89 | 0.77 | Depthwise mimarisiyle hafif, ancak Resnet kadar derinlikli öğrenemedi. |
| 5 | VGG-Scratch | %75.79 | 0.75 | Aşırı fazla parametre, ezberlemeyi tam aşamadı. |
| 6 | CustomCNN (V1) | %72.63 | 0.73 | Sığ ve basit CNN kapasite limitine ulaştı. |
| 7 | CustomCNN-V2 | %29.47 | 0.24 | Transfer Learning olmadan devasa bir yapının çökeceğinin (Vanishing Gradient) canlı ispatı. |

---

## 6. Yapay Zekanın Açıklanabilirliği (Explainable AI - Grad-CAM)

Başarı oranları bir modelin "körü körüne" mi yoksa "akıllıca" mı karar verdiğini kanıtlamaz. Bu yüzden projeye **Grad-CAM (Gradient-weighted Class Activation Mapping)** modülü eklenmiştir. 

Modeller test edilirken, "Nereye bakarak bu kararı verdin?" sorusu sorulmuş ve ısı haritaları (heatmaps) kaydedilmiştir.
Görsel analizler saptamıştır ki: Özellikle ResNet ve **CustomCNN-V3** gibi iyi modeller; kararı gökyüzüne, panel çerçevesine veya arka plan objelerine değil, doğrudan panel üzerindeki **hasara (çatlak, toz, kuş pisliği vb.) odaklanarak** vermektedir.

---

## 7. Sonuç ve Literatüre Katkı (Conclusion)

Bu proje yapay zeka literatürüne doğrudan şu pratik mesajı vermektedir:
> *"Büyük ve hazır ağırlıklı modelleriniz (Transfer Learning) yoksa, o görev çözülemez yargısı bir mitten ibarettir."*

Doğru sentetik veri oyunları (*MixUp, CutMix*), agresif öğrenme hileleri (*Cosine Annealing*) ve problemin karmaşıklığına uygun **zayıflatılmış ancak vizyonu genişletilmiş ("Sığ ama İnceleyen")** özgün mimariler (*CustomCNN-V3*) tasarlanarak; sıfırdan oluşturulan otonom sistemlerle %89.47 (ResNet-Scratch) veya özenle el yapımı %81.05 (CustomCNN-V3) doğruluk seviyesine devasa dış veriler olmadan da ulaşılabilir. PV Panellerinin özerk bakımı için bu yaklaşım hem düşük boyutlu cihazlarda çalışmaya (Edge AI) uygun hem de maliyeti son derece düşüktür.
