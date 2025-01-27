# Hasta Yatış Süresi Tahmini Projesi

Bu proje, **New York State SPARCS Hospital Discharge** veri seti kullanılarak hastaların hastanede kalış sürelerini tahmin etmek için çeşitli veri analizi ve makine öğrenimi tekniklerini içermektedir. Proje, hem regresyon hem de sınıflandırma modelleri kullanarak hasta yatış süresi ile ilgili tahminler yapmayı amaçlamaktadır.



## Veri Seti

Proje, **Hospital Inpatient Discharges (SPARCS)** adlı veri setini kullanmaktadır. Bu veri seti, hastaların demografik bilgileri, klinik durumları, yatış türleri ve diğer ilgili özelliklerini içermektedir.

### Özellikler:
- **Length of Stay:** Hastanın hastanede kaldığı süre.
- **Age Group:** Hastanın yaş grubu.
- **Type of Admission:** Yatış türü.
- **Payment Typology:** Hastanın ödeme yöntemi.
- **Race, Ethnicity:** Hastanın demografik bilgileri.


## Proje Adımları

### 1. Veri Yükleme ve Keşif
- Veri seti pandas kullanılarak yüklenmiştir.
- Sütunların türleri ve eksik değerler incelenmiştir.
- **Length of Stay** değişkenindeki hatalı veriler (`120 +`) düzenlenmiştir.

### 2. Görselleştirme
- Seaborn kütüphanesi ile aşağıdaki görselleştirmeler yapılmıştır:
  - **Payment Typology vs Length of Stay**
  - **Age Group vs Length of Stay**
  - **Type of Admission vs Length of Stay**
  - **Medicare Patients for Age Group**

### 3. Özellik Dönüştürme ve Kodlama
- Gerekli olmayan sütunlar çıkarılmıştır.
- Kategorik sütunlar **OrdinalEncoder** ile sayısal değerlere dönüştürülmüştür.
- Eksik veriler uygun şekilde doldurulmuştur.

### 4. Modelleme
#### Regresyon:
- **DecisionTreeRegressor** modeli kullanılmıştır.
- Model, yatış süresini tahmin etmek için eğitilmiştir.
- Performans metriği olarak **RMSE** (Root Mean Square Error) hesaplanmıştır.

#### Sınıflandırma:
- Hastaların yatış süresi, belirli aralıklara bölünerek sınıflandırılmıştır.
- **DecisionTreeClassifier** kullanılarak sınıflandırma yapılmıştır.
- Model, doğruluk oranı (`accuracy`) ve sınıflandırma raporu (`classification_report`) ile değerlendirilmiştir.
