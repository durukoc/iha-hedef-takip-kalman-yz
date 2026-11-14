# İHA Kamera Görüntülerinde Kalman Filtresi ve Yapay Zeka Destekli Kesintisiz Hedef Takip Simülasyonu

İnsansız Hava Araçlarının (İHA) elektro-optik kamera görüntülerinde **görüş engeli (oklüzyon)**, **yüksek irtifa ölçek küçülmesi** ve **yüksek sensör gürültüsü** altında dahi hedef takibini kesintisiz sürdüren ve gelecek yörüngeyi öngören modüler Python simülasyon sistemidir.

## Temel Özellikler

* **Hibrit Hedef Algılama (`app.py`):** Yüksek irtifadaki küçük araçları yakalamak için **YOLOv8n** (`imgsz=1280`) derin öğrenme modeli ile **MOG2** arka plan çıkarma + morfolojik filtreleme hareket dedektörü hibrit olarak birleştirilmiştir.
* **Çoklu-Sensör Kalman Filtresi (`kalman_filter.py`):** 4 boyutlu durum vektörü $[p_x, p_y, v_x, v_y]^T$ üzerinde çalışan doğrusal Kalman filtresi; düşük gürültülü kamera ölçümleri ($\sigma_{cam}=3.0\text{ px}$) ile yüksek gürültülü ikincil radar ölçümlerini ($\sigma_{radar}=18.0\text{ px}$) ardışık güncelleme (*sequential update*) ile füzyonlar.
* **Çevrimiçi Yapay Zeka Yörünge Tahmini (`ai_predictor.py`):** Hedefin son 8 karelik yer değiştirme vektörlerinden ($\Delta x, \Delta y$) kayan pencere yöntemiyle çevrimiçi (*online*) öğrenen **PyTorch MLP** ağı, gelecek 10 karelik hareket rotasını kestirir.
* **Etkileşimli Simülasyon ve Stres Testi Arayüzü (`app.py`):** Gerçek İHA videoları veya sentetik taktik senaryolar üzerinde çoklu/tekil hedef takibi, yapay bulut oklüzyonu, canlı Öklid hata grafikleri ve CSV telemetri raporu sunar.

## Kurulum ve Çalıştırma

```bash
# 1. Depoyu klonlayın
git clone https://github.com/<kullanici-adin>/iha-hedef-takip-kalman-yz.git
cd iha-hedef-takip-kalman-yz

# 2. Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# 3. Streamlit arayüzünü başlatın
streamlit run app.py
```

## Proje Dizin Yapısı

```text
iha-hedef-takip-kalman-yz/
├── app.py                  # Hibrit algılama, veri ilişkilendirme ve Streamlit GUI
├── kalman_filter.py        # Çoklu-Sensör Kalman Filtresi durum uzayı modülü
├── ai_predictor.py         # Çevrimiçi PyTorch MLP yörünge tahmin ağı
├── requirements.txt        # Python kütüphane bağımlılıkları
├── .gitignore              # Git hariç tutma kuralları
└── README.md               # Proje dokümantasyonu
```
