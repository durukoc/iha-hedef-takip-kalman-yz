"""
Modül Adı : kalman_filter.py
Proje     : İHA Kamera Görüntülerinde Kalman Filtresi ve YZ Destekli Hedef Takibi
Açıklama  : 4 boyutlu durum uzayında (konum ve hız) çalışan Çoklu-Sensör Doğrusal
            Kalman Filtresi sınıfı. Elektro-optik kameradan gelen düşük gürültülü
            tespitler ile ikincil radar/telemetri sensöründen gelen yüksek gürültülü
            ölçümleri ardışık ölçüm güncellemesi (sequential update) ile birleştirir.
"""

import numpy as np


class MultiSensorKalmanFilter:
    """
    Durum Vektörü (State Vector): x = [px, py, vx, vy]^T  (4x1 sütun vektörü)
      - px : Hedefin görüntü düzlemindeki yatay merkez konumu (piksel)
      - py : Hedefin görüntü düzlemindeki dikey merkez konumu (piksel)
      - vx : Hedefin x eksenindeki anlık hız bileşeni (piksel / kare)
      - vy : Hedefin y eksenindeki anlık hız bileşeni (piksel / kare)
    """

    def __init__(self, dt=1.0, process_noise=1.5, cam_noise=3.0, radar_noise=18.0):
        # Ardışık iki video karesi arasındaki ayrık örnekleme zaman adımı
        self.dt = float(dt)

        # Başlangıç durum vektörü: [px=0, py=0, vx=0, vy=0]^T
        self.x = np.zeros((4, 1), dtype=float)

        # Durum Geçiş Matrisi (F): Sabit Hız (Constant Velocity) kinematik modeli
        # Yeni konum = Eski konum + Hız * dt ; Yeni hız = Eski hız
        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0,     0.0],
            [0.0, 0.0, 0.0,     1.0]
        ], dtype=float)

        # Ölçüm Gözlem Matrisi (H): Sensörler yalnızca [px, py] konumunu ölçer.
        # Hız bileşenleri [vx, vy] doğrudan ölçülmez; ardışık konumlardan kestirilir.
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=float)

        # Durum Tahmin Hata Kovaryans Matrisi (P):
        # Başlangıçta hedef konumu bilinmediği için köşegen elemanlar yüksek seçilir.
        self.P = np.eye(4, dtype=float) * 500.0

        # Süreç Gürültüsü Kovaryans Matrisi (Q):
        # Hedefin sabit hız modelinden sapma (ivmelenme / dönüş) belirsizliğini modeller.
        self.Q = np.eye(4, dtype=float) * float(process_noise)

        # Kamera ve Radar ölçüm gürültü kovaryans matrisleri (R = sigma^2 * I_2):
        # Kamera hassas (düşük R), radar ise daha gürültülü (yüksek R) tanımlanır.
        self.R_cam = np.eye(2, dtype=float) * (float(cam_noise) ** 2)
        self.R_radar = np.eye(2, dtype=float) * (float(radar_noise) ** 2)

        # Filtrenin ilk hedef tespiti ile başlatılıp başlatılmadığını izleyen bayrak
        self.initialized = False

    def initialize(self, x, y):
        """
        İlk hedef tespiti gerçekleştiğinde durum vektörünü hedefin başlangıç
        piksel koordinatlarına kilitler ve başlangıç hızlarını sıfır atar.
        """
        self.x = np.array([[float(x)], [float(y)], [0.0], [0.0]], dtype=float)
        self.P = np.eye(4, dtype=float) * 100.0
        self.initialized = True

    def predict(self):
        """
        1. Aşama - Zaman Güncellemesi (Time Update / Predict):
        Bir önceki (k-1) durum ve hız kestirimini kullanarak k adımındaki
        önsel (a priori) konumu ve tahmin hata kovaryansını hesaplar.
        """
        # Önsel durum tahmini: x_{k|k-1} = F * x_{k-1|k-1}
        self.x = self.F @ self.x

        # Önsel kovaryans tahmini: P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return float(self.x[0, 0]), float(self.x[1, 0])

    def _update(self, z, R):
        """
        2. Aşama - Ölçüm Güncellemesi (Measurement Update / Correction):
        Sensörden gelen z = [u, v]^T ölçümünü kullanarak tahmin hatasını düzeltir.
        """
        z_vec = np.array(z, dtype=float).reshape(2, 1)

        # Adım 2.1: İnovasyon (Ölçüm Artığı) -> Gerçek ölçüm ile tahmin farkı
        y = z_vec - (self.H @ self.x)

        # Adım 2.2: İnovasyon Kovaryansı (S) -> Tahmin belirsizliği + Sensör gürültüsü
        S = self.H @ self.P @ self.H.T + R

        # Adım 2.3: Optimal Kalman Kazancı (K)
        # Sensör gürültüsü (R) yüksekse K küçülür (ölçüme daha az güvenilir).
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Adım 2.4: Sonsal (a posteriori) durum vektörü güncellemesi
        self.x = self.x + (K @ y)

        # Adım 2.5: Sonsal hata kovaryans matrisi güncellemesi
        I = np.eye(4, dtype=float)
        self.P = (I - K @ self.H) @ self.P

    def update_camera(self, z_cam):
        """Elektro-optik kameradan gelen hassas konum ölçümü ile füzyon yapar."""
        self._update(z_cam, self.R_cam)

    def update_radar(self, z_radar):
        """İkincil radar/telemetri sensöründen gelen ölçüm ile füzyon yapar."""
        self._update(z_radar, self.R_radar)

    def get_state(self):
        """Filtrenin güncel kestirdiği konum (px, py) ve hız (vx, vy) değerlerini döner."""
        return (
            float(self.x[0, 0]),
            float(self.x[1, 0]),
            float(self.x[2, 0]),
            float(self.x[3, 0])
        )
