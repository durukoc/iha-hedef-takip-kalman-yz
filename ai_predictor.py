"""
Modül Adı : ai_predictor.py
Proje     : İHA Kamera Görüntülerinde Kalman Filtresi ve YZ Destekli Hedef Takibi
Açıklama  : Hedefin Kalman filtresinden elde edilen geçmiş yörünge noktalarından
            çevrimiçi (online) öğrenme gerçekleştiren ve gelecek 10 karelik
            olası hareket rotasını kestiren Çok Katmanlı Algılayıcı (MLP) modülü.
"""

import numpy as np
import torch
import torch.nn as nn


class TrajectoryMLP(nn.Module):
    """
    Hafif ve Gerçek Zamanlı Yörünge Tahmin Sinir Ağı (Multi-Layer Perceptron):
      - Giriş Katmanı  : Son 'seq_len' karenin yer değiştirme vektörleri (seq_len * 2)
      - 1. Gizli Katman: 32 nöron + ReLU doğrusal olmayan aktivasyon
      - 2. Gizli Katman: 32 nöron + ReLU doğrusal olmayan aktivasyon
      - Çıkış Katmanı  : Gelecek 'pred_len' karenin yer değiştirmeleri (pred_len * 2)
    """

    def __init__(self, seq_len=8, pred_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # İleri beslemeli tam bağlantılı (Fully Connected) sinir ağı katmanları
        self.net = nn.Sequential(
            nn.Linear(seq_len * 2, 32),   # Giriş boyutu -> 32 gizli nöron
            nn.ReLU(),                    # Doğrusal olmayan dönüşüm (manevra eğrileri)
            nn.Linear(32, 32),            # 32 nöron -> 32 nöron
            nn.ReLU(),
            nn.Linear(32, pred_len * 2)   # 32 nöron -> Gelecek (dx, dy) dizisi
        )

    def forward(self, x):
        """Giriş tensörünü ağ katmanlarından geçirerek gelecek yer değiştirmeleri üretir."""
        return self.net(x)


class OnlineTrajectoryPredictor:
    """
    Her bir takip edilen hedef (ID) için bağımsız olarak oluşturulan çevrimiçi
    eğitim ve yörünge kestirim yöneticisi. Ön-eğitimli bir veri setine ihtiyaç
    duymadan hedef hareket ettikçe kendi ağırlıklarını anlık olarak eğitir.
    """

    def __init__(self, seq_len=8, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.history = []  # Kalman filtresinden gelen (x, y) konum geçmişi listesi

        # Sinir ağı modeli, Adam eniyileyicisi ve Ortalama Kare Hata (MSE) kaybı
        self.model = TrajectoryMLP(seq_len=seq_len, pred_len=pred_len)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def add_point(self, x, y):
        """Kalman filtresinin güncel konum tahminini geçmiş tamponuna ekler."""
        self.history.append([float(x), float(y)])

        # Bellek kullanımını sabit tutmak ve eski manevraları unutmak için
        # tampon boyutu en fazla son 120 kare ile sınırlandırılır.
        if len(self.history) > 120:
            self.history.pop(0)

    def train_and_predict(self):
        """
        Geçmiş yörünge verisi üzerinde çevrimiçi ağırlık güncellemesi yapar ve
        gelecek 'pred_len' (varsayılan 10) kare için (x, y) koordinatlarını döner.
        """
        # Yeterli sayıda ardışık fark vektörü oluşmadıysa boş liste döndür
        if len(self.history) < self.seq_len + 2:
            return []

        pts = np.array(self.history, dtype=np.float32)

        # Konum yerine ardışık kareler arası farkların (dx, dy) kullanılması,
        # modelin ekranın her bölgesinde konumdan bağımsız genelleme yapmasını sağlar.
        deltas = np.diff(pts, axis=0)

        # Kayan pencere (sliding window) ile eğitim örnekleri oluşturulduğunda
        # ağ ağırlıklarını gerçek zamanlı olarak 5 iterasyon güncelle
        if len(deltas) >= self.seq_len + self.pred_len:
            X_train, Y_train = [], []
            num_samples = len(deltas) - self.seq_len - self.pred_len + 1

            for i in range(num_samples):
                # Girdi: i'den itibaren seq_len adet yer değiştirme vektörü
                x_window = deltas[i : i + self.seq_len].flatten()
                # Hedef Çıktı: Sonraki pred_len adet gerçek yer değiştirme vektörü
                y_window = deltas[
                    i + self.seq_len : i + self.seq_len + self.pred_len
                ].flatten()
                X_train.append(x_window)
                Y_train.append(y_window)

            X_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
            Y_tensor = torch.tensor(np.array(Y_train), dtype=torch.float32)

            self.model.train()
            for _ in range(5):
                self.optimizer.zero_grad()          # Önceki gradyanları sıfırla
                predictions = self.model(X_tensor)  # İleri yayılım (Forward pass)
                loss = self.criterion(predictions, Y_tensor)  # MSE hatasını hesapla
                loss.backward()                     # Geri yayılım (Backpropagation)
                self.optimizer.step()               # Ağ ağırlıklarını güncelle

        # Değerlendirme (Inference) modunda gelecek 10 adımın yer değiştirmesini kestir
        self.model.eval()
        with torch.no_grad():
            recent_deltas = deltas[-self.seq_len:].flatten()
            inp = torch.tensor(recent_deltas, dtype=torch.float32).unsqueeze(0)
            pred_deltas = (
                self.model(inp)
                .squeeze(0)
                .numpy()
                .reshape(self.pred_len, 2)
            )

        # Tahmin edilen yer değiştirmeleri son Kalman konumuna kümülatif ekle
        curr = pts[-1].copy()
        future_points = []
        for dx, dy in pred_deltas:
            curr = curr + np.array([dx, dy])
            future_points.append((int(curr[0]), int(curr[1])))

        return future_points
