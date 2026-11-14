"""
Modül Adı : app.py
Proje     : İHA Kamera Görüntülerinde Kalman Filtresi ve YZ Destekli Hedef Takibi
Açıklama  : YOLOv8n + MOG2 hibrit hedef algılama, En Yakın Komşu veri
            ilişkilendirme, Çoklu-Sensör Kalman Filtresi füzyonu, çevrimiçi
            MLP yörünge tahmini, oklüzyon stres testi ve Streamlit analiz GUI.
"""

import time
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from kalman_filter import MultiSensorKalmanFilter
from ai_predictor import OnlineTrajectoryPredictor

# Streamlit sayfa yapılandırması (Geniş ekran yerleşimi)
st.set_page_config(
    page_title="İHA Hedef Takip & Sensör Füzyon Simülasyonu",
    layout="wide"
)
st.title("İHA Kamera Görüntülerinde Kalman Filtresi & YZ Hedef Takip Simülasyonu")


@st.cache_resource
def load_yolo():
    """
    YOLOv8n (Nano) ağırlıklarını bellekte önbelleğe alır.
    Her simülasyon başlatılışında diske tekrar erişimi engelleyerek hız kazandırır.
    """
    return YOLO("yolov8n.pt")


@st.cache_data
def create_terrain_background(width=854, height=480):
    """
    Sentetik İHA simülasyon modu için topografik eşyükselti halkaları ve
    60 piksellik taktik koordinat ızgarası içeren arazi arka planı üretir.
    """
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    bg[:] = (34, 42, 36)  # Koyu yeşil-gri taktik arazi taban rengi

    rng = np.random.default_rng(42)
    for _ in range(18):
        cx = int(rng.integers(50, width - 50))
        cy = int(rng.integers(50, height - 50))
        radius = int(rng.integers(40, 140))
        cv2.circle(bg, (cx, cy), radius, (42, 52, 44), 2)

    # Dikey ve yatay taktik koordinat ızgara çizgileri
    step = 60
    for x in range(0, width, step):
        cv2.line(bg, (x, 0), (x, height), (50, 62, 52), 1)
    for y in range(0, height, step):
        cv2.line(bg, (0, y), (width, y), (50, 62, 52), 1)

    return bg


def get_synthetic_frame(frame_idx, maneuver_type, base_bg, is_occluded):
    """
    Seçilen manevra senaryosuna göre sentetik hedef aracın anlık konumunu
    hesaplar ve arazi üzerine çizer. Oklüzyon aktifse bulut örtüsü bindirir.
    """
    h, w, _ = base_bg.shape
    frame = base_bg.copy()
    t = float(frame_idx)

    # Manevra tipine göre gerçek hedef (ground-truth) kinematik yörüngesi
    if maneuver_type == "S-Kıvrımlı Taktik Rota":
        tx = 70.0 + (t * 3.1) % (w - 140)
        ty = (h / 2.0) + 125.0 * np.sin(t * 0.045)
    elif maneuver_type == "Dairesel Devriye Rotası":
        angle = t * 0.035
        tx = (w / 2.0) + 250.0 * np.cos(angle)
        ty = (h / 2.0) + 145.0 * np.sin(angle)
    else:  # Doğrusal Hızlanan Rota
        tx = 60.0 + (t * 2.4 + 0.008 * (t ** 2)) % (w - 120)
        ty = 90.0 + (t * 1.4) % (h - 180)

    # Hedef kara aracının arazi üzerine çizimi
    box_w, box_h = 46, 28
    x1, y1 = int(tx - box_w / 2), int(ty - box_h / 2)
    x2, y2 = int(tx + box_w / 2), int(ty + box_h / 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 195, 200), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240), 1)

    # Oklüzyon (görüş kesintisi) aktifse hedef üzerine bulut katmanı çiz
    if is_occluded:
        overlay = frame.copy()
        cv2.circle(overlay, (int(tx), int(ty)), 75, (90, 95, 105), -1)
        cv2.circle(overlay, (int(tx - 35), int(ty + 15)), 55, (105, 110, 120), -1)
        cv2.circle(overlay, (int(tx + 35), int(ty - 10)), 55, (100, 105, 115), -1)
        frame = cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)
        cv2.putText(
            frame, "GORUS ENGELI (BULUT)", (int(tx - 85), int(ty - 82)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 220), 1
        )

    return frame, [(tx, ty, box_w, box_h, 0.99)]


def select_specific_target(detections, lock_rule, frame_w=854, frame_h=480):
    """
    Tekil hedef kilitleme (Single-Target Lock) modunda başlangıçta hangi
    araca kilitlenileceğini operatör kuralına göre belirler.
    """
    if not detections:
        return None
    if lock_rule == "Sol Taraftaki İlk Araç":
        return min(detections, key=lambda d: d[0])
    elif lock_rule == "Sağ Taraftaki İlk Araç":
        return max(detections, key=lambda d: d[0])
    elif lock_rule == "Ekran Merkezine En Yakın Araç":
        return min(
            detections,
            key=lambda d: np.hypot(d[0] - frame_w / 2.0, d[1] - frame_h / 2.0)
        )
    else:  # En Yüksek Güven Skorlu Araç
        return max(detections, key=lambda d: d[4])


def detect_motion_targets(frame, bg_subtractor, frame_idx):
    """
    Yüksek irtifa İHA görüntülerindeki çok küçük silüetli araçları kaçırmamak
    için MOG2 arka plan çıkarma ve morfolojik filtreleme tabanlı hareket dedektörü.
    """
    fg_mask = bg_subtractor.apply(frame)

    # İlk 5 kare arka plan istatistik modelinin oturması için beklenir
    if frame_idx <= 5:
        return []

    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Morfolojik açma (gürültü silme) ve kapama (araç gövdesini birleştirme)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    motion_dets = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Yalnızca makul araç boyutlarındaki hareketli bölgeleri kabul et
        if 14 <= w <= 180 and 6 <= h <= 85:
            motion_dets.append((x + w / 2.0, y + h / 2.0, float(w), float(h), 0.75))

    return motion_dets


# ==============================================================================
# SOL KONTROL PANELİ (SİMÜLASYON VE FİLTRE PARAMETRELERİ)
# ==============================================================================
start_btn = st.sidebar.button(
    "Simülasyonu Başlat", type="primary", use_container_width=True
)

st.sidebar.header("1. Veri Kaynağı ve Takip Modu")
data_source = st.sidebar.radio(
    "Test Modu Seçin",
    ["Video Dosyası Yükle (.mp4, .avi)", "Sentetik İHA Simülasyonu (Video Gerektirmez)"]
)

tracking_mode = st.sidebar.radio(
    "Hedef Takip Stratejisi",
    ["Tüm Araçları Takip Et (Çoklu Hedef)", "Spesifik Tek Araca Kilitlen (Tekil Hedef)"]
)

lock_rule = "Ekran Merkezine En Yakın Araç"
if tracking_mode == "Spesifik Tek Araca Kilitlen (Tekil Hedef)":
    lock_rule = st.sidebar.selectbox(
        "Kilitlenecek Spesifik Araç Kriteri",
        [
            "Ekran Merkezine En Yakın Araç",
            "Sol Taraftaki İlk Araç",
            "Sağ Taraftaki İlk Araç",
            "En Yüksek Güven Skorlu Araç"
        ]
    )

uploaded_video = None
target_type = None
detector_mode = "Hibrit (YOLOv8 + Hareket Analizi)"
total_frames = 250
conf_thresh = 0.12
maneuver_type = "S-Kıvrımlı Taktik Rota"

if data_source == "Video Dosyası Yükle (.mp4, .avi)":
    uploaded_video = st.sidebar.file_uploader(
        "İHA Test Videosu Yükle", type=["mp4", "avi", "mov"]
    )
    detector_mode = st.sidebar.selectbox(
        "Algılama Motoru",
        ["Hibrit (YOLOv8 + Hareket Analizi)", "Sadece YOLOv8 (Derin Öğrenme)"]
    )
    target_type = st.sidebar.selectbox(
        "Takip Edilecek Hedef Sınıfı",
        options=[
            ("Araç (Otomobil / Kamyon / Otobüs)", [2, 3, 5, 7]),
            ("İnsan", [0]),
            ("Hava Aracı / İHA", [4])
        ],
        format_func=lambda x: x[0]
    )
    total_frames = st.sidebar.slider("Maksimum Kare Sayısı", 100, 1000, 250, step=50)
    conf_thresh = st.sidebar.slider("YOLO Güven Eşiği", 0.05, 0.60, 0.12, step=0.05)
else:
    maneuver_type = st.sidebar.selectbox(
        "Hedef Manevra Senaryosu",
        ["S-Kıvrımlı Taktik Rota", "Dairesel Devriye Rotası", "Doğrusal Hızlanan Rota"]
    )
    total_frames = st.sidebar.slider("Simülasyon Süresi (Kare)", 140, 360, 220, step=20)

st.sidebar.header("2. Sensör Füzyon Parametreleri")
radar_noise = st.sidebar.slider(
    "İkincil Sensör (Radar) Gürültüsü (px)", 5.0, 50.0, 16.0, step=1.0
)
process_noise = st.sidebar.slider(
    "Kalman Süreç Gürültüsü (Q)", 0.1, 10.0, 1.5, step=0.1
)

st.sidebar.header("3. Stres Testi (Kamera Kesintisi)")
enable_occlusion = st.sidebar.checkbox(
    "Oklüzyon (Bulut/Ağaç Arkası) Simüle Et", value=True
)
occ_start = st.sidebar.number_input(
    "Kesinti Başlangıç Karesi", min_value=15, value=75, step=5
)
occ_end = st.sidebar.number_input(
    "Kesinti Bitiş Karesi", min_value=25, value=135, step=5
)
full_blackout = st.sidebar.checkbox(
    "Kesintide Radar da Kopsun (Tam Kör Uçuş)", value=False
)

# ==============================================================================
# ANA EKRAN PANELİ (CANLI GÖRÜNTÜ AKIŞI VE ANLIK HATA METRİKLERİ)
# ==============================================================================
col_video, col_stats = st.columns([2, 1])
frame_placeholder = col_video.empty()
metric_c1, metric_c2, metric_c3 = col_stats.columns(3)
m_active_tracks = metric_c1.empty()
m_error_kalman = metric_c2.empty()
m_error_radar = metric_c3.empty()
chart_placeholder = col_stats.empty()


def run_simulation():
    """
    Ana benzetim döngüsü:
    1) Görüntü okuma ve YOLOv8n + MOG2 hibrit algılama
    2) Tüm aktif izler için Kalman predict() adımı
    3) En Yakın Komşu (85 px geçitleme) veri ilişkilendirme
    4) Kamera + Radar sensör füzyonu ve oklüzyon dayanıklılığı
    5) Çevrimiçi MLP ağı ile gelecek 10 karelik yörünge tahmini
    """
    is_synthetic = data_source == "Sentetik İHA Simülasyonu (Video Gerektirmez)"
    is_multi_target = tracking_mode == "Tüm Araçları Takip Et (Çoklu Hedef)"
    use_hybrid = detector_mode == "Hibrit (YOLOv8 + Hareket Analizi)"

    if not is_synthetic and uploaded_video is None:
        st.warning("Lütfen sol menüden bir test videosu yükleyin.")
        return

    cap = None
    model = None
    base_bg = None
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=120, varThreshold=36, detectShadows=False
    )

    if is_synthetic:
        base_bg = create_terrain_background(854, 480)
    else:
        model = load_yolo()
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

    # Aktif izler sözlüğü: {track_id: {"id", "kf", "ai", "box_size", "missed", "hits"}}
    tracks = {}
    next_track_id = 1
    telemetry_log = []
    frame_idx = 0

    def create_track(det):
        """Yeni tespit edilen hedef için bağımsız Kalman + MLP örneği başlatır."""
        nonlocal next_track_id
        cx, cy, w_b, h_b, _ = det
        kf = MultiSensorKalmanFilter(
            dt=1.0,
            process_noise=process_noise,
            cam_noise=3.0,
            radar_noise=radar_noise
        )
        kf.initialize(cx, cy)
        ai = OnlineTrajectoryPredictor(seq_len=8, pred_len=10)
        tid = next_track_id
        next_track_id += 1
        tracks[tid] = {
            "id": tid,
            "kf": kf,
            "ai": ai,
            "box_size": (int(w_b), int(h_b)),
            "missed": 0,
            "hits": 1
        }

    while frame_idx < total_frames:
        frame_idx += 1
        is_occluded = enable_occlusion and (occ_start <= frame_idx <= occ_end)
        detections = []

        # --- ADIM 1: GÖRÜNTÜ OKUMA VE HİBRİT HEDEF ALGILAMA ---
        if is_synthetic:
            frame, detections = get_synthetic_frame(
                frame_idx, maneuver_type, base_bg, is_occluded
            )
        else:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (854, 480))

            # YOLOv8n çıkarımı (yüksek irtifa küçük araçlar için imgsz=1280)
            results = model(
                frame,
                classes=target_type[1],
                conf=conf_thresh,
                imgsz=1280,
                verbose=False
            )[0]

            if len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                for box, c_score in zip(boxes, confs):
                    w_b, h_b = box[2] - box[0], box[3] - box[1]
                    if 8 <= w_b <= 240 and 5 <= h_b <= 180:
                        detections.append((
                            (box[0] + box[2]) / 2.0,
                            (box[1] + box[3]) / 2.0,
                            w_b,
                            h_b,
                            float(c_score)
                        ))

            # Hibrit modda MOG2 hareket tespitlerini mükerrerliği önleyerek ekle
            if use_hybrid:
                motion_dets = detect_motion_targets(frame, bg_subtractor, frame_idx)
                for m_det in motion_dets:
                    is_dup = any(
                        np.hypot(m_det[0] - d[0], m_det[1] - d[1]) < 35.0
                        for d in detections
                    )
                    if not is_dup:
                        detections.append(m_det)

        vis_frame = frame.copy()

        # --- ADIM 2: KALMAN ZAMAN GÜNCELLEMESİ (PREDICT) ---
        for tr in tracks.values():
            tr["kf"].predict()

        # --- ADIM 3: EN YAKIN KOMŞU VERİ İLİŞKİLENDİRME (85 PX EŞİK) ---
        matched_tracks = {}
        used_det_indices = set()
        if tracks and detections:
            pairs = []
            for tid, tr in tracks.items():
                px, py = tr["kf"].x[0, 0], tr["kf"].x[1, 0]
                for d_idx, det in enumerate(detections):
                    dist = np.hypot(det[0] - px, det[1] - py)
                    if dist < 85.0:
                        pairs.append((dist, tid, d_idx))
            pairs.sort(key=lambda x: x[0])
            for dist, tid, d_idx in pairs:
                if tid not in matched_tracks and d_idx not in used_det_indices:
                    matched_tracks[tid] = detections[d_idx]
                    used_det_indices.add(d_idx)

        # --- ADIM 4: YENİ İZ BAŞLATMA (ÇOKLU VEYA TEKİL HEDEF MODU) ---
        if is_multi_target:
            if not is_occluded:
                for d_idx, det in enumerate(detections):
                    if d_idx not in used_det_indices:
                        create_track(det)
        else:
            if len(tracks) == 0 and detections and not is_occluded:
                chosen_det = select_specific_target(detections, lock_rule)
                if chosen_det is not None:
                    create_track(chosen_det)
                    matched_tracks[next_track_id - 1] = chosen_det

        # --- ADIM 5: SENSÖR FÜZYONU, YZ TAHMİNİ VE GÖRSELLEŞTİRME ---
        frame_k_errors, frame_r_errors, to_delete = [], [], []
        for tid, tr in tracks.items():
            kf, ai = tr["kf"], tr["ai"]
            det = matched_tracks.get(tid)
            true_pt, radar_pt = None, None

            if det is not None:
                tr["missed"] = 0
                tr["hits"] += 1
                tr["box_size"] = (int(det[2]), int(det[3]))
                true_pt = (det[0], det[1])

                # İkincil radar sensörü simülasyonu (Gauss gürültülü ölçüm)
                radar_pt = (
                    true_pt[0] + np.random.normal(0, radar_noise),
                    true_pt[1] + np.random.normal(0, radar_noise)
                )

                # Oklüzyon yoksa kamera ölçümüyle güncelle; varsa kamerayı kapat
                if not is_occluded:
                    kf.update_camera(true_pt)

                # Tam kör uçuş seçili değilse radar ölçümüyle füzyonu sürdür
                if not (is_occluded and full_blackout):
                    kf.update_radar(radar_pt)
            else:
                if not is_occluded:
                    tr["missed"] += 1

            est_x, est_y, vx, vy = kf.get_state()

            # Kadraj dışına çıkan veya 15 kare boyunca kaybolan izleri sil
            out_of_bounds = (
                est_x < -10 or est_x > 864 or est_y < -10 or est_y > 490
            )
            if out_of_bounds or tr["missed"] > 15:
                to_delete.append(tid)
                continue

            # Yapay sinir ağı yörünge geçmişini güncelle ve 10 kare ileri tahmin yap
            ai.add_point(est_x, est_y)
            future_pts = ai.train_and_predict()

            # En az 2 kare doğrulanan kararlı hedefleri ekrana çiz
            if tr["hits"] >= 2:
                if true_pt is not None and not is_occluded:
                    cv2.circle(
                        vis_frame, (int(true_pt[0]), int(true_pt[1])),
                        4, (0, 255, 0), -1
                    )
                if radar_pt is not None and not (is_occluded and full_blackout):
                    cv2.drawMarker(
                        vis_frame, (int(radar_pt[0]), int(radar_pt[1])),
                        (0, 0, 255), cv2.MARKER_CROSS, 10, 1
                    )

                # Kalman füzyon takip kutusu ve ID etiketi (Mavi)
                w_b, h_b = tr["box_size"]
                bx1, by1 = int(est_x - w_b / 2), int(est_y - h_b / 2)
                bx2, by2 = int(est_x + w_b / 2), int(est_y + h_b / 2)
                cv2.rectangle(vis_frame, (bx1, by1), (bx2, by2), (255, 160, 0), 2)
                cv2.putText(
                    vis_frame, f"ID #{tid}", (bx1, max(15, by1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 60), 2
                )

                # Yapay zeka 10 karelik gelecek rota öngörüsü (Sarı çizgi)
                for i in range(1, len(future_pts)):
                    cv2.line(
                        vis_frame, future_pts[i - 1], future_pts[i],
                        (0, 255, 255), 2
                    )

                # Öklid konum hatalarını kaydet
                if true_pt is not None and radar_pt is not None:
                    frame_k_errors.append(
                        np.hypot(est_x - true_pt[0], est_y - true_pt[1])
                    )
                    frame_r_errors.append(
                        np.hypot(radar_pt[0] - true_pt[0], radar_pt[1] - true_pt[1])
                    )

        for tid in to_delete:
            del tracks[tid]

        # Üst durum bilgi şeridi (HUD)
        active_confirmed = sum(1 for t in tracks.values() if t["hits"] >= 2)
        if is_occluded:
            mode_text = (
                "TAM KOR UCUS (MODEL TAHMINI)"
                if full_blackout
                else "KAMERA KAYBI (RADAR + KALMAN)"
            )
            cv2.rectangle(vis_frame, (0, 0), (854, 480), (0, 0, 255), 5)
            cv2.putText(
                vis_frame, f"KARE {frame_idx}/{total_frames} | {mode_text}",
                (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2
            )
        else:
            cv2.putText(
                vis_frame,
                f"KARE {frame_idx}/{total_frames} | AKTIF HEDEF: {active_confirmed}",
                (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2
            )

        # Sağ panel canlı metrikleri ve hata grafiği güncellemesi
        m_active_tracks.metric("Takip Edilen Hedef", f"{active_confirmed} Araç")
        if frame_k_errors:
            mean_fk = float(np.mean(frame_k_errors))
            mean_fr = float(np.mean(frame_r_errors))
            telemetry_log.append({
                "Kare": frame_idx,
                "Aktif_Hedef_Sayisi": active_confirmed,
                "Okluzyon_Durumu": int(is_occluded),
                "Kalman Füzyon Hatası (px)": round(mean_fk, 2),
                "Ham Radar Hatası (px)": round(mean_fr, 2)
            })
            m_error_kalman.metric("Ort. Kalman Hatası", f"{mean_fk:.1f} px")
            m_error_radar.metric("Ort. Radar Hatası", f"{mean_fr:.1f} px")

        frame_placeholder.image(
            cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB), use_container_width=True
        )
        if len(telemetry_log) > 2 and frame_idx % 3 == 0:
            cols = ["Kalman Füzyon Hatası (px)", "Ham Radar Hatası (px)"]
            df_chart = pd.DataFrame(telemetry_log[-60:])[cols]
            chart_placeholder.line_chart(df_chart, height=210)

        if is_synthetic:
            time.sleep(0.025)

    if cap is not None:
        cap.release()

    # --- ADIM 6: SİMÜLASYON SONU PERFORMANS ÖZETİ VE CSV RAPORU ---
    if telemetry_log:
        df_res = pd.DataFrame(telemetry_log)
        mean_k = df_res["Kalman Füzyon Hatası (px)"].mean()
        mean_r = df_res["Ham Radar Hatası (px)"].mean()
        improvement = max(0.0, ((mean_r - mean_k) / mean_r) * 100.0)

        st.success("Simülasyon tamamlandı! Performans özeti aşağıdadır:")
        s1, s2, s3 = st.columns(3)
        s1.metric("Ortalama Kalman Hatası", f"{mean_k:.2f} px")
        s2.metric("Ortalama Ham Sensör Hatası", f"{mean_r:.2f} px")
        s3.metric("Sensör Füzyon İyileştirmesi", f"%{improvement:.1f}")

        st.download_button(
            label="Simülasyon Performans Raporunu İndir (CSV)",
            data=df_res.to_csv(index=False).encode("utf-8"),
            file_name="iha_hedef_takip_raporu.csv",
            mime="text/csv"
        )


if start_btn:
    run_simulation()
