# etilang.py
# pip install ultralytics opencv-python easyocr flask pillow numpy
import os, sqlite3, time, sys, csv
from datetime import datetime
from collections import deque

# Try to import heavy deps but handle missing model file gracefully
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import cv2
except Exception as e:
    print("Error importing cv2:", e, file=sys.stderr)
    raise

try:
    import numpy as np
except Exception as e:
    print("Error importing numpy:", e, file=sys.stderr)
    raise

try:
    import easyocr
except Exception as e:
    print("Error importing easyocr:", e, file=sys.stderr)
    raise

try:
    from flask import Flask, request, jsonify, send_file, abort
except Exception as e:
    print("Error importing flask:", e, file=sys.stderr)
    raise

# ----------------------------
# Bagian A: kode asli (tetap utuh)
# ----------------------------

# Config
MODEL_PATH = "best_plate.pt"   # ganti dengan model deteksi plat Anda jika ada
EVIDENCE_DIR = "evidence"
DB_PATH = "etilang.db"
CSV_LOG = "tickets_log.csv"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Performance tuning parameters (tidak menghapus logika asli)
# Naikkan FRAME_SKIP untuk memproses lebih jarang (kurangi beban)
FRAME_SKIP = 5                 # proses 1 dari setiap N frame (default dioptimalkan)
# DETECT_EVERY: jalankan deteksi/OCR hanya setiap N loop terproses
DETECT_EVERY = max(1, FRAME_SKIP)
# Hanya jalankan OCR untuk deteksi dengan confidence deteksi >= threshold
DETECT_CONF_THRESHOLD = 0.35
# Tampilan target height (lebih kecil = lebih cepat)
DEFAULT_TARGET_H = 360

# Inisialisasi model (fallback jika tidak ada)
det_model = None
if YOLO is not None:
    try:
        if os.path.exists(MODEL_PATH):
            det_model = YOLO(MODEL_PATH)
        else:
            print(f"Warning: '{MODEL_PATH}' not found. Falling back to 'yolov8n.pt' (may download).", file=sys.stderr)
            det_model = YOLO("yolov8n.pt")
    except Exception as e:
        print("Warning: failed to load ultralytics model:", e, file=sys.stderr)
        det_model = None
else:
    print("ultralytics not available; using OpenCV heuristic fallback for detection.", file=sys.stderr)

# init easyocr reader (gpu=False default; ubah ke True jika Anda punya GPU dan PyTorch CUDA)
ocr_reader = easyocr.Reader(['en'], gpu=False)

# DB init
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS tickets (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 plate TEXT,
                 score REAL,
                 timestamp TEXT,
                 location TEXT,
                 image_path TEXT,
                 raw_ocr TEXT)''')
    conn.commit(); conn.close()
init_db()

# init CSV log header if not exists
if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticket_id","plate","score","timestamp","location","image_path","raw_ocr"])

# Pipeline: detect -> crop -> enhance -> ocr -> store
def detect_and_ocr(frame, location="unknown"):
    """
    Original single-best detection pipeline (kept intact).
    """
    results = det_model(frame) if det_model is not None else []
    boxes = []
    for r in results:
        for b in r.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0])
            boxes.append((x1,y1,x2,y2,conf))
    if not boxes:
        return None
    boxes.sort(key=lambda x: x[4], reverse=True)
    x1,y1,x2,y2,conf = boxes[0]
    crop = frame[max(0,y1):y2, max(0,x1):x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    ocr_res = ocr_reader.readtext(enhanced, detail=1)
    if not ocr_res:
        plate_text = ""
        ocr_conf = 0.0
    else:
        ocr_res.sort(key=lambda x: x[2], reverse=True)
        plate_text = ocr_res[0][1].replace(" ", "")
        ocr_conf = float(ocr_res[0][2])
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"{EVIDENCE_DIR}/evidence_{ts}.jpg"
    # Simpan bukti hanya saat ticket dibuat (store_ticket) — tetap menulis di sini agar kompatibel
    cv2.imwrite(fname, frame)
    score = conf * ocr_conf
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO tickets (plate, score, timestamp, location, image_path, raw_ocr) VALUES (?,?,?,?,?,?)",
              (plate_text, float(score), ts, location, fname, str(ocr_res)))
    conn.commit()
    ticket_id = c.lastrowid
    conn.close()
    # append to CSV log
    try:
        with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([ticket_id, plate_text or "UNKNOWN", score, ts, location, fname, str(ocr_res)])
    except Exception as e:
        print("CSV log write error:", e, file=sys.stderr)
    return {"ticket_id": ticket_id, "plate": plate_text, "score": float(score), "image": fname}

# Flask API (bagian asli)
app = Flask(__name__)

@app.route("/submit_frame", methods=["POST"])
def submit_frame():
    if 'image' not in request.files:
        return jsonify({"error":"no image"}), 400
    file = request.files['image']
    location = request.form.get('location', 'unknown')
    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    res = detect_and_ocr(frame, location)
    if res is None:
        return jsonify({"status":"no_plate_detected"}), 200
    return jsonify(res), 201

@app.route("/ticket/<int:ticket_id>", methods=["GET"])
def get_ticket(ticket_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id,plate,score,timestamp,location,image_path,raw_ocr FROM tickets WHERE id=?", (ticket_id,))
    row = c.fetchone(); conn.close()
    if not row:
        return jsonify({"error":"not found"}), 404
    return jsonify(dict(id=row[0], plate=row[1], score=row[2], timestamp=row[3], location=row[4], image_path=row[5], raw_ocr=row[6]))

# ----------------------------
# Bagian B: tambahan untuk pemrosesan video
# (menjaga semua fungsi/variabel asli tetap utuh)
# ----------------------------

# Video processing config (tambahan)
# NOTE: FRAME_SKIP sudah didefinisikan di atas (dioptimalkan)
LINE_Y = 400                   # posisi garis virtual (pixel) untuk deteksi crossing
COOLDOWN_SEC = 10              # jeda untuk mencegah duplikasi tiket per plat
TRACK_MAX_AGE = 1.5            # detik, usia maksimal track tanpa update
HISTORY_LEN = 8                # panjang history centroid per track

# Simple track structure: {track_id: {"centroids": deque, "last_seen": ts, "plate": str}}
tracks = {}

def store_ticket(plate_text, score, frame, location="unknown", ocr_res=None):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"{EVIDENCE_DIR}/evidence_{ts}.jpg"
    cv2.imwrite(fname, frame)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO tickets (plate, score, timestamp, location, image_path, raw_ocr) VALUES (?,?,?,?,?,?)",
              (plate_text, float(score), ts, location, fname, str(ocr_res)))
    conn.commit()
    ticket_id = c.lastrowid
    conn.close()
    # append to CSV log
    try:
        with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([ticket_id, plate_text or "UNKNOWN", score, ts, location, fname, str(ocr_res)])
    except Exception as e:
        print("CSV log write error:", e, file=sys.stderr)
    return {"ticket_id": ticket_id, "plate": plate_text, "score": float(score), "image": fname}

def detect_and_ocr_list(frame):
    """
    Returns list of detections. If ultralytics model available, use it.
    Otherwise use a simple OpenCV contour heuristic to find candidate plate regions,
    then run EasyOCR on candidates.
    Each detection dict keeps bbox, centroid, det_conf, plate, ocr_conf, ocr_raw, crop.
    Optimized: only run OCR for detections with det_conf >= DETECT_CONF_THRESHOLD.
    """
    detections = []
    if det_model is not None:
        results = det_model(frame)
        for r in results:
            for b in r.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                # Skip low-confidence detections early to avoid OCR overhead
                if conf < DETECT_CONF_THRESHOLD:
                    continue
                cx = int((x1+x2)/2); cy = int((y1+y2)/2)
                crop = frame[max(0,y1):y2, max(0,x1):x2]
                if crop.size == 0:
                    continue
                # enhance and run OCR (resize crop first for speed/stability)
                try:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    # resize crop for OCR stability (if too small)
                    h_c, w_c = enhanced.shape[:2]
                    if w_c > 0 and w_c < 300:
                        scale_c = 300.0 / float(w_c)
                        new_hc = max(24, int(h_c * scale_c))
                        enhanced = cv2.resize(enhanced, (300, new_hc))
                    ocr_res = ocr_reader.readtext(enhanced, detail=1)
                except Exception:
                    ocr_res = []
                plate_text = ""
                ocr_conf = 0.0
                if ocr_res:
                    ocr_res.sort(key=lambda x: x[2], reverse=True)
                    plate_text = ocr_res[0][1].replace(" ", "")
                    ocr_conf = float(ocr_res[0][2])
                detections.append({
                    "bbox": (x1,y1,x2,y2),
                    "centroid": (cx,cy),
                    "det_conf": conf,
                    "plate": plate_text,
                    "ocr_conf": ocr_conf,
                    "ocr_raw": ocr_res,
                    "crop": crop
                })
    else:
        # OpenCV heuristic: detect rectangular contours likely to be plates
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w//2, h//2))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        edged = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # skip small
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) >= 4 and len(approx) <= 6:
                x,y,wc,hc = cv2.boundingRect(approx)
                ar = wc / float(hc) if hc>0 else 0
                # typical plate aspect ratio heuristic
                if 2.0 <= ar <= 6.0 and wc*hc > 2000:
                    # scale back to original size
                    x1 = int(x*2); y1 = int(y*2); x2 = int((x+wc)*2); y2 = int((y+hc)*2)
                    crop = frame[max(0,y1):min(y2, frame.shape[0]), max(0,x1):min(x2, frame.shape[1])]
                    if crop.size == 0:
                        continue
                    try:
                        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        enhanced = clahe.apply(gray_crop)
                        # resize for OCR
                        h_c, w_c = enhanced.shape[:2]
                        if w_c > 0 and w_c < 300:
                            scale_c = 300.0 / float(w_c)
                            new_hc = max(24, int(h_c * scale_c))
                            enhanced = cv2.resize(enhanced, (300, new_hc))
                        ocr_res = ocr_reader.readtext(enhanced, detail=1)
                    except Exception:
                        ocr_res = []
                    plate_text = ""
                    ocr_conf = 0.0
                    if ocr_res:
                        ocr_res.sort(key=lambda x: x[2], reverse=True)
                        plate_text = ocr_res[0][1].replace(" ", "")
                        ocr_conf = float(ocr_res[0][2])
                    cx = int((x1+x2)/2); cy = int((y1+y2)/2)
                    detections.append({
                        "bbox": (x1,y1,x2,y2),
                        "centroid": (cx,cy),
                        "det_conf": 0.5,  # heuristic confidence
                        "plate": plate_text,
                        "ocr_conf": ocr_conf,
                        "ocr_raw": ocr_res,
                        "crop": crop
                    })
    return detections

def clean_tracks():
    now = time.time()
    remove_ids = []
    for tid, t in list(tracks.items()):
        if now - t["last_seen"] > TRACK_MAX_AGE:
            remove_ids.append(tid)
    for tid in remove_ids:
        del tracks[tid]

def update_tracks(detections, max_dist=80):
    """
    Simple nearest-centroid tracker that updates global `tracks`.
    Returns mapping detection_index -> track_id.
    """
    assigned = {}
    used = set()
    track_items = list(tracks.items())
    for i, det in enumerate(detections):
        cx, cy = det["centroid"]
        best_id = None
        best_d = max_dist**2 + 1
        for tid, t in track_items:
            px, py = t["centroids"][-1]
            d = (px - cx)**2 + (py - cy)**2
            if d < best_d and tid not in used:
                best_d = d
                best_id = tid
        if best_id is not None:
            assigned[i] = best_id
            used.add(best_id)
            tracks[best_id]["centroids"].append((cx,cy))
            tracks[best_id]["last_seen"] = time.time()
            if det["plate"]:
                tracks[best_id]["plate"] = det["plate"]
        else:
            nid = f"t{int(time.time()*1000)%1000000}_{i}"
            dq = deque(maxlen=HISTORY_LEN)
            dq.append((cx,cy))
            tracks[nid] = {"centroids": dq, "last_seen": time.time(), "plate": det["plate"]}
            assigned[i] = nid
    clean_tracks()
    return assigned

def check_line_crossing(track_centroids, line_y):
    """
    Determine if a track crossed the horizontal line from above to below.
    track_centroids: deque of (x,y) with oldest first.
    """
    if len(track_centroids) < 2:
        return False
    ys = [p[1] for p in track_centroids]
    if ys[0] < line_y and ys[-1] >= line_y:
        return True
    return False

def process_video(video_path, location="unknown"):
    """
    Non-GUI batch processing (kept for compatibility).
    """
    if not video_path or not os.path.exists(video_path):
        default_name = "loki.mp4"
        if os.path.exists(default_name):
            video_path = default_name
            print(f"Using default video file '{default_name}'")
        else:
            raise RuntimeError(f"Cannot open video: '{video_path}' not found and default 'loki.mp4' missing.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    frame_idx = 0
    last_ticket_time = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue
        detections = detect_and_ocr_list(frame)
        if not detections:
            clean_tracks()
            continue
        assigned = update_tracks(detections)
        for i, det in enumerate(detections):
            tid = assigned.get(i)
            if not tid:
                continue
            track = tracks.get(tid)
            if not track:
                continue
            plate = track.get("plate", "")
            score = det["det_conf"] * det["ocr_conf"]
            if check_line_crossing(track["centroids"], LINE_Y):
                now = time.time()
                plate_to_use = plate or det.get("plate") or ""
                last = last_ticket_time.get(plate_to_use or f"__id_{tid}", 0)
                if now - last > COOLDOWN_SEC:
                    ocr_raw = det.get("ocr_raw")
                    if not plate_to_use:
                        crop = det.get("crop")
                        if crop is not None and crop.size != 0:
                            try:
                                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                                enhanced = clahe.apply(gray)
                                extra_ocr = ocr_reader.readtext(enhanced, detail=1)
                                if extra_ocr:
                                    extra_ocr.sort(key=lambda x: x[2], reverse=True)
                                    plate_to_use = extra_ocr[0][1].replace(" ", "")
                                    ocr_raw = extra_ocr
                            except Exception as e:
                                print("Fallback OCR error:", e, file=sys.stderr)
                    if not plate_to_use:
                        plate_to_use = "UNKNOWN"
                    ticket = store_ticket(plate_to_use, score, frame, location, ocr_raw)
                    last_ticket_time[plate_to_use or f"__id_{tid}"] = now
                    print("Ticket created on crossing:", ticket)
    cap.release()
    return {"status":"done"}

# ----------------------------
# GUI: play video with window and live overlay + side panel + controls (DITAMBAHKAN)
# ----------------------------
def play_and_display(video_path=None, location="local"):
    """
    Play video in a window, draw detections/tracks, and record plates on crossing.
    Left: video. Right: panel with recent plate list.
    Controls overlay: Close, Pause/Play, Replay, Step Left, Step Right.
    Press 'q' or click Close to quit playback.
    """
    if not video_path or not os.path.exists(video_path):
        default_name = "loki.mp4"
        if os.path.exists(default_name):
            video_path = default_name
            print(f"Using default video file '{default_name}'")
        else:
            raise RuntimeError("Video file not found and 'loki.mp4' missing.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    # video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = 0
    paused = False
    last_ticket_time = {}
    recent_list = []
    max_list = 12
    window_name = "ETILANG - Live (Left: Video | Right: Plate Log)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    panel_width = 360
    font = cv2.FONT_HERSHEY_SIMPLEX

    # control buttons
    controls = {
        "close": {"label": "X"},
        "pause": {"label": "Pause"},
        "replay": {"label": "Replay"},
        "step_left": {"label": "<<"},
        "step_right": {"label": ">>"}
    }

    mouse_state = {"clicked": False, "x": 0, "y": 0, "button": None}

    # only handle left button down; ignore right/middle
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_state["clicked"] = True
            mouse_state["x"] = x
            mouse_state["y"] = y
            mouse_state["button"] = "L"
    cv2.setMouseCallback(window_name, on_mouse)

    def reset_mouse():
        mouse_state["clicked"] = False
        mouse_state["x"] = 0
        mouse_state["y"] = 0
        mouse_state["button"] = None

    def seek_frame(cap_obj, target_frame):
        target_frame = max(0, min(total_frames - 1, int(target_frame)))
        cap_obj.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        return target_frame

    # PERFORMANCE TUNING
    DETECT_EVERY_LOCAL = DETECT_EVERY
    last_frame = None
    processed_counter = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                paused = True
                if total_frames > 0:
                    seek_frame(cap, total_frames - 1)
                    ret, frame = cap.read()
                if not ret:
                    break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            last_frame = frame.copy() if frame is not None else last_frame
        else:
            # when paused, reuse last_frame (no cap.read)
            frame = last_frame
            if frame is None:
                # nothing to show
                ret, frame = cap.read()
                if not ret:
                    break
                last_frame = frame.copy()

        if frame is None:
            break

        # resize video frame to fixed height for consistent layout
        target_h = DEFAULT_TARGET_H
        h, w = frame.shape[:2]
        scale = target_h / float(h)
        new_w = int(w * scale)
        frame_resized = cv2.resize(frame, (new_w, target_h))
        display_frame = frame_resized.copy()

        # run detection only every DETECT_EVERY_LOCAL frames to reduce load
        run_detection = (processed_counter % DETECT_EVERY_LOCAL == 0) and (not paused or (paused and processed_counter % (DETECT_EVERY_LOCAL*2) == 0))
        detections = []
        assigned = {}
        if run_detection:
            detections = detect_and_ocr_list(frame_resized)
            assigned = update_tracks(detections) if detections else {}
            # draw virtual line
            line_y = int(LINE_Y * scale)
            cv2.line(display_frame, (0, line_y), (display_frame.shape[1], line_y), (0,0,255), 2)
            # draw detections
            for i, det in enumerate(detections):
                x1,y1,x2,y2 = det["bbox"]
                # clamp coords
                x1 = max(0, min(display_frame.shape[1]-1, x1))
                x2 = max(0, min(display_frame.shape[1]-1, x2))
                y1 = max(0, min(display_frame.shape[0]-1, y1))
                y2 = max(0, min(display_frame.shape[0]-1, y2))
                plate = det.get("plate") or "?"
                conf = det.get("det_conf", 0)
                ocr_conf = det.get("ocr_conf", 0)
                tid = assigned.get(i)
                color = (0,255,0) if plate and plate != "UNKNOWN" else (0,165,255)
                cv2.rectangle(display_frame, (x1,y1), (x2,y2), color, 2)
                label = f"{plate} d{conf:.2f} o{ocr_conf:.2f}"
                cv2.putText(display_frame, label, (x1, max(10,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if tid:
                    cv2.putText(display_frame, tid, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            # evaluate crossing per track and store tickets
            for tid, t in list(tracks.items()):
                if check_line_crossing(t["centroids"], line_y):
                    plate = t.get("plate") or ""
                    score = 0.0
                    ocr_raw = None
                    for i, det in enumerate(detections):
                        if assigned.get(i) == tid:
                            score = det.get("det_conf",0) * det.get("ocr_conf",0)
                            ocr_raw = det.get("ocr_raw")
                            break
                    now = time.time()
                    key = plate or f"__id_{tid}"
                    last = last_ticket_time.get(key, 0)
                    if now - last > COOLDOWN_SEC:
                        plate_to_use = plate or "UNKNOWN"
                        ticket = store_ticket(plate_to_use, score, frame_resized, location, ocr_raw)
                        last_ticket_time[key] = now
                        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                        recent_list.insert(0, (ts, plate_to_use, round(float(score),3)))
                        recent_list = recent_list[:max_list]
                        print("Ticket created (GUI):", ticket)
        else:
            # still draw line for consistency
            line_y = int(LINE_Y * scale)
            cv2.line(display_frame, (0, line_y), (display_frame.shape[1], line_y), (0,0,255), 2)

        # build right panel
        panel_h = display_frame.shape[0]
        panel = np.full((panel_h, panel_width, 3), 30, dtype=np.uint8)  # dark background
        # header
        cv2.putText(panel, "Detected Plates", (12,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.line(panel, (8,44), (panel_width-8,44), (80,80,80), 1)
        # list entries
        y0 = 64
        line_h = 34
        for idx, entry in enumerate(recent_list):
            if idx >= max_list:
                break
            ts, plate, score = entry
            y = y0 + idx * line_h
            cv2.rectangle(panel, (10, y-18), (panel_width-10, y+14), (50,50,50), -1)
            cv2.putText(panel, f"{plate}", (16, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if plate!="UNKNOWN" else (0,165,255), 2)
            cv2.putText(panel, f"{ts}", (16, y+16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
            cv2.putText(panel, f"s:{score}", (panel_width-80, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        # combine left and right
        combined = np.hstack((display_frame, panel))
        cv2.imshow(window_name, combined)

        # draw control buttons (top-left area)
        btn_w, btn_h = 90, 34
        spacing = 8
        start_x = 12
        start_y = 8
        btns = []
        labels = [controls["close"]["label"], controls["pause"]["label"], controls["replay"]["label"],
                  controls["step_left"]["label"], controls["step_right"]["label"]]
        for i, lbl in enumerate(labels):
            bx = start_x + i * (btn_w + spacing)
            by = start_y
            btns.append((bx, by, bx + btn_w, by + btn_h, lbl))
        for (bx1, by1, bx2, by2, lbl) in btns:
            cv2.rectangle(combined, (bx1, by1), (bx2, by2), (60,60,60), -1)
            cv2.rectangle(combined, (bx1, by1), (bx2, by2), (180,180,180), 1)
            cv2.putText(combined, lbl, (bx1 + 8, by1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        pos_text = f"Frame: {frame_idx}/{total_frames}  FPS: {fps:.1f}  {'PAUSED' if paused else 'PLAY'}"
        cv2.putText(combined, pos_text, (12, combined.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow(window_name, combined)

        # handle mouse clicks on buttons
        if mouse_state["clicked"]:
            mx, my = mouse_state["x"], mouse_state["y"]
            reset_mouse()
            for (bx1, by1, bx2, by2, lbl) in btns:
                if bx1 <= mx <= bx2 and by1 <= my <= by2:
                    if lbl == controls["close"]["label"]:
                        cap.release()
                        cv2.destroyAllWindows()
                        return {"status":"closed_by_user"}
                    if lbl == controls["pause"]["label"]:
                        paused = not paused
                        controls["pause"]["label"] = "Play" if paused else "Pause"
                    if lbl == controls["replay"]["label"]:
                        seek_frame(cap, 0)
                        paused = True
                        controls["pause"]["label"] = "Play"
                    if lbl == controls["step_left"]["label"]:
                        step_frames = int(fps * 2)
                        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        seek_frame(cap, current - step_frames)
                        paused = True
                        controls["pause"]["label"] = "Play"
                    if lbl == controls["step_right"]["label"]:
                        step_frames = int(fps * 2)
                        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        seek_frame(cap, current + step_frames)
                        paused = True
                        controls["pause"]["label"] = "Play"
                    break  # only one button per click

        # keyboard controls as alternative
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):  # space toggles pause
            paused = not paused
            controls["pause"]["label"] = "Play" if paused else "Pause"
        if key == ord('r'):  # replay
            seek_frame(cap, 0)
            paused = True
            controls["pause"]["label"] = "Play"
        if key == ord('a'):  # step left
            step_frames = int(fps * 2)
            current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            seek_frame(cap, current - step_frames)
            paused = True
            controls["pause"]["label"] = "Play"
        if key == ord('d'):  # step right
            step_frames = int(fps * 2)
            current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            seek_frame(cap, current + step_frames)
            paused = True
            controls["pause"]["label"] = "Play"

        processed_counter += 1

    cap.release()
    cv2.destroyAllWindows()
    return {"status":"done"}

# Flask endpoint to process uploaded video (bagian asli)
@app.route("/process_video", methods=["POST"])
def api_process_video():
    if 'video' not in request.files:
        return jsonify({"error":"no video"}), 400
    file = request.files['video']
    location = request.form.get('location','unknown')
    tmp_path = f"tmp_{int(time.time())}.mp4"
    file.save(tmp_path)
    try:
        res = process_video(tmp_path, location)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({"error": str(e)}), 500
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return jsonify(res), 200

# ----------------------------
# Akhir skrip
# ----------------------------

if __name__ == "__main__":
    # Jika ingin langsung memutar dan melihat jendela GUI saat menjalankan skrip:
    try:
        print("Starting GUI player for 'loki.mp4' (press 'q' to quit)...")
        play_and_display("loki.mp4", location="local")
    except Exception as e:
        print("GUI player failed or loki.mp4 missing:", e, file=sys.stderr)
        # fallback: jalankan Flask server agar endpoint tetap tersedia
        app.run(host="0.0.0.0", port=8000, debug=True)
