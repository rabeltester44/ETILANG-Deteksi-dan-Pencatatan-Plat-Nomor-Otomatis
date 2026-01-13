# ETILANG-Deteksi-dan-Pencatatan-Plat-Nomor-Otomatis
Sistem sederhana untuk mendeteksi plat nomor dari video, menjalankan OCR, menyimpan bukti, dan mencatat tiket ke SQLite serta CSV. Menyediakan antarmuka GUI untuk pemutaran video dengan panel daftar plat dan kontrol pemutaran, serta endpoint Flask untuk upload dan pemrosesan.

Fitur Utama
Deteksi plat menggunakan model Ultralytics YOLO (fallback OpenCV heuristic bila model tidak tersedia).

OCR menggunakan EasyOCR dengan preprocessing CLAHE untuk meningkatkan akurasi.

Pencatatan hasil ke SQLite (etilang.db) dan CSV (tickets_log.csv).

GUI: jendela pemutaran video dengan panel kanan menampilkan daftar plat terbaru, tombol kontrol (Close, Pause/Play, Replay, Step Left, Step Right).

Endpoint HTTP /submit_frame dan /process_video untuk integrasi dan upload.

Persyaratan
Python 3.10+  

Paket Python: ultralytics, opencv-python, easyocr, flask, pillow, numpy.

  GPU dan versi PyTorch dengan CUDA untuk performa OCR/deteksi lebih baik.

Instalasi Singkat
Buat virtual environment dan aktifkan:

python3 -m venv venv

source venv/bin/activate

Pasang dependensi:

python3 -m pip install ultralytics opencv-python easyocr flask pillow numpy

atau

pip3 install -r requirements.txt
