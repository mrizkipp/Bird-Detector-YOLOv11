# 🎯 Bird Detector with YOLOv11 + NCNN Thesis

> **Rancang Bangun Penghalau Hama Burung Pipit dengan Gelombang Suara Audiosonik Berbasis Object Detection YOLOv11**

Proyek ini mengimplementasikan deteksi burung secara real-time menggunakan **YOLOv11**.  
Model dilatih menggunakan dataset burung, kemudian diekspor ke **NCNN** agar dapat dijalankan pada perangkat low-power seperti **Raspberry Pi + AI Accelerator**.  
Sistem ini terintegrasi dengan **relay + speaker** untuk menghasilkan suara audiosonik sebagai pengusir burung.

---

## 📂 Struktur Project
```bash
bird-detector-yolo11/
│
├── notebooks/
│   └── Train_YOLOv11_Fix.ipynb      # Notebook untuk training YOLOv11
│
├── src/                             # Kode utama
│   ├── app.py                       # Web App (Flask)
│   ├── main.py                      # Script deteksi + kontrol relay
│   └── yolo2ncnn.py                 # Konversi YOLOv11 → NCNN
│
├── templates/                       # File HTML untuk web
│   ├── index.html
│   ├── browse.html
│   └── session.html
│
├── requirements.txt                 # Dependency Python
├── README.md                        # Dokumentasi project
└── .gitignore                       # File/Folder yang diabaikan Git
