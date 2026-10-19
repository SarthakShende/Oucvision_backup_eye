# OcuVision Platform (VM-side)

Secure, standalone VM infrastructure for OcuVisionAI. 
- FastAPI VM Agent on 127.0.0.1:8000 with Bearer auth
- SQLite job queue with worker isolation
- Manifest-driven dataset acquisition (Kaggle/Zenodo/Figshare/HTTP)
- Image validation, deduplication, QC, metadata
- Model 5 (ConvNeXt-Tiny) training pipeline
- Systemd deployment, n8n integration ready

IMPORTANT: Synthetic dataset v7 preserved. Real clinical data quarantined.
