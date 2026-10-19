# Deployment Report

## Summary
- Platform scaffolded at vmn8n/ocuvision-platform
- API: FastAPI on 127.0.0.1:8000 (Bearer auth), health/system/jobs endpoints, training requires approval
- Job queue: SQLite, recovery, cancellation
- Acquisition: checksum, secure ZIP/TAR extraction, manifest validation stubs
- Preprocessing: validation, dedup (phash/dhash), metadata, label mapping, normalizer
- QC: audit, leakage check, contact sheets
- Training: ConvNeXt-Tiny scaffold + config + metrics + tests
- Security: classes preserved (13), token via env, disk guard
- Tests: 14 passed (API, acquisition, preprocessing, model5, security, jobs, leakage)
- Systemd template + scripts provided (not enabled)

## Disk
57GB free (after removing eye_dataset_v6). Reserve 25GB.

## GPU
Tesla V100-PCIE-32GB, torch 2.3.1+cu121 working.

## Status
All required core pieces implemented and smoke-tested. No destructive changes to existing repo. Ready for Stage 11 review; systemd not enabled until approved.
