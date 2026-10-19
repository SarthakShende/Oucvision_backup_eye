# n8n Integration Guide (Readiness)

n8n Cloud will connect via authenticated HTTPS (Tailscale Funnel or Cloudflare Tunnel). 
Do NOT expose SSH or shell execution.

## Connection
- Target: https://<tunnel-domain> (or configured)
- Local: http://127.0.0.1:8000

## Auth
Header: `Authorization: Bearer <OCUVISION_API_TOKEN>`

## Endpoints
- GET /health (no auth)
- GET /system/status
- GET /jobs, GET /jobs/{job_id}
- POST /jobs/discover-manifest, POST /jobs/download, POST /jobs/prepare-dataset, POST /jobs/validate-dataset, POST /jobs/train
- POST /jobs/{job_id}/cancel

## Job polling
Poll GET /jobs/{job_id} until status in {completed,failed,cancelled}. Respect 2s minimum interval.

## Training approval
POST /jobs/train returns requires_approval: true. Gate execution via n8n workflow (human-in-loop).

## Security
- SSRF/DNS rebinding protection (to be enforced in acquisition)
- Rate limiting, max request sizes
- Bearer token only; no public shell
