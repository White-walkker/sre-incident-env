---
title: SRE Incident Response Environment
emoji: 🚨
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - sre
pinned: false
---

# SRE Incident Response Environment

An OpenEnv environment where AI agents learn to respond to production incidents.

## Description
Simulates a real microservices production system with failures. The agent acts as an on-call SRE — reads alerts, examines metrics and logs, takes actions to restore system health.

## Action Space
- `check_logs` — read logs for a service
- `check_metrics` — check CPU/memory/error rate
- `restart_service` — restart a degraded service
- `scale_up` — increase resources/replicas
- `rollback` — rollback to previous version
- `list_alerts` — list active alerts
- `list_services` — list all service statuses

## Observation Space
- `alerts` — active critical/warning alerts
- `services` — status of each microservice
- `logs` — recent log lines
- `available_actions` — valid actions
- `reward` — score for last action (0.0–1.0)
- `done` — episode complete flag
- `message` — human-readable feedback

## Tasks
| Task | Description | Baseline Score |
|------|-------------|----------------|
| easy | Single service OOMKilled — restart it | 1.000 |
| medium | DB connection pool exhausted — find root cause | 1.000 |
| hard | Kafka schema mismatch — multi-step investigation | 1.000 |

## Setup
```bash
docker build -t sre-env .
docker run -p 7860:7860 sre-env
```

## Baseline Inference
```bash
python3 baseline_inference.py
```

## API
- `POST /reset` — start new episode `{"task_id": "easy"}`
- `POST /step` — take action `{"action_type": "check_logs", "target": "payment-service"}`
- `GET /state` — episode metadata