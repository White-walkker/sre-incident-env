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

An OpenEnv environment where an AI agent learns to respond to production incidents.

## Tasks
- **Easy**: Single service OOMKilled — restart it
- **Medium**: DB connection pool exhausted — find root cause
- **Hard**: Kafka schema mismatch — multi-step investigation

## Actions
check_logs, check_metrics, restart_service, scale_up, rollback, list_alerts, list_services

## API
- POST /reset — start episode
- POST /step — take action  
- GET /state — episode metadata