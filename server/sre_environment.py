from typing import Any, Optional
from uuid import uuid4


class SREEnvironment:

    def __init__(self):
        self._services = {}
        self._alerts = []
        self._logs = {}
        self._step_count = 0
        self._task_id = "easy"
        self._action_history = []
        self._episode_id = str(uuid4())
        self._done = False

    def reset(self, task_id: str = "easy") -> dict:
        self._task_id = task_id
        self._step_count = 0
        self._action_history = []
        self._episode_id = str(uuid4())
        self._done = False

        if task_id == "easy":
            self._services = {
                "payment-service": {"status": "degraded",
                    "cpu_pct": 98.0, "memory_pct": 99.0, "error_rate": 87.0},
                "api-gateway": {"status": "healthy",
                    "cpu_pct": 34.0, "memory_pct": 44.0, "error_rate": 2.0},
                "database": {"status": "healthy",
                    "cpu_pct": 41.0, "memory_pct": 62.0, "error_rate": 0.1},
            }
            self._alerts = [{"service": "payment-service",
                "severity": "critical",
                "message": "HTTP 503 rate > 85% for 5 minutes"}]
            self._logs = {"payment-service": [
                "[ERROR] OOMKilled: container exceeded memory limit",
                "[ERROR] Pod restarting... attempt 3",
                "[WARN] Memory usage at 99%"]}

        elif task_id == "medium":
            self._services = {
                "payment-service": {"status": "degraded",
                    "cpu_pct": 55.0, "memory_pct": 58.0, "error_rate": 45.0},
                "auth-service": {"status": "degraded",
                    "cpu_pct": 52.0, "memory_pct": 54.0, "error_rate": 38.0},
                "api-gateway": {"status": "degraded",
                    "cpu_pct": 48.0, "memory_pct": 50.0, "error_rate": 52.0},
                "database": {"status": "degraded",
                    "cpu_pct": 20.0, "memory_pct": 30.0, "error_rate": 0.5},
            }
            self._alerts = [
                {"service": "payment-service", "severity": "critical",
                 "message": "DB connection timeout"},
                {"service": "auth-service", "severity": "critical",
                 "message": "DB connection timeout"},
            ]
            self._logs = {"database": [
                "[ERROR] Connection pool exhausted (max=50, active=50)",
                "[WARN] Queue for connections: 127 requests waiting"]}

        elif task_id == "hard":
            self._services = {
                "kafka": {"status": "degraded",
                    "cpu_pct": 88.0, "memory_pct": 75.0, "error_rate": 31.0},
                "payment-service": {"status": "degraded",
                    "cpu_pct": 68.0, "memory_pct": 71.0, "error_rate": 22.0},
                "api-gateway": {"status": "healthy",
                    "cpu_pct": 41.0, "memory_pct": 48.0, "error_rate": 3.0},
                "database": {"status": "healthy",
                    "cpu_pct": 35.0, "memory_pct": 55.0, "error_rate": 0.2},
            }
            self._alerts = [
                {"service": "kafka", "severity": "critical",
                 "message": "Consumer lag > 100k messages on payments-topic"},
            ]
            self._logs = {"kafka": [
                "[ERROR] Deserialization error on payments-topic partition 3",
                "[ERROR] Schema mismatch: expected v2, got v3",
                "[WARN] Consumer group payments-consumer falling behind"]}

        return self._make_obs(reward=0.0,
            message=f"Incident started. Task: {task_id}.")

    def step(self, action_type: str, target: str,
             reasoning: str = "") -> dict:
        if self._done:
            return self._make_obs(reward=0.0,
                message="Episode already ended.")

        self._step_count += 1
        self._action_history.append((action_type, target))

        reward, message = self._apply(action_type, target)

        if self._is_resolved():
            self._done = True
            score = self._score()
            return self._make_obs(reward=score,
                message=f"✅ Incident resolved! Final score: {score:.2f}")

        if self._step_count >= 15:
            self._done = True
            score = self._score()
            return self._make_obs(reward=score,
                message=f"❌ Max steps reached. Score: {score:.2f}")

        return self._make_obs(reward=reward, message=message)

    def state(self) -> dict:
        return {
            "episode_id": self._episode_id,
            "task_id": self._task_id,
            "step_count": self._step_count,
            "is_done": self._done,
        }

    def _apply(self, action_type: str, target: str):
        if action_type == "check_logs":
            logs = self._logs.get(target,
                [f"No logs for {target}"])
            r = 0.05 if target == self._root_cause() else 0.01
            return r, f"Logs for {target}: " + " | ".join(logs)

        elif action_type == "check_metrics":
            svc = self._services.get(target)
            if not svc:
                return 0.0, f"Service {target} not found"
            r = 0.05 if target == self._root_cause() else 0.01
            return r, (f"{target}: CPU={svc['cpu_pct']}% "
                       f"MEM={svc['memory_pct']}% "
                       f"ERR={svc['error_rate']}% "
                       f"status={svc['status']}")

        elif action_type == "restart_service":
            if (target == "payment-service"
                    and self._task_id == "easy"):
                self._services["payment-service"] = {
                    "status": "healthy", "cpu_pct": 35.0,
                    "memory_pct": 52.0, "error_rate": 1.0}
                self._alerts = []
                return 0.3, "✅ payment-service restarted and healthy."
            return 0.0, f"Restarted {target} — no improvement."

        elif action_type == "scale_up":
            if (target == "database"
                    and self._task_id == "medium"):
                for k in self._services:
                    self._services[k] = {"status": "healthy",
                        "cpu_pct": 40.0, "memory_pct": 50.0,
                        "error_rate": 1.0}
                self._alerts = []
                return 0.3, "✅ DB scaled up. All services recovering."
            return 0.0, f"Scaled {target} — no improvement."

        elif action_type == "rollback":
            if (target == "payment-service"
                    and self._task_id == "hard"):
                self._services["kafka"] = {"status": "healthy",
                    "cpu_pct": 45.0, "memory_pct": 55.0,
                    "error_rate": 2.0}
                self._services["payment-service"] = {"status": "healthy",
                    "cpu_pct": 42.0, "memory_pct": 48.0,
                    "error_rate": 1.0}
                self._alerts = []
                return 0.3, "✅ Rolled back. Schema mismatch fixed."
            return 0.0, f"Rolled back {target} — no improvement."

        elif action_type == "list_alerts":
            if not self._alerts:
                return 0.0, "No active alerts."
            msgs = [f"[{a['severity']}] {a['service']}: {a['message']}"
                    for a in self._alerts]
            return 0.0, "Alerts: " + " | ".join(msgs)

        elif action_type == "list_services":
            msgs = [f"{n}: {s['status']} CPU={s['cpu_pct']}%"
                    for n, s in self._services.items()]
            return 0.0, "Services: " + " | ".join(msgs)

        return 0.0, f"Unknown action: {action_type}"

    def _root_cause(self) -> str:
        return {"easy": "payment-service",
                "medium": "database",
                "hard": "kafka"}.get(self._task_id, "")

    def _is_resolved(self) -> bool:
        return (len(self._alerts) == 0 and
                all(s["status"] == "healthy"
                    for s in self._services.values()))

    def _score(self) -> float:
        pairs = self._action_history
        score = 0.0
        if self._task_id == "easy":
            if ("check_logs", "payment-service") in pairs: score += 0.3
            if ("restart_service", "payment-service") in pairs: score += 0.5
            if self._is_resolved(): score += 0.2
        elif self._task_id == "medium":
            if ("check_logs", "database") in pairs: score += 0.3
            if ("scale_up", "database") in pairs: score += 0.4
            if self._is_resolved(): score += 0.3
        elif self._task_id == "hard":
            if ("check_logs", "kafka") in pairs: score += 0.2
            if ("check_metrics", "kafka") in pairs: score += 0.1
            if ("rollback", "payment-service") in pairs: score += 0.4
            if self._is_resolved(): score += 0.3
        return min(0.99, max(0.01, score))

    def _make_obs(self, reward: float, message: str) -> dict:
        return {
            "alerts": self._alerts,
            "services": self._services,
            "logs": self._logs,
            "available_actions": [
                "check_logs", "check_metrics", "restart_service",
                "scale_up", "rollback", "list_alerts", "list_services"
            ],
            "step_number": self._step_count,
            "reward": reward,
            "done": self._done,
            "message": message,
        }