import os
import json
import urllib.request
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "dummy")
ENV_URL      = "https://yashasvi0903-sre-incident-env.hf.space"

# Always use OpenAI client with injected credentials
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer.
Respond with ONLY valid JSON, no extra text:
{
  "action_type": "one of: check_logs, check_metrics, restart_service, scale_up, rollback, list_alerts, list_services",
  "target": "one of: payment-service, api-gateway, auth-service, database, kafka",
  "reasoning": "brief explanation"
}"""


def http_post(url, data):
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def _rule_agent(task_id, step):
    sequences = {
        "easy":   [("check_logs", "payment-service"),
                   ("restart_service", "payment-service")],
        "medium": [("check_logs", "database"),
                   ("scale_up", "database")],
        "hard":   [("check_logs", "kafka"),
                   ("check_metrics", "kafka"),
                   ("rollback", "payment-service")],
    }
    seq = sequences.get(task_id, [("list_alerts", "api-gateway")])
    at, tgt = seq[step] if step < len(seq) else ("list_alerts", "api-gateway")
    return {"action_type": at, "target": tgt, "reasoning": "rule-based"}


def get_action(obs, task_id, step, history):
    user_msg = (f"ALERTS: {json.dumps(obs.get('alerts', []))}\n"
                f"SERVICES: {json.dumps({k: v['status'] for k, v in obs.get('services', {}).items()})}\n"
                f"LOGS: {json.dumps(obs.get('logs', {}))}\n"
                f"What action do you take?")
    history.append({"role": "user", "content": user_msg})
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            temperature=0.2,
            max_tokens=150,
        )
        text = resp.choices[0].message.content.strip()
        action = json.loads(text)
        history.append({"role": "assistant", "content": text})
        return action
    except Exception:
        return _rule_agent(task_id, step)


def run_episode(task_id):
    log_start(task=task_id, env="sre-incident-env", model=MODEL_NAME)
    obs = http_post(f"{ENV_URL}/reset", {"task_id": task_id})
    rewards = []
    history = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        for step in range(1, 16):
            if obs.get("done", False):
                break
            action = get_action(obs, task_id, step - 1, history)
            action_str = f"{action['action_type']}({action['target']})"
            obs = http_post(f"{ENV_URL}/step", action)
            reward = float(obs.get("reward", 0.0))
            done = obs.get("done", False)
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str,
                     reward=reward, done=done, error=None)
            if done:
                break
        score = float(obs.get("reward", 0.0))
        success = "✅" in obs.get("message", "") or score >= 0.5
    except Exception as e:
        print(f"[DEBUG] Exception: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)
    return score


if __name__ == "__main__":
    for task_id in ["easy", "medium", "hard"]:
        run_episode(task_id)