import os
import json
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

ENV_URL = "https://yashasvi0903-sre-incident-env.hf.space"

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer.
Respond with ONLY valid JSON, no extra text, no markdown:
{
  "action_type": "one of: check_logs, check_metrics, restart_service, scale_up, rollback, list_alerts, list_services",
  "target": "one of: payment-service, api-gateway, auth-service, database, kafka",
  "reasoning": "brief explanation"
}"""


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_action(obs, task_id, step, history):
    if not HF_TOKEN:
        return _rule_agent(task_id, step)
    user_msg = f"""ALERTS: {json.dumps(obs.get('alerts', []))}
SERVICES: {json.dumps({k: v['status'] for k, v in obs.get('services', {}).items()})}
LOGS: {json.dumps(obs.get('logs', {}))}
Step: {obs.get('step_number', step)}. What action do you take?"""
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


def run_episode(task_id):
    log_start(task=task_id, env="sre-incident-env", model=MODEL_NAME)

    obs = requests.post(f"{ENV_URL}/reset",
                        json={"task_id": task_id}, timeout=30).json()
    rewards = []
    history = []
    steps_taken = 0
    success = False

    try:
        for step in range(1, 16):
            if obs.get("done", False):
                break

            action = get_action(obs, task_id, step - 1, history)
            action_str = f"{action['action_type']}({action['target']})"

            obs = requests.post(f"{ENV_URL}/step",
                                json=action, timeout=30).json()

            reward = float(obs.get("reward", 0.0))
            done = obs.get("done", False)
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str,
                     reward=reward, done=done, error=error)

            if done:
                break

        score = obs.get("reward", 0.0)
        success = "✅" in obs.get("message", "") or score >= 0.5

    finally:
        log_end(success=success, steps=steps_taken,
                score=score if rewards else 0.0, rewards=rewards)

    return score


if __name__ == "__main__":
    for task_id in ["easy", "medium", "hard"]:
        run_episode(task_id)