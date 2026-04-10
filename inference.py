import os
import json
import urllib.request
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN", "dummy")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

ENV_URL = "https://yashasvi0903-sre-incident-env.hf.space"

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


def get_action(obs, history):
    user_msg = (
        f"ALERTS: {json.dumps(obs.get('alerts', []))}\n"
        f"SERVICES: {json.dumps({k: v['status'] for k, v in obs.get('services', {}).items()})}\n"
        f"LOGS: {json.dumps(obs.get('logs', {}))}\n"
        f"What action do you take?"
    )
    history.append({"role": "user", "content": user_msg})
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            temperature=0.2,
            max_tokens=200,
        )
        text = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": text})
        return json.loads(text)
    except Exception as e:
        print(f"[DEBUG] get_action error: {e}", flush=True)
        return {"action_type": "list_alerts",
                "target": "api-gateway",
                "reasoning": "fallback"}


def run_episode(task_id):
    log_start(task=task_id, env="sre-incident-env", model=MODEL_NAME)
    rewards = []
    history = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        obs = http_post(f"{ENV_URL}/reset", {"task_id": task_id})

        for step in range(1, 16):
            if obs.get("done", False):
                break
            action = get_action(obs, history)
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
        score = min(0.99, max(0.01, score))
        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)

    return score


if __name__ == "__main__":
    for task_id in ["easy", "medium", "hard"]:
        run_episode(task_id)