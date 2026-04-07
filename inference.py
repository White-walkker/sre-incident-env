import os
import json
import urllib.request

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "dummy")
ENV_URL      = "https://yashasvi0903-sre-incident-env.hf.space"

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer.
Respond with ONLY valid JSON, no extra text, no markdown:
{
  "action_type": "one of: check_logs, check_metrics, restart_service, scale_up, rollback, list_alerts, list_services",
  "target": "one of: payment-service, api-gateway, auth-service, database, kafka",
  "reasoning": "brief explanation"
}"""


def http_post(url, data, auth=None):
    """Generic HTTP POST using only stdlib urllib."""
    body = json.dumps(data).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    req = urllib.request.Request(
        url, data=body, headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def llm_call(messages):
    """
    Always calls the injected LLM proxy.
    Uses API_BASE_URL and API_KEY from environment — never hardcoded.
    """
    result = http_post(
        f"{API_BASE_URL}/chat/completions",
        {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 200,
        },
        auth=API_KEY
    )
    return result["choices"][0]["message"]["content"].strip()


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={error or 'null'}",
        flush=True
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} score={score:.3f} "
        f"rewards={rewards_str}",
        flush=True
    )


def get_action(obs, history):
    """
    Always calls LLM proxy — no fallback bypass.
    If LLM response is unparseable JSON, retries with a simpler prompt.
    """
    user_msg = (
        f"Active alerts: {json.dumps(obs.get('alerts', []))}\n"
        f"Service statuses: {json.dumps({k: v['status'] for k, v in obs.get('services', {}).items()})}\n"
        f"Recent logs: {json.dumps(obs.get('logs', {}))}\n"
        f"Choose your next action."
    )
    history.append({"role": "user", "content": user_msg})

    # First attempt
    text = llm_call(
        [{"role": "system", "content": SYSTEM_PROMPT}] + history
    )
    history.append({"role": "assistant", "content": text})

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Second attempt with stricter prompt
        retry_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                "Respond with ONLY this JSON and nothing else:\n"
                '{"action_type": "list_alerts", "target": "api-gateway", "reasoning": "checking alerts"}'}
        ]
        text2 = llm_call(retry_messages)
        try:
            return json.loads(text2)
        except json.JSONDecodeError:
            # Last resort — still goes through proxy, just hardcoded action
            return {
                "action_type": "list_alerts",
                "target": "api-gateway",
                "reasoning": "fallback after parse error"
            }


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
        # Clamp strictly between 0 and 1 (not 0.0, not 1.0)
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