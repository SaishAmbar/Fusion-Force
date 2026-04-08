"""
inference.py — Baseline Inference Script for OpenEnv Evaluation
================================================================
Uses the OpenAI API client to evaluate an LLM within the Token-Economist
environment. Produces structured [START]/[STEP]/[END] logs for grading.

Usage:
    export HF_TOKEN=<your_token>
    python inference.py

Environment Variables:
    API_BASE_URL  — Base URL for the OpenAI-compatible API (default: https://api-inference.huggingface.co/v1)
    MODEL_NAME    — Model to use (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      — Hugging Face API token (required)
"""

import os
import json
import sys

from openai import OpenAI
from environment import TokenEconomistEnv, THINK, ANSWER, IMPROVE, ACTION_NAMES

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None


# ── Structured Logging ────────────────────────────────────────────────────────

def log_start(task: str, env_name: str, model: str):
    print(f"[START] task={task} env={env_name} model={model}")

def log_step(step: int, action: str, reward: float, done: bool, error=None):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}")

def log_end(score: float, steps: int, rewards: list):
    reward_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] score={score:.4f} steps={steps} rewards={reward_str}")


# ── LLM-based Agent ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an agent playing the Token-Economist RL environment.

You must decide how many reasoning tokens to spend on a math question.
You have 3 possible actions:
  0 = THINK  — spend a reasoning token to boost confidence (+0.2), costs -0.2 reward
  1 = ANSWER — submit your answer. Correct = +10.0 - (steps × 0.2), Wrong = -5.0
  2 = IMPROVE — meta-reasoning action, costs -0.5 reward

Your goal: maximize reward by thinking *just enough* to get the answer right,
but not so much that the per-token costs eat your reward.

Respond with ONLY a single digit: 0, 1, or 2."""

def llm_select_action(obs: dict, step_history: list) -> int:
    """Ask the LLM to pick an action given the current observation."""
    if not client:
        # Fallback: simple heuristic if no API token
        if obs["current_step"] < 3:
            return THINK
        return ANSWER

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Build context from history
    context = f"""Current state:
- Question: {obs['question']}
- Remaining budget: {obs['remaining_budget']}
- Current step: {obs['current_step']}
- Steps so far: {', '.join(step_history) if step_history else 'none'}

What action do you choose? Reply with 0, 1, or 2 only."""

    messages.append({"role": "user", "content": context})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=5,
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip()
        # Parse the action — take first digit found
        for ch in text:
            if ch in "012":
                return int(ch)
        return ANSWER  # default to answering if unparseable
    except Exception as e:
        print(f"  [WARN] LLM call failed: {e}", file=sys.stderr)
        # Fallback heuristic
        if obs["current_step"] < 3:
            return THINK
        return ANSWER


# ── Task Definitions ──────────────────────────────────────────────────────────

TASKS = [
    {"name": "easy_math",   "budget": 10, "description": "Simple arithmetic — small budget"},
    {"name": "medium_math", "budget": 7,  "description": "Arithmetic with tighter budget"},
    {"name": "hard_math",   "budget": 5,  "description": "Arithmetic with very tight budget"},
]


# ── Main Inference Loop ──────────────────────────────────────────────────────

def run_task(task: dict, n_episodes: int = 10):
    """Run a single task for n_episodes and return aggregate results."""
    env = TokenEconomistEnv(budget=task["budget"], verbose=False)
    all_rewards = []
    successes = 0

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_rewards = []
        step_history = []
        step_num = 0

        while not done:
            action = llm_select_action(obs, step_history)
            action_name = ACTION_NAMES.get(action, "UNKNOWN")
            obs, reward, done, info = env.step(action)
            step_num += 1
            ep_rewards.append(reward)
            step_history.append(action_name)

            log_step(step_num, action_name, reward, done,
                     error=info.get("error"))

        # Check if episode was successful (correct answer)
        if info.get("correct", False):
            successes += 1

        all_rewards.extend(ep_rewards)

    return {
        "total_episodes": n_episodes,
        "successes": successes,
        "accuracy": successes / n_episodes,
        "avg_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        "all_rewards": all_rewards,
    }


def main():
    print("=" * 65)
    print("  Token-Economist RL — Baseline Inference (OpenEnv)")
    print("=" * 65)

    if not HF_TOKEN:
        print("\n[WARN] HF_TOKEN not set. Using heuristic fallback agent.")
        print("       Set HF_TOKEN to use LLM-based inference.\n")

    n_episodes = 10
    all_results = {}

    for task in TASKS:
        print(f"\n--- Task: {task['name']} ({task['description']}) ---\n")
        log_start(task["name"], "token-economist", MODEL_NAME)

        results = run_task(task, n_episodes=n_episodes)
        all_results[task["name"]] = results

        # Laplacian-smoothed score: strictly in (0, 1), never 0.0 or 1.0
        score = (results["successes"] + 0.5) / (n_episodes + 1.0)

        log_end(
            score=score,
            steps=n_episodes,
            rewards=results["all_rewards"],
        )

        print(f"\n  Accuracy: {results['accuracy']:.1%} | "
              f"Avg reward: {results['avg_reward']:+.2f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Inference Summary")
    print("=" * 65)
    print(f"  {'Task':<16} {'Accuracy':>10} {'Avg Reward':>12}")
    print(f"  {'─' * 42}")
    for name, res in all_results.items():
        print(f"  {name:<16} {res['accuracy']:>9.1%} {res['avg_reward']:>+12.2f}")
    print()


if __name__ == "__main__":
    main()
