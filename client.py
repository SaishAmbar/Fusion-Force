"""
client.py — Python client for the Token-Economist RL REST API
==============================================================
Usage:
    1. Start the server:  uvicorn app:app --reload
    2. Run this client:   python client.py
"""

import urllib.request
import json

BASE = "http://localhost:8000"

def _post(path, body=None):
    data = json.dumps(body or {}).encode()
    req  = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def _get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return json.loads(r.read())

THINK, ANSWER, IMPROVE = 0, 1, 2

def reset():
    return _post("/reset")

def step(action: int):
    return _post("/step", {"action": action})

def state():
    return _get("/state")

def info():
    return _get("/info")


# ── Demo run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Token-Economist RL — Client Demo")
    print("=" * 50)

    # Show environment info
    print("\n📋 Environment Info:")
    env_info = info()
    print(f"   Name: {env_info['name']}")
    print(f"   Budget: {env_info['budget']} steps")
    print(f"   Reward structure:")
    for k, v in env_info["reward_structure"].items():
        print(f"     {k}: {v}")

    # Run a demo episode
    print("\n▶  Starting demo episode …\n")
    obs = reset()
    print(f"   Question: {obs['question']}  |  budget={obs['remaining_budget']}")

    for i in range(3):
        r = step(THINK)
        print(f"   Step {i+1}: THINK  → reward={r['reward']:+.1f}, "
              f"budget={r['observation']['remaining_budget']}")

    r = step(ANSWER)
    print(f"   Step 4: ANSWER → reward={r['reward']:+.1f}, "
          f"done={r['done']}, info={r['info']}")

    s = state()
    print(f"\n   📊 Final episode reward: {s['episode_reward']:+.1f}")
    print("\n✅ Demo complete.")