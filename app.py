"""
app.py — FastAPI server exposing the TokenEconomistEnv as a REST API
=====================================================================
Run:  uvicorn app:app --reload

Endpoints:
  POST /reset      → start a new episode
  POST /step       → take one action (0=THINK, 1=ANSWER, 2=IMPROVE)
  GET  /state      → inspect current episode state
  GET  /info       → reward structure & environment details
  GET  /           → health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from environment import TokenEconomistEnv, THINK, ANSWER, IMPROVE, ACTION_NAMES

app = FastAPI(
    title="Token-Economist RL Environment",
    description=(
        "An RL environment that trains agents to self-regulate their reasoning "
        "budget — directly analogous to how LLMs decide how many tokens to "
        "spend on chain-of-thought reasoning."
    ),
    version="1.0.0",
)

# Single shared environment instance (stateful, one game at a time)
_env = TokenEconomistEnv(budget=10, verbose=False)
_last_obs: dict = {}
_episode_done: bool = True
_ep_reward: float = 0.0


# ── Pydantic models ───────────────────────────────────────────────────────────

class Observation(BaseModel):
    question:         str
    remaining_budget: int
    current_step:     int

class StepRequest(BaseModel):
    action: int           # 0=THINK, 1=ANSWER, 2=IMPROVE_POLICY

class StepResponse(BaseModel):
    observation: Observation
    reward:      float
    done:        bool
    info:        dict

class StateResponse(BaseModel):
    observation:    Observation
    episode_done:   bool
    episode_reward: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=Observation, summary="Start a new episode")
def reset():
    global _last_obs, _episode_done, _ep_reward
    obs            = _env.reset()
    _last_obs      = obs
    _episode_done  = False
    _ep_reward     = 0.0
    return Observation(**obs)


@app.post("/step", response_model=StepResponse, summary="Take one action")
def step(req: StepRequest):
    global _last_obs, _episode_done, _ep_reward
    if _episode_done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset first.")
    if req.action not in (THINK, ANSWER, IMPROVE):
        raise HTTPException(
            status_code=422,
            detail="action must be 0 (THINK), 1 (ANSWER), or 2 (IMPROVE)."
        )

    obs, reward, done, info = _env.step(req.action)
    _last_obs     = obs
    _episode_done = done
    _ep_reward   += reward

    return StepResponse(
        observation=Observation(**obs),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=StateResponse, summary="Inspect current state")
def state():
    if not _last_obs:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return StateResponse(
        observation=Observation(**_last_obs),
        episode_done=_episode_done,
        episode_reward=_ep_reward,
    )


@app.get("/info", summary="Environment reward structure")
def info():
    return {
        "name": "Token-Economist RL",
        "description": "RL environment for training agents to budget reasoning tokens",
        "actions": {
            "0": "THINK — boost confidence, costs -0.2",
            "1": "ANSWER — end episode, reward = 10.0 - (steps × 0.2) if correct, -5.0 if wrong",
            "2": "IMPROVE — adjust policy threshold, costs -0.5",
        },
        "reward_structure": {
            "correct_answer": "10.0 - (steps_taken × 0.2)",
            "wrong_answer": -5.0,
            "think_cost": -0.2,
            "improve_cost": -0.5,
            "timeout": -2.0,
        },
        "budget": _env.max_budget,
    }


@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "name": "Token-Economist RL", "docs": "/docs"}