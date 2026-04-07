"""
TokenEconomistEnv — Self-Improving Thinking Budget RL Environment
=================================================================
An RL environment that trains agents to self-regulate their reasoning
budget — directly analogous to how LLMs like Llama decide how many
tokens to spend on chain-of-thought reasoning.

Actions:
    THINK      → add a reasoning token, small cost
    ANSWER     → end episode, reward penalizes excess thinking
    IMPROVE    → adjust internal policy threshold, moderate cost

Reward Design (Token Economy):
    Correct answer  →  10.0 - (steps_taken × 0.2)
    Wrong answer    →  -5.0
    Timeout         →  -2.0
    Each THINK      →  -0.2  (per-token cost)
    Each IMPROVE    →  -0.5  (meta-reasoning cost)
"""

import random
import math

# ── Action constants ──────────────────────────────────────────────────────────
THINK   = 0
ANSWER  = 1
IMPROVE = 2
ACTION_NAMES = {THINK: "THINK", ANSWER: "ANSWER", IMPROVE: "IMPROVE_POLICY"}


class TokenEconomistEnv:
    """
    RL Environment: agent answers math questions under a thinking budget.

    The agent is penalized per thought used and rewarded more for correct
    answers that used fewer thoughts — teaching it to be efficient with
    reasoning tokens.

    Observation (returned by reset/step):
        question         (str)   — question text
        remaining_budget (int)   — steps left
        current_step     (int)   — steps taken so far

    Hidden internal state:
        _true_answer     (int)   — correct answer
        _confidence      (float) — probability of answering correctly [0, 1]
    """

    # Reward structure — Token Economy
    REWARD_CORRECT_BASE = 10.0
    REWARD_PER_TOKEN    = -0.2    # cost per thought token
    REWARD_WRONG        = -5.0
    REWARD_TIMEOUT      = -2.0
    REWARD_THINK        = -0.2    # per-token cost for THINK action
    REWARD_IMPROVE      = -0.5    # meta-reasoning cost

    def __init__(self, budget: int = 10, verbose: bool = False):
        self.max_budget = budget
        self.verbose    = verbose
        # internal state — initialised by reset()
        self._question        = ""
        self._true_answer     = 0
        self._confidence      = 0.0
        self._budget          = 0
        self._step            = 0
        self._done            = False

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        """Start a new episode with a dynamically generated math question."""
        a = random.randint(10, 50)
        b = random.randint(2, 9)
        c = random.randint(5, 20)
        self._question    = f"({a} × {b}) + {c}"
        self._true_answer = (a * b) + c
        self._confidence  = random.uniform(0.1, 0.2)   # start with low confidence
        self._budget      = self.max_budget
        self._step        = 0
        self._done        = False
        if self.verbose:
            print(f"\n=== New episode: {self._question} = {self._true_answer} ===")
        return self._obs()

    def step(self, action: int) -> tuple:
        """
        Take one step.

        Returns:
            obs    (dict)  — next observation
            reward (float) — reward signal
            done   (bool)  — episode over?
            info   (dict)  — diagnostics
        """
        assert not self._done, "Episode is done — call reset() first."
        assert action in (THINK, ANSWER, IMPROVE), f"Unknown action: {action}"

        self._step += 1

        if action == THINK:
            return self._do_think()
        elif action == ANSWER:
            return self._do_answer()
        else:
            return self._do_improve()

    # ── Action handlers ───────────────────────────────────────────────────────

    def _do_think(self):
        self._confidence = min(1.0, self._confidence + 0.2)
        self._budget    -= 1
        reward           = self.REWARD_THINK
        done             = self._budget <= 0

        if self.verbose:
            print(f"  Step {self._step}: THINK  → conf={self._confidence:.2f}, "
                  f"budget={self._budget}  reward={reward}")

        if done:
            # Budget exhausted without answering → timeout
            self._done = True
            reward = self.REWARD_TIMEOUT
            if self.verbose:
                print(f"  ⏰ TIMEOUT — budget exhausted  reward={reward}")
            return self._obs(), reward, True, {"action": "THINK", "timeout": True,
                                                "confidence": self._confidence}

        return self._obs(), reward, False, {"action": "THINK", "confidence": self._confidence}

    def _do_answer(self):
        correct = random.random() < self._confidence
        if correct:
            # Token economy: reward decreases with more steps used
            reward = self.REWARD_CORRECT_BASE + (self._step * self.REWARD_PER_TOKEN)
        else:
            reward = self.REWARD_WRONG

        self._done = True

        if self.verbose:
            result = "CORRECT ✓" if correct else "WRONG ✗"
            print(f"  Step {self._step}: ANSWER → {result}  reward={reward:+.1f} "
                  f"(steps={self._step})")

        return self._obs(), reward, True, {
            "action": "ANSWER", "correct": correct,
            "confidence": self._confidence, "steps_used": self._step
        }

    def _do_improve(self):
        # The IMPROVE action — the agent's policy class modifies its own
        # threshold when it receives this signal.
        self._budget -= 1
        reward        = self.REWARD_IMPROVE
        done          = self._budget <= 0

        if self.verbose:
            print(f"  Step {self._step}: IMPROVE  reward={reward}")

        if done:
            self._done = True
            reward = self.REWARD_TIMEOUT
            if self.verbose:
                print(f"  ⏰ TIMEOUT — budget exhausted  reward={reward}")
            return self._obs(), reward, True, {"action": "IMPROVE", "timeout": True}

        return self._obs(), reward, False, {"action": "IMPROVE"}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _obs(self) -> dict:
        return {
            "question":         self._question,
            "remaining_budget": self._budget,
            "current_step":     self._step,
        }