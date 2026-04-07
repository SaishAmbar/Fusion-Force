"""
agents.py — Three RL Agents for the Token-Economist Environment
================================================================
1. RandomAgent          — picks a random action every step (baseline)
2. FixedAgent           — THINK N times then ANSWER (deterministic)
3. SelfImprovingAgent   — threshold policy + epsilon-greedy + self-improvement
"""

import random
from environment import THINK, ANSWER, IMPROVE


# ── 1. Random Agent ───────────────────────────────────────────────────────────

class RandomAgent:
    """Baseline: selects a uniformly random action each step."""

    name = "RandomAgent"

    def select_action(self, obs: dict) -> int:
        return random.choice([THINK, ANSWER, IMPROVE])

    def on_episode_end(self, total_reward: float):
        pass   # nothing to learn

    def summary(self) -> str:
        return "RandomAgent — no policy, picks actions at random."


# ── 2. Fixed Agent ────────────────────────────────────────────────────────────

class FixedAgent:
    """
    Deterministic strategy: THINK exactly `think_steps` times then ANSWER.
    Ignores IMPROVE entirely.
    """

    def __init__(self, think_steps: int = 3):
        self.think_steps = think_steps
        self.name        = f"FixedAgent(n={think_steps})"

    def select_action(self, obs: dict) -> int:
        if obs["current_step"] < self.think_steps:
            return THINK
        return ANSWER

    def on_episode_end(self, total_reward: float):
        pass

    def summary(self) -> str:
        return f"FixedAgent — THINK {self.think_steps} times, then ANSWER."


# ── 3. Self-Improving Agent ──────────────────────────────────────────────────

class SelfImprovingAgent:
    """
    Adaptive agent with three components:

    Policy
    ------
    Maintains a confidence `threshold`. Steps:
        if confidence < threshold  →  THINK
        else                       →  ANSWER
    Occasionally fires IMPROVE_POLICY to update the threshold based on
    recent performance.

    Exploration
    -----------
    Epsilon-greedy: with probability `epsilon` choose a random action.
    Epsilon decays towards `epsilon_min` over episodes.

    Self-Improvement
    ----------------
    Every `improve_interval` episodes the agent evaluates its recent average
    reward and shifts the threshold:
        avg < performance_target  →  raise threshold (think more)
        avg ≥ performance_target  →  lower threshold (act faster)
    """

    name = "SelfImprovingAgent"

    def __init__(
        self,
        threshold:          float = 0.5,
        epsilon:            float = 0.3,
        epsilon_decay:      float = 0.99,
        epsilon_min:        float = 0.05,
        improve_interval:   int   = 10,
        performance_target: float = 4.0,
    ):
        self.threshold          = threshold
        self.epsilon            = epsilon
        self.epsilon_decay      = epsilon_decay
        self.epsilon_min        = epsilon_min
        self.improve_interval   = improve_interval
        self.performance_target = performance_target

        # internal bookkeeping
        self._episode_count     = 0
        self._recent_rewards    = []
        self._current_confidence = 0.0     # updated from obs each step

    # ── Called by the training loop on every step ─────────────────────────────

    def select_action(self, obs: dict, confidence: float = None) -> int:
        """
        confidence is NOT in obs (it's a hidden variable).
        The training loop may pass it explicitly so the agent can use it;
        a real deployment would estimate it indirectly.
        """
        # store for self-improvement decisions
        if confidence is not None:
            self._current_confidence = confidence

        # epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice([THINK, ANSWER, IMPROVE])

        # budget guard — answer if we're about to run out
        if obs["remaining_budget"] <= 1:
            return ANSWER

        # threshold policy
        if self._current_confidence < self.threshold:
            return THINK

        # small chance to fire IMPROVE even when ready to answer
        if random.random() < 0.05:
            return IMPROVE

        return ANSWER

    def on_episode_end(self, total_reward: float):
        """Called at episode end to update policy and decay epsilon."""
        self._episode_count  += 1
        self._recent_rewards.append(total_reward)

        # keep a rolling window of the last N episodes
        if len(self._recent_rewards) > self.improve_interval * 2:
            self._recent_rewards.pop(0)

        # decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # self-improve every `improve_interval` episodes
        if self._episode_count % self.improve_interval == 0:
            self._improve_policy()

    def _improve_policy(self):
        """Shift threshold based on recent performance."""
        if not self._recent_rewards:
            return
        avg = sum(self._recent_rewards) / len(self._recent_rewards)
        if avg < self.performance_target:
            # performing poorly → think more before answering
            self.threshold = min(0.95, self.threshold + 0.05)
        else:
            # performing well → be bolder, answer sooner
            self.threshold = max(0.2, self.threshold - 0.02)

    def summary(self) -> str:
        return (
            f"SelfImprovingAgent — threshold={self.threshold:.3f}, "
            f"epsilon={self.epsilon:.3f}, episodes={self._episode_count}"
        )


# ── 4. Q-Learning Agent ─────────────────────────────────────────────────────

class QLearningAgent:
    """
    Tabular Q-learning agent that discretizes the state space and
    learns Q-values for each (state, action) pair.

    State is discretized as (remaining_budget_bin, step_bin) where each
    dimension is bucketed into a small number of bins.
    """

    name = "QLearningAgent"

    def __init__(
        self,
        alpha:         float = 0.1,     # learning rate
        gamma:         float = 0.95,    # discount factor
        epsilon:       float = 0.3,     # exploration rate
        epsilon_decay: float = 0.995,
        epsilon_min:   float = 0.05,
        budget_bins:   int   = 5,
        step_bins:     int   = 5,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon        = epsilon
        self.epsilon_decay  = epsilon_decay
        self.epsilon_min    = epsilon_min
        self.budget_bins    = budget_bins
        self.step_bins      = step_bins

        # Q-table: {(budget_bin, step_bin): [q_think, q_answer, q_improve]}
        self.q_table: dict  = {}
        self._prev_state    = None
        self._prev_action   = None
        self._episode_count = 0

    def _discretize(self, obs: dict) -> tuple:
        budget = min(obs["remaining_budget"], 10)
        step   = min(obs["current_step"], 10)
        b_bin  = min(budget * self.budget_bins // 11, self.budget_bins - 1)
        s_bin  = min(step * self.step_bins // 11, self.step_bins - 1)
        return (b_bin, s_bin)

    def _get_q(self, state: tuple) -> list:
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        return self.q_table[state]

    def select_action(self, obs: dict, confidence: float = None) -> int:
        state = self._discretize(obs)

        # epsilon-greedy
        if random.random() < self.epsilon:
            action = random.choice([THINK, ANSWER, IMPROVE])
        else:
            q = self._get_q(state)
            action = q.index(max(q))

        # Q-update for previous step
        if self._prev_state is not None:
            self._update_q(self._prev_state, self._prev_action, 0.0, state, done=False)

        self._prev_state  = state
        self._prev_action = action
        return action

    def _update_q(self, state, action, reward, next_state, done):
        q = self._get_q(state)
        if done:
            target = reward
        else:
            next_q = self._get_q(next_state)
            target = reward + self.gamma * max(next_q)
        q[action] += self.alpha * (target - q[action])

    def on_episode_end(self, total_reward: float):
        # Terminal Q-update
        if self._prev_state is not None:
            self._update_q(self._prev_state, self._prev_action, total_reward,
                           self._prev_state, done=True)
        self._prev_state  = None
        self._prev_action = None
        self._episode_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def summary(self) -> str:
        return (
            f"QLearningAgent — alpha={self.alpha}, epsilon={self.epsilon:.3f}, "
            f"states={len(self.q_table)}, episodes={self._episode_count}"
        )