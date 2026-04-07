"""
train.py — Training loop + visualization for the Token-Economist RL project
=============================================================================
Trains three agents and compares their performance at learning to budget
reasoning tokens efficiently.

Run:  python train.py
"""

import random
import math
import statistics

# ── Local imports ─────────────────────────────────────────────────────────────
from environment import TokenEconomistEnv, THINK, ANSWER, IMPROVE, ACTION_NAMES
from agents import RandomAgent, FixedAgent, SelfImprovingAgent, QLearningAgent


# ── Training loop ─────────────────────────────────────────────────────────────

def run_training(agent, n_episodes: int = 300, budget: int = 10, verbose_every: int = 0):
    """
    Train (or evaluate) an agent over n_episodes.

    Returns a dict with per-episode statistics for plotting.
    """
    env = TokenEconomistEnv(budget=budget, verbose=False)

    rewards      = []
    accuracies   = []
    think_counts = []
    thresholds   = []
    efficiencies = []   # correct answers per thought used

    correct_total  = 0
    answered_total = 0

    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        ep_reward  = 0
        ep_thinks  = 0
        ep_log     = []

        while not done:
            # SelfImprovingAgent and QLearningAgent get to see confidence
            if isinstance(agent, (SelfImprovingAgent, QLearningAgent)):
                action = agent.select_action(obs, confidence=env._confidence)
            else:
                action = agent.select_action(obs)

            obs, reward, done, info = env.step(action)
            ep_reward += reward

            if info.get("action") == "THINK":
                ep_thinks += 1
            ep_log.append(info)

        # update agent at episode end
        agent.on_episode_end(ep_reward)

        # track accuracy
        final = ep_log[-1]
        was_correct = False
        if final.get("action") == "ANSWER":
            answered_total += 1
            if final.get("correct"):
                correct_total += 1
                was_correct = True

        rewards.append(ep_reward)
        think_counts.append(ep_thinks)
        acc = correct_total / answered_total if answered_total else 0.0
        accuracies.append(acc)

        # efficiency: if correct, how many thoughts? (lower is better)
        if was_correct and ep_thinks > 0:
            efficiencies.append(1.0 / ep_thinks)
        elif was_correct:
            efficiencies.append(1.0)  # perfect efficiency — correct with 0 thinks
        else:
            efficiencies.append(0.0)

        if isinstance(agent, SelfImprovingAgent):
            thresholds.append(agent.threshold)
        else:
            thresholds.append(None)

        # optional verbose logging
        if verbose_every and (ep + 1) % verbose_every == 0:
            window = rewards[max(0, ep - 19):ep + 1]
            print(
                f"  Ep {ep+1:4d} | "
                f"reward={ep_reward:+5.1f} | "
                f"avg(last20)={sum(window)/len(window):+5.2f} | "
                f"thinks={ep_thinks} | "
                f"acc={acc:.2%} | "
                f"{agent.summary()}"
            )

    return {
        "rewards":      rewards,
        "accuracies":   accuracies,
        "think_counts": think_counts,
        "thresholds":   thresholds,
        "efficiencies": efficiencies,
    }


# ── Smoothing helper ──────────────────────────────────────────────────────────

def smooth(values, window=20):
    out = []
    for i, v in enumerate(values):
        start = max(0, i - window + 1)
        out.append(sum(values[start:i + 1]) / (i - start + 1))
    return out


# ── ASCII plot (no dependencies) ──────────────────────────────────────────────

def ascii_plot(series: list, title: str, width: int = 60, height: int = 12):
    lo, hi = min(series), max(series)
    rng    = hi - lo if hi != lo else 1
    print(f"\n  {title}")
    print(f"  {'─' * width}")
    for row in range(height - 1, -1, -1):
        threshold_val = lo + (row / (height - 1)) * rng
        line = ""
        for i, v in enumerate(series):
            x = int(i * width / len(series))
            if x >= len(line):
                line += " " * (x - len(line))
                char = "█" if v >= threshold_val else " "
                line += char
        print(f"  {threshold_val:+6.1f} │{line}")
    print(f"         └{'─' * width}")
    print(f"          0{' ' * (width // 2 - 2)}episode{' ' * (width // 2 - 4)}{len(series)}")


# ── Optional matplotlib plots ─────────────────────────────────────────────────

def plot_results(results_by_agent: dict, smooth_window: int = 20):
    """
    Plot reward, accuracy, and efficiency curves for all agents.
    Requires matplotlib — skip gracefully if not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for Docker/CI
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("\n[INFO] matplotlib not installed — skipping plots.")
        print("       pip install matplotlib   to enable.")
        return

    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
    fig.suptitle("Token-Economist RL — Agent Comparison\n"
                 "\"How many reasoning tokens should you spend?\"",
                 fontsize=13, fontweight="bold")

    colors = {
        "RandomAgent":        "#e74c3c",
        "FixedAgent":         "#3498db",
        "SelfImprovingAgent": "#2ecc71",
        "QLearningAgent":     "#9b59b6",
    }

    for name, stats in results_by_agent.items():
        color = colors.get(name.split("(")[0], "#333")
        eps   = list(range(1, len(stats["rewards"]) + 1))

        # Reward curve
        axes[0].plot(eps, smooth(stats["rewards"], smooth_window),
                     label=name, color=color, linewidth=1.5)

        # Accuracy curve
        axes[1].plot(eps, [a * 100 for a in smooth(stats["accuracies"], smooth_window)],
                     color=color, linewidth=1.5)

        # Efficiency curve (thoughts per correct answer)
        axes[2].plot(eps, smooth(stats["think_counts"], smooth_window),
                     color=color, linewidth=1.5)

    axes[0].set_ylabel("Episode Reward (smoothed)")
    axes[0].legend(fontsize=9)
    axes[0].axhline(0, color="#ccc", linewidth=0.5)
    axes[0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%+.0f"))

    axes[1].set_ylabel("Accuracy % (smoothed)")
    axes[1].set_ylim(0, 105)
    axes[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

    axes[2].set_ylabel("Avg Thoughts Used (smoothed)")

    # ── Threshold evolution plot (SelfImprovingAgent) ─────────────────────
    if "SelfImprovingAgent" in results_by_agent:
        thresholds = results_by_agent["SelfImprovingAgent"]["thresholds"]
        valid = [t for t in thresholds if t is not None]
        if valid:
            eps_t = list(range(1, len(valid) + 1))
            axes[3].plot(eps_t, valid, color="#2ecc71", linewidth=1.5,
                         label="SelfImprovingAgent threshold")
            axes[3].set_ylabel("Confidence Threshold")
            axes[3].set_ylim(0.1, 1.0)
            axes[3].axhline(0.5, color="#ccc", linewidth=0.5, linestyle="--",
                            label="Initial threshold")
            axes[3].legend(fontsize=9)
    axes[3].set_xlabel("Episode")

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\n[INFO] Plot saved → results.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    N_EPISODES = 300
    BUDGET     = 10

    print("=" * 65)
    print("  Token-Economist RL — Self-Regulating Reasoning Budget")
    print("  \"Teaching agents when to think and when to just answer.\"")
    print("=" * 65)

    agents = {
        "RandomAgent":          RandomAgent(),
        "FixedAgent(n=3)":      FixedAgent(think_steps=3),
        "SelfImprovingAgent":   SelfImprovingAgent(threshold=0.5, epsilon=0.3),
        "QLearningAgent":       QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.3),
    }

    results = {}
    for name, agent in agents.items():
        print(f"\n▶  Training {name} for {N_EPISODES} episodes …")
        stats = run_training(agent, n_episodes=N_EPISODES, budget=BUDGET, verbose_every=50)
        results[name] = stats

        last_n = 50
        avg_r  = sum(stats["rewards"][-last_n:]) / last_n
        avg_a  = sum(stats["accuracies"][-last_n:]) / last_n
        avg_t  = sum(stats["think_counts"][-last_n:]) / last_n
        print(f"   Last {last_n} eps  →  avg reward={avg_r:+.2f} | "
              f"acc={avg_a:.2%} | avg thinks={avg_t:.1f}")
        print(f"   Final policy: {agent.summary()}")

        # quick ASCII reward curve
        sampled = stats["rewards"][::max(1, N_EPISODES // 60)]
        ascii_plot(smooth(sampled, 10), f"Reward curve — {name}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Summary comparison (last 50 episodes)")
    print("=" * 65)
    print(f"  {'Agent':<26} {'Avg Reward':>12} {'Accuracy':>10} {'Avg Thinks':>12}")
    print(f"  {'─' * 62}")
    for name, stats in results.items():
        last_n = 50
        avg_r  = sum(stats["rewards"][-last_n:])      / last_n
        avg_a  = sum(stats["accuracies"][-last_n:])    / last_n
        avg_t  = sum(stats["think_counts"][-last_n:])  / last_n
        print(f"  {name:<26} {avg_r:>+12.2f} {avg_a:>9.1%} {avg_t:>12.1f}")

    plot_results(results)
    print("\n✅ Training complete. Check results.png for visualizations.")


if __name__ == "__main__":
    main()