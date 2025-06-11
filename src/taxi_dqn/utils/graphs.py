import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# reduce reusage of the same logic
def ensure_parent(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def plot_training(rewards: list[float],
                  avg100: list[float],
                  epsilons: list[float],
                  losses: list[float | None],
                  successes: list[int],
                  save_path: str = "training_summary.png"):

    ensure_parent(save_path)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    x = np.arange(1, len(rewards) + 1)

    # rewards curve
    axs[0, 0].plot(x, rewards, label="Reward / episode")
    axs[0, 0].plot(x, avg100, label="Rolling-100 mean", linewidth=2)
    axs[0, 0].set_title("Episode reward")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].legend()

    # epsilon decay over time
    axs[0, 1].plot(x, epsilons)
    axs[0, 1].set_title("Epsilon decay")
    axs[0, 1].set_xlabel("Episode")

    # average loss over time
    axs[1, 0].plot(x, losses, color="tab:orange")
    axs[1, 0].set_title("Avg loss / episode")
    axs[1, 0].set_xlabel("Episode")

    # success rate (rolling-100)  over time
    roll = np.convolve(successes, np.ones(100)/100, mode="same")
    axs[1, 1].plot(x, roll, color="tab:green")
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].set_title("Success rate (rolling-100)")
    axs[1, 1].set_xlabel("Episode")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print("Graphs saved to", save_path)

def plot_test_hist(rewards: list[float], save_path: str = "test_reward_hist.png"):
    ensure_parent(save_path)
    plt.figure(figsize=(6,4))
    plt.hist(rewards, bins=15, edgecolor="black")
    plt.xlabel("Episode reward")
    plt.ylabel("Count")
    plt.title("Reward distribution on test set")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("Histogram saved to", save_path)
