import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

CSV_PATH = Path("learning_curves/value_iteration_learning_curve.csv")

df = pd.read_csv(CSV_PATH)

# Экспоненциальное сглаживание reward и score
alpha = 0.01  # сглаживание: чем меньше, тем более гладкая кривая

df["reward_ema"] = df["reward"].ewm(alpha=alpha, adjust=False).mean()
df["score_ema"] = df["score"].ewm(alpha=alpha, adjust=False).mean()

print(df[["episode", "reward", "reward_ema", "score", "score_ema"]].head())

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# График для reward
axes[0].plot(df["episode"], df["reward"], label="reward (raw)", color="#f4a261", alpha=0.2, linewidth=0.5)
axes[0].plot(df["episode"], 1.075 ** df["reward_ema"], label=f"reward EMA (α={alpha})", color="#e76f51", linewidth=2)
axes[0].set_ylabel("reward")
axes[0].set_title("Episode reward with exponential smoothing and exp. scaling")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# График для score
axes[1].plot(df["episode"], df["score"], label="score (raw)", color="#2a9d8f", alpha=0.2, linewidth=0.5)
axes[1].plot(df["episode"], 1.009 ** df["score_ema"], label=f"score EMA (α={alpha})", color="#264653", linewidth=2)
axes[1].set_xlabel("episode")
axes[1].set_ylabel("score")
axes[1].set_title("Episode score with exponential smoothing and exp. scaling")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
plt.show()