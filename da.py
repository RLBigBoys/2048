import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "./learning_curves/training_log_sample.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("episode").reset_index(drop=True)
    return df


df = load_data()
print(df.head())

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].plot(df["episode"], df["reward"], label="reward", color="#c97b63")
axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
axes[0].set_ylabel("Reward")
axes[0].set_title("Reward over episodes")
axes[0].legend()

axes[1].plot(df["episode"], df["score"], label="score", color="#2f5d62")
axes[1].set_ylabel("Score")
axes[1].set_title("Score over episodes")
axes[1].legend()

axes[2].plot(df["episode"], df["max_tile"], label="max_tile", color="#264653")
axes[2].set_xlabel("Episode")
axes[2].set_ylabel("Max tile")
axes[2].set_title("Max tile over episodes")
axes[2].legend()

fig.tight_layout()
plt.show()