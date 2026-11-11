
import numpy as np
import matplotlib.pyplot as plt

from train import Trainer
from plots import Evaluator

seeds = [0, 1, 2, 3, 4]

results = {
    "fixed": [],
    "learned": []
}

for seed in seeds:
    print(f"\nTrening seed: {seed}")

    for learn_sigma in [False, True]:

        name = "learned" if learn_sigma else "fixed"

        trainer = Trainer(
            n_iters=600,
            batch_eps=5,
            seed=seed,
            sigma0=0.525,
            learn_sigma=learn_sigma
        )

        policy, log = trainer.train(verbose=False)

        evaluator = Evaluator(policy, log)

        returns = evaluator.evaluate_returns(
            "policy", policy, n=30
        )

        results[name].append(np.mean(returns))

        print(
            f"{name:8s}: "
            f"{np.mean(returns):.2f}"
        )

# Završno poređenje
print("\n========== ZAVRŠNI REZULTATI ==========")

for name in ["fixed", "learned"]:
    values = np.array(results[name])

    print(
        f"{name:8s} | "
        f"prosek: {values.mean():.2f} | "
        f"std: {values.std():.2f}"
    )

np.savez("results/multi_seed_results.npz",
         fixed=results["fixed"],
         learned=results["learned"])

print("\nRezultati sačuvani.")


# Grafikon poređenja pet seedova
fixed_values = np.array(results["fixed"])
learned_values = np.array(results["learned"])

fig, ax = plt.subplots(figsize=(9, 6))

data = [fixed_values, learned_values]

ax.boxplot(
    data,
    tick_labels=["Sigma (rucno)", "Naučena sigma"],
    showmeans=True
)

# Prikaz rezultata svakog pojedinačnog seed-a
for i, values in enumerate(data, start=1):
    offsets = np.linspace(-0.08, 0.08, len(values))

    ax.scatter(
        i + offsets,
        values,
        s=65,
        label="Pojedinačni seedovi" if i == 1 else None,
        zorder=3
    )

ax.set_title("Poređenje REINFORCE agenata – 5 seedova")
ax.set_ylabel("Prosečan povrat na 30 testnih dana")
ax.grid(axis="y", alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("results/multi_seed_comparison.png", dpi=150)
plt.close()

print("Grafikon sačuvan: results/multi_seed_comparison.png")

