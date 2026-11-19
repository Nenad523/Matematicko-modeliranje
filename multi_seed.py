"""
Više seedova: pokazuje da rezultat nije slučajnost jednog pokretanja.

Trenira agenta za svaki seed, mjeri povrat na istih 30 novih dana i poredi
ga sa on/off termostatom.

Pokretanje:  python multi_seed.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import Trainer
from plots import Evaluator

SEEDS = [0, 1, 2, 3, 4]
H = 8
RESULTS = "results"


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)

    povrati_agent = []
    for seed in SEEDS:
        trainer = Trainer(seed=seed, H=H)
        policy, log = trainer.train(verbose=False)

        ev = Evaluator(policy, log)
        r = ev.evaluate_returns("policy", policy, n=30)
        povrati_agent.append(r.mean())
        print(f"seed {seed}: agent {r.mean():9.1f} ± {r.std():6.1f}")

    # termostat ne zavisi od seeda treninga - isti dani, isti rezultat
    ev = Evaluator(policy, log)
    r_term = ev.evaluate_returns("thermostat", n=30)

    agent = np.array(povrati_agent)
    print("\n========== ZAVRŠNI REZULTATI ==========")
    print(f"agent     | prosjek {agent.mean():9.1f} | std {agent.std():7.1f} "
          f"| najgori {agent.min():9.1f} | najbolji {agent.max():9.1f}")
    print(f"termostat | prosjek {r_term.mean():9.1f} | std {r_term.std():7.1f}")

    np.savez(os.path.join(RESULTS, "multi_seed_results.npz"),
             agent=agent, termostat=r_term, seeds=np.array(SEEDS))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axhline(r_term.mean(), color="#d62728", ls="--",
               label=f"termostat ({r_term.mean():.0f})")
    ax.bar(range(len(SEEDS)), agent, color="#1f77b4", alpha=0.75, label="agent")
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("prosječan povrat na 30 novih dana")
    ax.set_title(f"REINFORCE agent kroz {len(SEEDS)} seedova (H={H})")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "multi_seed_comparison.png"), dpi=140)
    plt.close()
    print(f"\nGrafikon sačuvan: {RESULTS}/multi_seed_comparison.png")