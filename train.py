"""
Povezuje svijet (environment.Room) i mozak (policy.Policy) u petlju učenja.

Trainer (REINFORCE sa Gausovom politikom, σ se uči kao izlaz mreže):
  - odigrava dane (epizode) trenutnom politikom,
  - računa povrate G_t unazad kroz svaku epizodu,
  - grupiše više epizoda u batch i normalizuje povrate preko cijelog batcha,
  - ažurira težine gradijentnim usponom,
  - bilježi sve što treba za izvještaj.

Pokretanje:  python train.py
"""

import os
import time
import numpy as np

from environment import Room
from policy import Policy


def returns_to_go(steps, gamma):
    """G_t unazad, zasebno za jednu epizodu. steps = lista (s, u, m, sigma, r)."""
    out, G = [], 0.0
    for (s, u, m, sigma, r) in reversed(steps):
        G = r + gamma * G
        out.append((s, u, m, sigma, G))
    out.reverse()
    return out


class Trainer:

    def __init__(
        self,
        env=None,
        policy=None,
        n_iters=1500,
        batch_eps=7,
        gamma=0.95,
        alpha0=0.2,
        alpha_decay=0.9995,
        alpha_min=0.02,
        clip_norm=5.0,
        H=8,
        seed=0,
    ):
        self.env = env if env is not None else Room(seed=seed)
        self.policy = policy if policy is not None else Policy(seed=seed, H=H)

        self.n_iters = n_iters
        self.batch_eps = batch_eps
        self.gamma = gamma
        self.alpha = alpha0
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.clip_norm = clip_norm

        self.log = {k: [] for k in
                    ["ret", "sigma", "alpha", "W1", "b1", "W2", "b2"]}

    # Jedna epizoda (jedan dan)
    def play_day(self):
        s = self.env.reset()
        steps, total, done = [], 0.0, False
        while not done:
            u, m, sigma = self.policy.act(s)
            s_next, r, done, _ = self.env.step(u)     # okolina sama odsijeca akciju
            steps.append((s, u, m, sigma, r))
            total += r
            s = s_next
        return steps, total

    # Jedna iteracija treninga (batch od batch_eps epizoda)
    def train_iteration(self):
        batch, rets = [], []
        for _ in range(self.batch_eps):
            steps, total = self.play_day()
            batch.extend(returns_to_go(steps, self.gamma))
            rets.append(total)

        # normalizacija povrata preko CIJELOG batcha
        Gs = np.array([t[-1] for t in batch], dtype=np.float64)
        Gs = (Gs - Gs.mean()) / (Gs.std() + 1e-8)
        batch = [(s, u, m, sg, g) for (s, u, m, sg, _), g in zip(batch, Gs)]

        self.policy.update(batch, self.alpha, clip_norm=self.clip_norm)

        avg_sigma = np.mean([t[3] for t in batch], axis=0)      # (2,)
        self.alpha = max(self.alpha * self.alpha_decay ** self.batch_eps,
                         self.alpha_min)
        return float(np.mean(rets)), avg_sigma

    def train(self, verbose=True):
        t0 = time.time()
        for it in range(self.n_iters):
            mean_ret, avg_sigma = self.train_iteration()

            self.log["ret"].append(mean_ret)
            self.log["sigma"].append(avg_sigma)
            self.log["alpha"].append(self.alpha)
            p = self.policy
            self.log["W1"].append(p.W1.copy())      # kopije, ne reference
            self.log["b1"].append(p.b1.copy())
            self.log["W2"].append(p.W2.copy())
            self.log["b2"].append(p.b2.copy())

            if verbose and (it % 25 == 0 or it == self.n_iters - 1):
                print(f"iter {it:4d} | povrat {mean_ret:9.1f} | "
                      f"sigma {avg_sigma[0]:.3f} {avg_sigma[1]:.3f} | "
                      f"alpha {self.alpha:.4f} | {time.time() - t0:5.0f}s")
        return self.policy, self.log

    def save(self, path="results/trained.npz"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.policy.save(path)
        np.savez(path.replace(".npz", "_log.npz"),
                 **{k: np.array(v) for k, v in self.log.items()})
        print(f"Model i istorija sačuvani: {path}")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(verbose=True)
    trainer.save("results/trained.npz")