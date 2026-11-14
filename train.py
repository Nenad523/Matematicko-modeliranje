
"""
Povezuje svijet (environment.Room) i mozak (policy.Policy) u petlju učenja.
Sadrži klasu Trainer koja:
  - odigrava dane (epizode) agentom,
  - računa povrate G_t unazad kroz epizodu,
  - grupiše više epizoda (batch) radi manje varijanse,
  - normalizuje povrate,
  - ažurira težine mreže (gradijentni uspon),
  - omogućava ručno podešavanje ili učenje sigme,
  - bilježi (logging) sve što treba za izvještaj.

Pokretanje:  python train.py
"""

import os
import numpy as np

from environment import Room
from policy import Policy


# Pomoćne funkcije za povrate
def returns_to_go(steps, gamma):
    """G_t unazad, zasebno za jednu epizodu."""
    out, G = [], 0.0

    for (x1, x2, u, m, sigma, r) in reversed(steps):
        G = r + gamma * G
        out.append((x1, x2, u, m, sigma, G))

    out.reverse()
    return out


# =====================================================================
# TRENER
# =====================================================================
class Trainer:
    """
    Petlja učenja po algoritmu REINFORCE sa Gausovom politikom.

    learn_sigma=True:
        Mreža uči standardnu devijaciju sigma.

    learn_sigma=False:
        Sigma je ručno zadana i smanjuje se po rasporedu.
    """

    def __init__(
        self,
        env=None,
        policy=None,
        n_iters=500,
        batch_eps=7,
        gamma=0.95,
        alpha0=0.2,
        alpha_decay=0.9995,
        alpha_min=0.01,
        sigma0=1.0,
        sigma_decay=0.9993,
        sigma_min=0.03,
        learn_sigma=True,
        seed=0,
        N=8
    ):

        # okolina i politika
        self.env = env if env is not None else Room(seed=seed)
        self.policy = policy if policy is not None else Policy(seed=seed, N=N)

        self.n_iters = n_iters
        self.batch_eps = batch_eps
        self.gamma = gamma
        self.alpha = alpha0
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.sigma = sigma0
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min

        self.learn_sigma = learn_sigma

        # logging
        self.log = {
            "ret": [],
            "A": [],      # lista matrica (N,2) — jedna po iteraciji
            "c": [],      # lista vektora (N,)
            "w": [],      # lista vektora (N,)
            "b": [],      # lista skalara
            "v": [],      # lista vektora (N,)
            "d": [],      # lista skalara
            "sigma": [],
            "alpha": [],  # dodaj ako želiš da vidiš α na grafikonu
        }

    # Jedna epizoda (jedan dan)
    def play_day(self):
        """Odigra jedan dan i vrati listu koraka i sirovi zbir nagrada."""

        s = self.env.reset()
        x1, x2 = float(s[0]), float(s[1])

        steps, total = [], 0.0
        done = False

        while not done:

            u, m, sigma = self.policy.act(
                x1,
                x2,
                learn_sigma=self.learn_sigma,
                fixed_sigma=self.sigma
            )

            s_next, r, done, info = self.env.step(u)

            steps.append((x1, x2, u, m, sigma, r))

            total += r

            x1, x2 = float(s_next[0]), float(s_next[1])

        return steps, total

    # Jedna iteracija treninga (batch od batch_eps epizoda)
    def train_iteration(self):

        batch, rets = [], []

        for _ in range(self.batch_eps):

            steps, total = self.play_day()

            batch.extend(returns_to_go(steps, self.gamma))
            rets.append(total)

        # Normalizacija povrata preko CIJELOG batch-a
        Gs = np.array([t[-1] for t in batch], dtype=np.float64)

        Gs = (Gs - Gs.mean()) / (Gs.std() + 1e-8)

        batch = [
            (x1, x2, u, m, sigma, g)
            for (x1, x2, u, m, sigma, _), g in zip(batch, Gs)
        ]

        # Ažuriranje parametara politike
        self.policy.update(
            batch,
            self.alpha,
            learn_sigma=self.learn_sigma
        )

        # Sigma opada samo kada se ručno podešava
        if not self.learn_sigma:

            self.sigma = max(
                self.sigma * self.sigma_decay ** self.batch_eps,
                self.sigma_min
            )

            avg_sigma = self.sigma

        else:

            avg_sigma = np.mean([t[4] for t in batch])


        # Alpha opada
        self.alpha = max(self.alpha * self.alpha_decay ** self.batch_eps, self.alpha_min)

        return float(np.mean(rets)), float(avg_sigma)

    # -----------------------------------------------------------------
    # Glavna petlja
    # -----------------------------------------------------------------
    def train(self, verbose=True):

        for it in range(self.n_iters):

            mean_ret, avg_sigma = self.train_iteration()

            self.log["ret"].append(mean_ret)

            # kopije! (da se ne mijenjaju kroz reference)
            self.log["A"].append(self.policy.A.copy())
            self.log["c"].append(self.policy.c.copy())
            self.log["w"].append(self.policy.w.copy())
            self.log["b"].append(self.policy.b)

            self.log["v"].append(self.policy.v.copy())
            self.log["d"].append(self.policy.d)

            self.log["sigma"].append(avg_sigma)
            self.log["alpha"].append(self.alpha)   # ako si dodao

            if verbose and (it % 25 == 0 or it == self.n_iters - 1):

                print(
                    f"iter {it:4d} | "
                    f"povrat {mean_ret:9.1f} | "
                    f"sigma {avg_sigma:.3f} | "
                    f"alpha {self.alpha:.3f} | "
                )

        return self.policy, self.log

    # Čuvanje naučenih podataka
    def save(self, path="results/trained.npz"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        self.policy.save(path)

        log_path = path.replace(".npz", "_log.npz")

        np.savez(
            log_path,
            ret=np.array(self.log["ret"]),
            A=np.array(self.log["A"]),     # shape (n_iters, N, 2)
            c=np.array(self.log["c"]),     # shape (n_iters, N)
            w=np.array(self.log["w"]),     # shape (n_iters, N)
            b=np.array(self.log["b"]),     # shape (n_iters,)
            v=np.array(self.log["v"]),     # shape (n_iters, N)
            d=np.array(self.log["d"]),     # shape (n_iters,)
            sigma=np.array(self.log["sigma"]),
            alpha=np.array(self.log["alpha"]) if "alpha" in self.log else np.array([]),
        )

        print(f"Model i istorija sačuvani: {path}")


# Ako se fajl pokreće direktno: istreniraj i sačuvaj
if __name__ == "__main__":

    for learn_sigma in [True]:

        name = "learned" if learn_sigma else "fixed"

        trainer = Trainer(
            learn_sigma=learn_sigma
        )

        trainer.train(verbose=True)
        trainer.save(f"results/{name}.npz")