
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
        n_iters=1250,
        batch_eps=5,
        gamma=0.95,
        alpha0=0.2,
        alpha_decay=0.9994,
        alpha_min=0.005,
        sigma0=1.0,
        sigma_decay=0.9993,
        sigma_min=0.03,
        learn_sigma=True,
        seed=0
    ):

        # okolina i politika
        self.env = env if env is not None else Room(seed=seed)
        self.policy = policy if policy is not None else Policy(seed=seed)

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
            "w1": [], "w2": [], "b": [],
            "v1": [], "v2": [], "c": [],
            "sigma": [],
        }

    # Jedna epizoda (jedan dan)
    def play_day(self):
        """Odigra jedan dan i vrati listu koraka i sirovi zbir nagrada."""

        s = self.env.reset()
        x1, x2 = float(s[0]), float(s[1])

        steps, total = [], 0.0
        done = False

        while not done:

            a, u, m, sigma = self.policy.act(
                x1,
                x2,
                learn_sigma=self.learn_sigma,
                fixed_sigma=self.sigma
            )

            s_next, r, done, info = self.env.step(a)

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

            self.log["w1"].append(self.policy.w1)
            self.log["w2"].append(self.policy.w2)
            self.log["b"].append(self.policy.b)

            self.log["v1"].append(self.policy.v1)
            self.log["v2"].append(self.policy.v2)
            self.log["c"].append(self.policy.c)

            self.log["sigma"].append(avg_sigma)

            if verbose and (it % 25 == 0 or it == self.n_iters - 1):

                print(
                    f"iter {it:4d} | "
                    f"povrat {mean_ret:9.1f} | "
                    f"sigma {avg_sigma:.3f} | "
                    f"alpha {self.alpha:.3f} | "
                    f"w1 {self.policy.w1:+.3f} "
                    f"w2 {self.policy.w2:+.3f} "
                    f"b {self.policy.b:+.3f}"
                )

        return self.policy, self.log

    # Čuvanje naučenih podataka
    def save(self, path="results/trained.npz"):

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        self.policy.save(path)

        log_path = path.replace(".npz", "_log.npz")

        np.savez(
            log_path,
            ret=self.log["ret"],
            w1=self.log["w1"],
            w2=self.log["w2"],
            b=self.log["b"],
            v1=self.log["v1"],
            v2=self.log["v2"],
            c=self.log["c"],
            sigma=self.log["sigma"]
        )

        print(f"Model i istorija sačuvani: {path}")


# Ako se fajl pokreće direktno: istreniraj i sačuvaj
if __name__ == "__main__":

    for learn_sigma in [False, True]:

        name = "learned" if learn_sigma else "fixed"

        trainer = Trainer(
            learn_sigma=learn_sigma
        )

        trainer.train(verbose=True)
        trainer.save(f"results/{name}.npz")