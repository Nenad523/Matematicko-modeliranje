"""
Numerička provjera gradijenta (centralne razlike).

Provjerava da ručni backprop iz policy.grad() daje isto što i numerički izvod
surogat-funkcije  L(theta) = (1/N) * sum_t G_t * log pi(a_t | s_t)
po SVIM parametrima, za više veličina skrivenog sloja H.

Pokretanje:  python check_gradient.py
"""

import numpy as np
from policy import Policy


def surrogate(policy, S, U, G):
    return float(np.mean(G * policy.log_prob_batch(S, U)))


def check(H, n=40, seed=1, eps=1e-6):
    rng = np.random.default_rng(seed)
    policy = Policy(seed=seed, H=H)

    # nasumične težine veće od početnih, da provjera ne bude "previše lagana"
    for p in policy.params():
        p += rng.normal(0.0, 0.3, size=p.shape)

    S = rng.normal(0.0, 0.7, size=(n, 4))
    _, _, mu, sigma = policy.forward(S)
    U = mu + sigma * rng.standard_normal((n, 2))
    G = rng.normal(0.0, 1.0, size=n)
    batch = [(S[i], U[i], mu[i], sigma[i], G[i]) for i in range(n)]

    analytic = policy.grad(batch)

    worst = 0.0
    for name, p, g in zip(["W1", "b1", "W2", "b2"], policy.params(), analytic):
        num = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = p[idx]
            p[idx] = old + eps
            up = surrogate(policy, S, U, G)
            p[idx] = old - eps
            down = surrogate(policy, S, U, G)
            p[idx] = old
            num[idx] = (up - down) / (2 * eps)

        rel = np.abs(num - g).max() / (np.abs(num).max() + 1e-12)
        worst = max(worst, rel)
        print(f"  H={H:2d}  {name}: najveća relativna greška = {rel:.2e}")
    return worst


if __name__ == "__main__":
    ok = True
    for H in (8, 16, 3):
        w = check(H)
        ok = ok and (w < 1e-5)
    print("\nGRADIJENT JE ISPRAVAN" if ok else "\nGREŠKA U GRADIJENTU")