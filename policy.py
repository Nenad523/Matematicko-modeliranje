"""
Politika: neuronska mreža 4 – H – 4 sa ručnim backpropagation-om (samo NumPy).

Ulaz:   s = [x1, x2, x3, x4]  (skalirano stanje iz environment.Room)
Skriveni sloj: H neurona, tanh
Izlazi (z = h @ W2 + b2, 4 broja):
    z[0] -> mu1    = tanh(z0)                      klima        a1 u [-1, 1]
    z[1] -> sigma1 = SIGMA_MIN + softplus(z1)      šum klime
    z[2] -> mu2    = sigmoid(z2)                   ventilacija  a2 u [0, 1]
    z[3] -> sigma2 = SIGMA_MIN + softplus(z3)      šum ventilacije

Akcije: a_i ~ N(mu_i, sigma_i^2). Okolina odsijeca akcije na dozvoljeni opseg,
a u log-gustini i u batch-u koristi se NEODSJEČENA vrijednost.

Broj parametara: 9*H + 4  (za H = 8 to je 76).

Ugovor prema ostalim modulima:
    Policy(seed, H)                      konstruktor
    act(s)           -> u, m, sigma      sve su vektori dužine 2
    mu(s)            -> m                deterministička akcija (za evaluaciju)
    update(batch, alpha)                 batch = lista (s, u, m, sigma, G)
    params() / save(path) / load(path)
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softplus(z):
    # numerički stabilan ln(1 + e^z)
    return np.logaddexp(0.0, z)


class Policy:

    SIGMA_MIN = 0.05
    SIGMA_INIT = 0.30
    N_IN = 4
    N_OUT = 4          # mu1, sigma1, mu2, sigma2

    def __init__(self, seed=0, H=8):
        self.rng = np.random.default_rng(seed)
        self.H = int(H)

        # skriveni sloj
        self.W1 = self.rng.normal(0.0, 0.5, size=(self.N_IN, self.H))
        self.b1 = np.zeros(self.H)

        # izlazni sloj: male težine, pa mreža kreće s mu1 ~ 0 i mu2 ~ 0.5
        self.W2 = self.rng.normal(0.0, 0.1, size=(self.H, self.N_OUT))
        self.b2 = np.zeros(self.N_OUT)

        # bias za sigma tako da početni sigma bude oko SIGMA_INIT
        b_sigma = np.log(np.expm1(self.SIGMA_INIT - self.SIGMA_MIN))
        self.b2[1] = b_sigma
        self.b2[3] = b_sigma

    # ------------------------------------------------------------------
    # Prolaz unaprijed (radi i za jedno stanje (4,) i za batch (n, 4))
    # ------------------------------------------------------------------
    def forward(self, S):
        S = np.atleast_2d(np.asarray(S, dtype=np.float64))
        h = np.tanh(S @ self.W1 + self.b1)            # (n, H)
        z = h @ self.W2 + self.b2                     # (n, 4)

        mu = np.stack([np.tanh(z[:, 0]), sigmoid(z[:, 2])], axis=1)             # (n, 2)
        sigma = np.stack([self.SIGMA_MIN + softplus(z[:, 1]),
                          self.SIGMA_MIN + softplus(z[:, 3])], axis=1)          # (n, 2)
        return h, z, mu, sigma

    def mu(self, s):
        """Deterministička akcija: srednje vrijednosti (vektor dužine 2)."""
        return self.forward(s)[2][0]

    def act(self, s):
        """Uzorkuje akciju. Vraća (u, m, sigma), svaki vektor dužine 2."""
        _, _, mu, sigma = self.forward(s)
        m, sg = mu[0], sigma[0]
        eps = self.rng.standard_normal(2)
        u = m + sg * eps
        return u, m, sg

    # ------------------------------------------------------------------
    # Log-gustina (zbir po dvije nezavisne Gausove akcije)
    # ------------------------------------------------------------------
    def log_prob_batch(self, S, U):
        _, _, mu, sigma = self.forward(S)
        U = np.atleast_2d(U)
        lp = (-(U - mu) ** 2 / (2.0 * sigma ** 2)
              - np.log(sigma) - 0.5 * np.log(2.0 * np.pi))
        return lp.sum(axis=1)                          # (n,)

    def log_prob(self, s, u):
        return float(self.log_prob_batch(s, u)[0])

    # ------------------------------------------------------------------
    # Gradijent funkcije cilja: (1/N) * sum_t G_t * grad log pi(a_t | s_t)
    # ------------------------------------------------------------------
    def grad(self, batch):
        n = len(batch)
        if n == 0:
            return [np.zeros_like(p) for p in self.params()]

        S = np.array([b[0] for b in batch], dtype=np.float64)     # (n, 4)
        U = np.array([b[1] for b in batch], dtype=np.float64)     # (n, 2)
        G = np.array([b[4] for b in batch], dtype=np.float64)     # (n,)

        h, z, mu, sigma = self.forward(S)
        mu1, mu2 = mu[:, 0], mu[:, 1]
        s1, s2 = sigma[:, 0], sigma[:, 1]
        d1, d2 = U[:, 0] - mu1, U[:, 1] - mu2

        # greške izlaznog sloja (derivacija log pi po z), po jedna kolona za izlaz
        Delta = np.empty((n, 4))
        Delta[:, 0] = d1 / s1 ** 2 * (1.0 - mu1 ** 2)                       # tanh
        Delta[:, 1] = (d1 ** 2 - s1 ** 2) / s1 ** 3 * sigmoid(z[:, 1])      # softplus
        Delta[:, 2] = d2 / s2 ** 2 * mu2 * (1.0 - mu2)                      # sigmoid
        Delta[:, 3] = (d2 ** 2 - s2 ** 2) / s2 ** 3 * sigmoid(z[:, 3])      # softplus

        D2 = Delta * G[:, None]                         # delta^(2) = Delta * G_t
        D1 = (D2 @ self.W2.T) * (1.0 - h ** 2)          # delta^(1)

        gW2 = h.T @ D2 / n
        gb2 = D2.sum(axis=0) / n
        gW1 = S.T @ D1 / n
        gb1 = D1.sum(axis=0) / n
        return [gW1, gb1, gW2, gb2]

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def update(self, batch, alpha, clip_norm=None):
        """Gradijentni uspon: theta <- theta + alpha * grad J."""
        grads = self.grad(batch)
        if clip_norm is not None:
            total = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
            if total > clip_norm:
                grads = [g * (clip_norm / total) for g in grads]
        for p, g in zip(self.params(), grads):
            p += alpha * g

    # ------------------------------------------------------------------
    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, H=self.H)

    def load(self, path):
        d = np.load(path)
        self.W1, self.b1 = d["W1"], d["b1"]
        self.W2, self.b2 = d["W2"], d["b2"]
        self.H = int(d["H"])
        return self


if __name__ == "__main__":
    p = Policy(seed=0, H=8)
    s = np.array([0.1, -0.3, 0.5, 0.2])
    u, m, sg = p.act(s)
    print("mu   :", m)
    print("sigma:", sg)
    print("u    :", u)
    print("log pi:", p.log_prob(s, u))
    print("broj parametara:", sum(x.size for x in p.params()))