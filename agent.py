import os

"""
test.py: cijeli projekat u jednom fajlu (za učenje i brzo isprobavanje).
Sadrži:
  1. Policy         (Član 2) jednoslojna Gausova politika, ručni gradijent
  2. RoomEnv        (Član 1) simulirana soba + obični on/off termostat
  3. Treniranje     (Član 3) REINFORCE, batch od 5 epizoda, povrat G_t
  4. Evaluacija     demo ispis u terminalu i grafikoni

Pokretanje:  python test.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                 # crta u fajl, bez prozora
import matplotlib.pyplot as plt


# =====================================================================
# 1. POLITIKA (mozak agenta)
# =====================================================================
class Policy:

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.w1 = float(self.rng.normal(0, 0.1))
        self.w2 = float(self.rng.normal(0, 0.1))
        self.b = 0.0

    def mu(self, x1, x2):
        """Forward pass: srednja vrijednost akcije u [-1, 1]."""
        z = self.w1 * x1 + self.w2 * x2 + self.b
        return np.tanh(z)

    def act(self, x1, x2, sigma):
        """Izvuče akciju a ~ N(mu, sigma^2). Vraća (a, mu). a je NEKLIPOVANA."""
        m = self.mu(x1, x2)
        a = m + sigma * self.rng.standard_normal()
        return a, m

    def act_deterministic(self, x1, x2):
        """Bez šuma: a = mu (za demo i evaluaciju)."""
        return self.mu(x1, x2)

    def grad(self, batch, sigma):
        """
        Procjena gradijenta J po (w1, w2, b), prosjek preko N koraka:
            (1/N) * sum  G_t * dlogpi/dtheta
        dlogpi/dw1 = (a - mu)/sigma^2 * (1 - mu^2) * x1  (isto za w2, b)
        """
        g1 = g2 = gb = 0.0
        for (x1, x2, a, m, G) in batch:
            delta = (a - m) / sigma**2 * (1 - m**2) * G
            g1 += delta * x1
            g2 += delta * x2
            gb += delta
        N = len(batch)
        return g1 / N, g2 / N, gb / N

    def update(self, batch, sigma, alpha):
        """Gradijentni uspon: theta <- theta + alpha * grad J."""
        g1, g2, gb = self.grad(batch, sigma)
        self.w1 += alpha * g1
        self.w2 += alpha * g2
        self.b += alpha * gb

    def save(self, path):
        np.savez(path, w1=self.w1, w2=self.w2, b=self.b)

    def load(self, path):
        data = np.load(path)
        self.w1 = float(data["w1"])
        self.w2 = float(data["w2"])
        self.b = float(data["b"])


# =====================================================================
# 2. OKOLINA (soba) i baseline termostat
# =====================================================================
T_TARGET = 22.0        # željena temperatura (°C)
STEPS = 240            # koraka u danu (korak = 6 min)
K1 = 0.02              # koliko brzo soba "curi" ka spoljnoj temperaturi
K2 = 0.30              # koliko jako klima mijenja temperaturu (°C po koraku)
NOISE = 0.05           # slučajni šum u fizici sobe
LAMBDA = 0.05          # težina kazne za potrošnju struje


class RoomEnv:
    """
    Jedan dan = jedna epizoda (240 koraka po 6 min).
    Stanje (skalirano, da tanh ne uđe u zasićenje):
        x1 = (T_soba - T_cilj) / 5
        x2 = (T_spolja - T_soba) / 10
    Akcija: snaga klime u [-1, 1] (-1 hlađenje, +1 grijanje).
    """

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.T = T_TARGET

    def outside(self, t):
        """Spoljna temperatura: sinusoida, maksimum oko 15h, 12 do 32 °C."""
        return 22.0 + 10.0 * np.sin(2 * np.pi * (t - 90) / STEPS)

    def _state(self):
        x1 = (self.T - T_TARGET) / 5.0
        x2 = (self.outside(self.t) - self.T) / 10.0
        return x1, x2

    def reset(self, T0=None):
        self.t = 0
        self.T = self.rng.uniform(18, 26) if T0 is None else T0
        return self._state()

    def step(self, action):
        a = float(np.clip(action, -1, 1))        # clip tek ovdje
        T_out = self.outside(self.t)
        self.T += K1 * (T_out - self.T) + K2 * a + NOISE * self.rng.standard_normal()
        reward = -(self.T - T_TARGET) ** 2 - LAMBDA * a ** 2
        self.t += 1
        done = self.t >= STEPS
        return self._state(), reward, done, a


def thermostat_action(T):
    """Obični on/off termostat sa mrtvom zonom ±0.5 °C."""
    if T < T_TARGET - 0.5:
        return 1.0
    if T > T_TARGET + 0.5:
        return -1.0
    return 0.0


# =====================================================================
# 3. TRENIRANJE (REINFORCE)
# =====================================================================
def play_day(env, policy, sigma):
    """Odigra jedan dan i vrati listu koraka i sirov zbir nagrada."""
    (x1, x2) = env.reset()
    steps, total = [], 0.0
    done = False
    while not done:
        a, m = policy.act(x1, x2, sigma)
        (nx1, nx2), r, done, _ = env.step(a)
        steps.append((x1, x2, a, m, r))
        total += r
        x1, x2 = nx1, nx2
    return steps, total


def returns_to_go(steps, gamma):
    """G_t unazad, zasebno za jednu epizodu."""
    out, G = [], 0.0
    for (x1, x2, a, m, r) in reversed(steps):
        G = r + gamma * G
        out.append((x1, x2, a, m, G))
    return out


def train(n_iters=600, batch_eps=5, gamma=0.99, alpha=0.1,
          sigma0=1.0, sigma_decay=0.9985, sigma_min=0.05, seed=0, verbose=True):
    env = RoomEnv(seed=seed)
    policy = Policy(seed=seed)
    sigma = sigma0
    log = {"ret": [], "w1": [], "w2": [], "b": [], "sigma": []}

    for it in range(n_iters):
        batch, rets = [], []
        for _ in range(batch_eps):
            steps, total = play_day(env, policy, sigma)
            batch += returns_to_go(steps, gamma)
            rets.append(total)

        # normalizacija povrata preko CIJELOG batch-a
        Gs = np.array([t[-1] for t in batch])
        Gs = (Gs - Gs.mean()) / (Gs.std() + 1e-8)
        batch = [(x1, x2, a, m, g) for (x1, x2, a, m, _), g in zip(batch, Gs)]

        policy.update(batch, sigma, alpha)

        # sigma se smanjuje po epizodi (ovdje 5 epizoda odjednom)
        sigma = max(sigma * sigma_decay ** batch_eps, sigma_min)

        log["ret"].append(np.mean(rets))
        log["w1"].append(policy.w1)
        log["w2"].append(policy.w2)
        log["b"].append(policy.b)
        log["sigma"].append(sigma)

        if verbose and (it % 25 == 0 or it == n_iters - 1):
            print(f"iter {it:4d} | povrat {np.mean(rets):9.1f} | sigma {sigma:.3f} | "
                  f"w1 {policy.w1:+.3f} w2 {policy.w2:+.3f} b {policy.b:+.3f}")
    return policy, log


# =====================================================================
# 4. EVALUACIJA, DEMO, GRAFIKONI
# =====================================================================
def run_day(kind, policy=None, seed=123):
    """Isti dan (isti seed, ista početna temperatura) za bilo kojeg kontrolera."""
    env = RoomEnv(seed=seed)
    x1, x2 = env.reset(T0=19.0)
    rec = {"T": [], "Tout": [], "a": [], "r": []}
    done = False
    while not done:
        if kind == "policy":
            a = policy.act_deterministic(x1, x2)
        else:
            a = thermostat_action(env.T)
        Tout = env.outside(env.t)
        (x1, x2), r, done, a_used = env.step(a)
        rec["T"].append(env.T); rec["Tout"].append(Tout)
        rec["a"].append(a_used); rec["r"].append(r)
    return rec


def avg_return(kind, policy=None, n=30):
    tot = []
    for s in range(n):
        env = RoomEnv(seed=1000 + s)
        x1, x2 = env.reset()
        total, done = 0.0, False
        while not done:
            a = policy.act_deterministic(x1, x2) if kind == "policy" else thermostat_action(env.T)
            (x1, x2), r, done, _ = env.step(a)
            total += r
        tot.append(total)
    return np.mean(tot)


def print_demo(untrained, trained):
    A = run_day("policy", untrained)
    B = run_day("policy", trained)
    C = run_day("thermostat")
    print("\nDemo istog dana (svakih sat vremena):")
    print(" sat | T_spolja | NETRENIRAN: T    akc | TRENIRAN: T    akc | TERMOSTAT: T    akc")
    for i in range(0, STEPS, 10):
        print(f" {i // 10:3d} |  {A['Tout'][i]:6.1f}  |   {A['T'][i]:6.2f}  {A['a'][i]:+.2f}   |"
              f"   {B['T'][i]:6.2f}  {B['a'][i]:+.2f}  |   {C['T'][i]:6.2f}  {C['a'][i]:+.2f}")
    print(f"\nZbir nagrada za taj dan:  netreniran {sum(A['r']):.1f} | "
          f"treniran {sum(B['r']):.1f} | termostat {sum(C['r']):.1f}")


def make_plots(log, untrained, trained, path):
    A = run_day("policy", untrained)
    B = run_day("policy", trained)
    C = run_day("thermostat")
    hours = np.arange(STEPS) * 6 / 60

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    # 1. kriva učenja
    r = np.array(log["ret"])
    ax[0, 0].plot(r, alpha=0.3)
    k = 15
    ax[0, 0].plot(np.arange(k - 1, len(r)), np.convolve(r, np.ones(k) / k, "valid"))
    ax[0, 0].set(title="Kriva učenja", xlabel="iteracija (batch od 5 dana)", ylabel="zbir nagrada po danu")
    ax[0, 0].set_yscale("symlog")

    # 2. temperatura
    ax[0, 1].plot(hours, A["Tout"], "k:", label="spolja")
    ax[0, 1].plot(hours, A["T"], label="netreniran")
    ax[0, 1].plot(hours, C["T"], label="termostat")
    ax[0, 1].plot(hours, B["T"], label="treniran")
    ax[0, 1].axhline(T_TARGET, color="gray", ls="--")
    ax[0, 1].set(title="Temperatura u sobi", xlabel="sat", ylabel="°C")
    ax[0, 1].legend()

    # 3. snaga klime
    ax[0, 2].plot(hours, A["a"], label="netreniran", alpha=0.7)
    ax[0, 2].plot(hours, C["a"], label="termostat", alpha=0.7)
    ax[0, 2].plot(hours, B["a"], label="treniran")
    ax[0, 2].set(title="Snaga klime", xlabel="sat", ylabel="akcija [-1, 1]")
    ax[0, 2].legend()

    # 4. težine
    ax[1, 0].plot(log["w1"], label="w1")
    ax[1, 0].plot(log["w2"], label="w2")
    ax[1, 0].plot(log["b"], label="b")
    ax[1, 0].set(title="Težine kroz trening", xlabel="iteracija")
    ax[1, 0].legend()

    # 5. sigma
    ax[1, 1].plot(log["sigma"])
    ax[1, 1].set(title="sigma kroz trening", xlabel="iteracija")

    # 6. potrošnja
    labels = ["netreniran", "termostat", "treniran"]
    energy = [np.sum(np.square(X["a"])) for X in (A, C, B)]
    ax[1, 2].bar(labels, energy)
    ax[1, 2].set(title="Potrošnja struje (zbir a²)", ylabel="a²")

    plt.tight_layout()
    plt.savefig(path, dpi=110)
    print(f"Grafikoni sačuvani: {path}")


# =====================================================================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    untrained = Policy(seed=0)                  # nasumične težine, bez učenja

    trained, log = train()
    trained.save("results/trained.npz")

    print_demo(untrained, trained)

    print("\nProsječan zbir nagrada po danu (30 novih dana, bez šuma u akciji):")
    print(f"  netreniran : {avg_return('policy', untrained):9.1f}")
    print(f"  termostat  : {avg_return('thermostat'):9.1f}")
    print(f"  treniran   : {avg_return('policy', trained):9.1f}")

    make_plots(log, untrained, trained, "results/grafikoni.png")