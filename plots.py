"""
Evaluacija, demo ispis i grafikoni.

Sadrži klasu Evaluator koja:
  - odigrava ISTI dan za: netreniranu politiku, treniranu politiku i on/off termostat,
  - računa prosječan povrat preko više novih dana,
  - štampa demo tabelu u terminalu,
  - crta 6 grafikona koji pričaju priču o učenju.

Pokretanje:  python plots.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # bez prozora, crta u fajl
import matplotlib.pyplot as plt

from environment import Room, OnOffThermostat
from policy import Policy


# =====================================================================
# EVALUATOR
# =====================================================================
class Evaluator:
    """
    Sve što treba za dokaz da agent radi:
        - run_day: jedna deterministička epizoda za dati kontroler
        - avg_return: prosječan povrat preko n novih dana
        - evaluate_returns: vraća niz povrata (za statistiku)
        - print_demo: tabela sat-po-sat
        - make_comparison_plots: 6 grafikona poređenja
    """

    def __init__(self, trained_policy, log, seed_demo=123):
        self.policy = trained_policy
        self.log = log
        self.seed_demo = seed_demo

    # Jedan dan sa izabranim kontrolerom (isti seed -> isti dan)
    def run_day(self, kind, policy=None, seed=None, T0=19.0):
        """
        kind = "policy"     -> koristi proslijeđenu politiku (deterministički, a = mu)
        kind = "thermostat" -> on/off termostat
        Vraća dict sa T, T_spolja, a, r, t.
        """
        seed = self.seed_demo if seed is None else seed
        room = Room(seed=seed)
        s = room.reset(T0=T0)
        x1, x2 = float(s[0]), float(s[1])

        thermostat = OnOffThermostat() if kind == "thermostat" else None
        if thermostat is not None:
            thermostat.reset()

        rec = {"T": [], "T_spolja": [], "a": [], "r": [], "t": []}
        done = False
        while not done:
            if kind == "policy":
                # deterministička akcija: a = mu(x1, x2)
                a = policy.mu(x1, x2)
            else:
                a = thermostat.act(room.T_soba, room.T_CILJ)

            s_next, r, done, info = room.step(a)
            rec["T"].append(info["T_soba"])
            rec["T_spolja"].append(info["T_spolja"])
            rec["a"].append(info["a"])
            rec["r"].append(r)
            rec["t"].append(info["t"])
            x1, x2 = float(s_next[0]), float(s_next[1])

        return {k: np.array(v) for k, v in rec.items()}

    # Prosječan povrat preko n novih dana (bez šuma u akciji)
    def avg_return(self, kind, policy=None, n=30):
        totals = []
        for i in range(n):
            room = Room(seed=1000 + i)
            s = room.reset()
            x1, x2 = float(s[0]), float(s[1])

            thermostat = OnOffThermostat() if kind == "thermostat" else None
            if thermostat is not None:
                thermostat.reset()

            total, done = 0.0, False
            while not done:
                if kind == "policy":
                    a = policy.mu(x1, x2)
                else:
                    a = thermostat.act(room.T_soba, room.T_CILJ)

                s_next, r, done, _ = room.step(a)
                total += r
                x1, x2 = float(s_next[0]), float(s_next[1])
            totals.append(total)
        return float(np.mean(totals))

    # Niz povrata preko n dana (za statistiku: prosek, std, razlika)
    def evaluate_returns(self, kind, policy=None, n=30):
        totals = []
        for i in range(n):
            room = Room(seed=1000 + i)
            s = room.reset()
            x1, x2 = float(s[0]), float(s[1])

            thermostat = OnOffThermostat() if kind == "thermostat" else None
            if thermostat is not None:
                thermostat.reset()

            total, done = 0.0, False
            while not done:
                if kind == "policy":
                    a = policy.mu(x1, x2)
                else:
                    a = thermostat.act(room.T_soba, room.T_CILJ)

                s_next, r, done, _ = room.step(a)
                total += r
                x1, x2 = float(s_next[0]), float(s_next[1])
            totals.append(total)

        return np.array(totals)

    # Demo ispis u terminalu
    def print_demo(self, untrained):
        A = self.run_day("policy", untrained)
        B = self.run_day("policy", self.policy)
        C = self.run_day("thermostat")

        n = len(A["T"])
        print("\nDemo istog dana (svakih sat vremena):")
        print(" sat | T_spolja | NETRENIRAN: T    akc | TRENIRAN: T    akc | "
              "TERMOSTAT: T    akc")
        for i in range(0, n, 10):
            sat = int(round(A["t"][i]))
            print(f" {sat:3d} |  {A['T_spolja'][i]:6.1f}  |   "
                  f"{A['T'][i]:6.2f}  {A['a'][i]:+.2f}   |   "
                  f"{B['T'][i]:6.2f}  {B['a'][i]:+.2f}  |   "
                  f"{C['T'][i]:6.2f}  {C['a'][i]:+.2f}")

        print(f"\nZbir nagrada za taj dan:  "
              f"netreniran {A['r'].sum():.1f} | "
              f"treniran {B['r'].sum():.1f} | "
              f"termostat {C['r'].sum():.1f}")

    # 6 grafikona: poređenje fiksne vs. naučene σ
    def make_comparison_plots(self, fixed, fixed_log, learned, learned_log,
                              untrained, path="results/poredjenje.png"):

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Isti dan za sve: koristimo isti seed_demo
        fixed_ev = Evaluator(fixed, fixed_log, seed_demo=self.seed_demo)
        learned_ev = Evaluator(learned, learned_log, seed_demo=self.seed_demo)

        A = fixed_ev.run_day("policy", fixed)
        B = learned_ev.run_day("policy", learned)
        C = fixed_ev.run_day("thermostat")

        hours = A["t"]

        fig, ax = plt.subplots(2, 3, figsize=(17, 9))

        # ---- 1. Krive učenja ----
        for log, name in [(fixed_log, "Sigma (ručno)"),
                          (learned_log, "Naučena sigma")]:
            r = np.array(log["ret"])
            ax[0, 0].plot(r, alpha=0.2)
            k = 15
            if len(r) >= k:
                smooth = np.convolve(r, np.ones(k) / k, "valid")
                ax[0, 0].plot(np.arange(k - 1, len(r)), smooth, label=name)

        ax[0, 0].set_title("Kriva učenja")
        ax[0, 0].set_xlabel("Iteracija")
        ax[0, 0].set_ylabel("Povrat po danu")
        ax[0, 0].legend()
        ax[0, 0].grid()

        # ---- 2. Temperatura ----
        ax[0, 1].plot(hours, A["T"], label="Sigma (ručno)")
        ax[0, 1].plot(hours, B["T"], label="Naučena sigma")
        ax[0, 1].plot(hours, C["T"], label="Termostat")
        ax[0, 1].axhline(Room.T_CILJ, ls="--", color="gray",
                         label="Ciljna temperatura")
        ax[0, 1].set_title("Temperatura sobe")
        ax[0, 1].set_xlabel("Sat")
        ax[0, 1].set_ylabel("°C")
        ax[0, 1].legend()
        ax[0, 1].grid()

        # ---- 3. Akcije ----
        ax[0, 2].plot(hours, A["a"], label="Sigma (ručno)")
        ax[0, 2].plot(hours, B["a"], label="Naučena sigma")
        ax[0, 2].plot(hours, C["a"], label="Termostat")
        ax[0, 2].set_title("Snaga klime")
        ax[0, 2].set_xlabel("Sat")
        ax[0, 2].set_ylabel("Akcija [-1, 1]")
        ax[0, 2].legend()
        ax[0, 2].grid()

        # ---- 4. Norme težina (umjesto w1, w2, b) ----
        # A, c, w, v su matrice/vektori -> crtamo njihove norme kroz vrijeme.
        for log, suffix in [(fixed_log, "fiksna"),
                            (learned_log, "naučena")]:
            A_norm = np.linalg.norm(np.array(log["A"]), axis=(1, 2))
            w_norm = np.linalg.norm(np.array(log["w"]), axis=1)
            v_norm = np.linalg.norm(np.array(log["v"]), axis=1)
            ax[1, 0].plot(A_norm, label=f"|A| {suffix}")
            ax[1, 0].plot(w_norm, "--", label=f"|w| {suffix}")
            ax[1, 0].plot(v_norm, ":", label=f"|v| {suffix}")

        ax[1, 0].set_title("Norme težina kroz trening")
        ax[1, 0].set_xlabel("Iteracija")
        ax[1, 0].set_ylabel("Norma")
        ax[1, 0].legend(fontsize=8)
        ax[1, 0].grid()

        # ---- 5. Sigma kroz trening ----
        ax[1, 1].plot(fixed_log["sigma"], label="Sigma (ručno)")
        ax[1, 1].plot(learned_log["sigma"], label="Naučena sigma")
        ax[1, 1].set_title("Sigma kroz trening")
        ax[1, 1].set_xlabel("Iteracija")
        ax[1, 1].set_ylabel("σ")
        ax[1, 1].legend()
        ax[1, 1].grid()

        # ---- 6. Potrošnja struje ----
        labels = ["Sigma (ručno)", "Naučena sigma", "Termostat"]
        energy = [
            float(np.sum(A["a"] ** 2)),
            float(np.sum(B["a"] ** 2)),
            float(np.sum(C["a"] ** 2)),
        ]
        ax[1, 2].bar(labels, energy)
        ax[1, 2].set_title("Potrošnja struje")
        ax[1, 2].set_ylabel("Zbir a²")
        ax[1, 2].tick_params(axis="x", rotation=15)
        ax[1, 2].grid(axis="y")

        fig.suptitle("Poređenje REINFORCE agenata", fontsize=16)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)

        print(f"Grafikon sačuvan: {path}")


# =====================================================================
# Ako se fajl pokreće direktno: učitaj modele, demo + grafikoni
# =====================================================================
if __name__ == "__main__":

    untrained = Policy(seed=0)

    # --- fiksna sigma ---
    fixed = Policy(seed=0)
    fixed.load("results/fixed.npz")

    # --- naučena sigma ---
    learned = Policy(seed=0)
    learned.load("results/learned.npz")

    # --- logovi ---
    fixed_data = np.load("results/fixed_log.npz")
    learned_data = np.load("results/learned_log.npz")

    fixed_log = {key: fixed_data[key] for key in fixed_data.files}
    learned_log = {key: learned_data[key] for key in learned_data.files}

    fixed_ev = Evaluator(fixed, fixed_log, seed_demo=123)
    learned_ev = Evaluator(learned, learned_log, seed_demo=123)

    # --- Demo ispis ---
    print("\n=== DEMO: netreniran vs. treniran (fiksna sigma) vs. termostat ===")
    fixed_ev.print_demo(untrained)

    # --- Evaluacija na 30 istih testnih dana ---
    fixed_returns = fixed_ev.evaluate_returns("policy", fixed)
    learned_returns = learned_ev.evaluate_returns("policy", learned)
    thermostat_returns = fixed_ev.evaluate_returns("thermostat")

    print("\nREZULTATI EVALUACIJE (30 dana)")
    print("-----------------------------------")

    for name, values in [
        ("Sigma (ručno)", fixed_returns),
        ("Naučena sigma", learned_returns),
        ("Termostat", thermostat_returns),
    ]:
        print(
            f"{name:15s} | "
            f"prosek: {np.mean(values):8.2f} | "
            f"std: {np.std(values):8.2f}"
        )

    difference = learned_returns - fixed_returns
    print("-----------------------------------")
    print(f"Razlika (naučena - fiksna): {np.mean(difference):.2f}")

    # --- Grafikoni ---
    fixed_ev.make_comparison_plots(
        fixed, fixed_log,
        learned, learned_log,
        untrained,
        path="results/poredjenje.png",
    )