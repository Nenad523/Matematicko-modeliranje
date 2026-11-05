"""
Evaluacija, demo ispis i grafikoni

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

from environment import Room, OnOffThermostat, run_thermostat
from policy import Policy


# =====================================================================
# EVALUATOR
# =====================================================================
class Evaluator:
    """
    Sve što treba za dokaz da agent radi:
        - run_day: jedna deterministička epizoda za dati kontroler
        - avg_return: prosječan povrat preko n novih dana
        - print_demo: tabela sat-po-sat
        - make_plots: 6 grafikona
    """

    def __init__(self, trained_policy, log, seed_demo=123):
        self.policy = trained_policy
        self.log = log
        self.seed_demo = seed_demo

    
    # Jedan dan sa izabranim kontrolerom (isti seed -> isti dan)
    def run_day(self, kind, policy=None, seed=None, T0=19.0):
        """
        kind = "policy"     -> koristi proslijeđenu politiku (deterministički)
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
                a = policy.act_deterministic(x1, x2) \
                    if hasattr(policy, "act_deterministic") \
                    else policy.mu(x1, x2)
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
                    a = policy.act_deterministic(x1, x2) \
                        if hasattr(policy, "act_deterministic") \
                        else policy.mu(x1, x2)
                else:
                    a = thermostat.act(room.T_soba, room.T_CILJ)
                s_next, r, done, _ = room.step(a)
                total += r
                x1, x2 = float(s_next[0]), float(s_next[1])
            totals.append(total)
        return float(np.mean(totals))


    # Demo ispis u terminalu
    def print_demo(self, untrained):
        A = self.run_day("policy", untrained)
        B = self.run_day("policy", self.policy)
        C = self.run_day("thermostat")

        n = len(A["T"])
        # prikaži svaki 10. korak (svakih sat vremena)
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


    # 6 grafikona
    def make_plots(self, untrained, path="results/grafikoni.png"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        A = self.run_day("policy", untrained)
        B = self.run_day("policy", self.policy)
        C = self.run_day("thermostat")
        hours = A["t"]

        fig, ax = plt.subplots(2, 3, figsize=(15, 8))

        # 1. Kriva učenja
        r = np.array(self.log["ret"])
        ax[0, 0].plot(r, alpha=0.3)
        k = 15
        if len(r) >= k:
            ax[0, 0].plot(np.arange(k - 1, len(r)),
                          np.convolve(r, np.ones(k) / k, "valid"),
                          label="pokretni prosjek")
        ax[0, 0].set(title="Kriva učenja",
                     xlabel="iteracija (batch od 5 dana)",
                     ylabel="zbir nagrada po danu")
        ax[0, 0].set_yscale("symlog")
        ax[0, 0].legend()

        # 2. Temperatura
        ax[0, 1].plot(hours, A["T_spolja"], "k:", label="spolja")
        ax[0, 1].plot(hours, A["T"], label="netreniran")
        ax[0, 1].plot(hours, C["T"], label="termostat")
        ax[0, 1].plot(hours, B["T"], label="treniran")
        ax[0, 1].axhline(Room.T_CILJ, color="gray", ls="--", label="cilj")
        ax[0, 1].set(title="Temperatura u sobi",
                     xlabel="sat", ylabel="°C")
        ax[0, 1].legend()

        # 3. Snaga klime
        ax[0, 2].plot(hours, A["a"], label="netreniran", alpha=0.7)
        ax[0, 2].plot(hours, C["a"], label="termostat", alpha=0.7)
        ax[0, 2].plot(hours, B["a"], label="treniran")
        ax[0, 2].set(title="Snaga klime",
                     xlabel="sat", ylabel="akcija [-1, 1]")
        ax[0, 2].legend()

        # 4. Težine + b
        ax[1, 0].plot(self.log["w1"], label="w1")
        ax[1, 0].plot(self.log["w2"], label="w2")
        ax[1, 0].plot(self.log["b"], label="b")
        ax[1, 0].set(title="Težine kroz trening", xlabel="iteracija")
        ax[1, 0].legend()

        # 5. sigma
        ax[1, 1].plot(self.log["sigma"])
        ax[1, 1].set(title="sigma kroz trening", xlabel="iteracija")

        # 6. Potrošnja struje
        labels = ["netreniran", "termostat", "treniran"]
        energy = [float(np.sum(X["a"] ** 2)) for X in (A, C, B)]
        ax[1, 2].bar(labels, energy)
        ax[1, 2].set(title="Potrošnja struje (zbir a²)", ylabel="a²")

        plt.tight_layout()
        plt.savefig(path, dpi=110)
        plt.close(fig)
        print(f"Grafikoni sačuvani: {path}")


# =====================================================================
# Ako se fajl pokreće direktno: učítaj politiku, demo + grafikoni (nece se prikazati grafici 1,4,5 jer nema istorije)
# =====================================================================
if __name__ == "__main__":
    # Ako nema istrenirane politike, istreniraj brzo preko Trainer-a
    model_path = "results/trained.npz"

    if os.path.exists(model_path):
        trained = Policy(seed=0)
        trained.load(model_path)
        # pošto nemamo log, napravimo minimalni "prazan" log
        # (za pravi izvještaj pokreni train.py pa plots.py nad istim log-om)
        log = {"ret": [], "w1": [], "w2": [], "b": [], "sigma": []}
    else:
        print(f"Nema {model_path}; treniram brzo (100 iter) samo za demo...")
        from train import Trainer
        trainer = Trainer(n_iters=100, seed=0)
        trained, log = trainer.train(verbose=True)
        trainer.save(model_path)

    untrained = Policy(seed=0)

    ev = Evaluator(trained, log, seed_demo=123)
    ev.print_demo(untrained)

    print("\nProsječan zbir nagrada po danu (30 novih dana, bez šuma u akciji):")
    print(f"  netreniran : {ev.avg_return('policy', untrained):9.1f}")
    print(f"  termostat  : {ev.avg_return('thermostat'):9.1f}")
    print(f"  treniran   : {ev.avg_return('policy', trained):9.1f}")

    ev.make_plots(untrained, path="results/grafikoni.png")