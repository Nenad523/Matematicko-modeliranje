"""
Evaluacija, demo ispis i grafikoni.

Evaluator:
  - run_day: odigra JEDAN isti dan za dati kontroler (deterministički),
  - evaluate_returns: povrati preko n novih dana (za statistiku),
  - decompose: rastavlja kaznu na temperaturu, CO2 i energiju,
  - print_demo: tabela sat po sat,
  - make_plots: šest grafikona koji pričaju priču o učenju.

Pokretanje:  python plots.py        (trenira ako nema results/trained.npz)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                  # crta u fajl, bez prozora
import matplotlib.pyplot as plt

from environment import Room, OnOffThermostat
from policy import Policy

RESULTS = "results"
BOJA = {"agent": "#1f77b4", "termostat": "#d62728", "netreniran": "#7f7f7f"}


class Evaluator:

    def __init__(self, trained_policy, log=None, seed_demo=123):
        self.policy = trained_policy
        self.log = log or {}
        self.seed_demo = seed_demo

    # ---------------------------------------------------------------
    # Jedan dan sa izabranim kontrolerom (isti seed -> isti dan)
    # ---------------------------------------------------------------
    def run_day(self, kind, policy=None, seed=None, T0=19.0):
        """kind = "policy" (a = mu, bez šuma) ili "thermostat"."""
        seed = self.seed_demo if seed is None else seed
        room = Room(seed=seed)
        s = room.reset(T0=T0)

        thermostat = OnOffThermostat() if kind == "thermostat" else None
        if thermostat is not None:
            thermostat.reset()

        rec = {k: [] for k in ["T", "T_spolja", "CO2", "ljudi", "a1", "a2", "r", "t"]}
        done = False
        while not done:
            if kind == "policy":
                a = policy.mu(s)
            else:
                a = thermostat.act(room.T_room, room.T_TARGET, room.CO2_room)

            s, r, done, info = room.step(a)
            rec["T"].append(info["T_soba"])
            rec["T_spolja"].append(info["T_spolja"])
            rec["CO2"].append(info["CO2_soba"])
            rec["ljudi"].append(info["broj_ljudi"])
            rec["a1"].append(info["a1"])
            rec["a2"].append(info["a2"])
            rec["r"].append(r)
            rec["t"].append(info["t"])
        return {k: np.array(v) for k, v in rec.items()}

    # ---------------------------------------------------------------
    # Povrati preko n novih dana (deterministički, isti dani za sve)
    # ---------------------------------------------------------------
    def evaluate_returns(self, kind, policy=None, n=30, seed0=1000):
        totals = []
        for i in range(n):
            room = Room(seed=seed0 + i)
            s = room.reset()
            thermostat = OnOffThermostat() if kind == "thermostat" else None
            if thermostat is not None:
                thermostat.reset()

            total, done = 0.0, False
            while not done:
                if kind == "policy":
                    a = policy.mu(s)
                else:
                    a = thermostat.act(room.T_room, room.T_TARGET, room.CO2_room)
                s, r, done, _ = room.step(a)
                total += r
            totals.append(total)
        return np.array(totals)

    def avg_return(self, kind, policy=None, n=30):
        return float(self.evaluate_returns(kind, policy, n).mean())

    # ---------------------------------------------------------------
    # Rastavljanje kazne: na šta tačno odlazi nagrada
    # ---------------------------------------------------------------
    def decompose(self, kind, policy=None, n=30, seed0=1000):
        acc = np.zeros(4)                      # T^2, CO2 prekoracenje^2, a1^2, a2^2
        for i in range(n):
            room = Room(seed=seed0 + i)
            s = room.reset()
            thermostat = OnOffThermostat() if kind == "thermostat" else None
            if thermostat is not None:
                thermostat.reset()

            done = False
            while not done:
                if kind == "policy":
                    a = policy.mu(s)
                else:
                    a = thermostat.act(room.T_room, room.T_TARGET, room.CO2_room)
                s, r, done, info = room.step(a)
                acc += [(info["T_soba"] - Room.T_TARGET) ** 2,
                        max(0.0, info["CO2_soba"] - Room.CO2_LIMIT) ** 2,
                        info["a1"] ** 2, info["a2"] ** 2]
        acc /= n
        return dict(zip(["T2", "CO2_prekoracenje2", "a1_2", "a2_2"], acc))

    # ---------------------------------------------------------------
    def print_demo(self, untrained=None):
        A = self.run_day("policy", untrained) if untrained is not None else None
        B = self.run_day("policy", self.policy)
        C = self.run_day("thermostat")

        print("\nDemo istog dana (svaka dva sata):")
        print("  sat | T_spolja | ljudi |   AGENT: T   a1    a2  |  TERMOSTAT: T   a1    a2")
        for i in range(0, len(B["T"]), 20):
            print(f"  {int(round(B['t'][i])):3d} |  {B['T_spolja'][i]:6.1f}  |   "
                  f"{int(B['ljudi'][i]):d}   |  {B['T'][i]:6.2f} {B['a1'][i]:+.2f} {B['a2'][i]:5.2f}  |"
                  f"     {C['T'][i]:6.2f} {C['a1'][i]:+.2f} {C['a2'][i]:5.2f}")

        print(f"\n  povrat agenta    : {B['r'].sum():9.1f}")
        print(f"  povrat termostata: {C['r'].sum():9.1f}")
        if A is not None:
            print(f"  povrat netrenirane politike: {A['r'].sum():9.1f}")

    # ---------------------------------------------------------------
    def make_plots(self, untrained=None, path=os.path.join(RESULTS, "grafikoni.png")):
        os.makedirs(RESULTS, exist_ok=True)
        A = self.run_day("policy", untrained) if untrained is not None else None
        B = self.run_day("policy", self.policy)
        C = self.run_day("thermostat")
        h = B["t"]

        fig, ax = plt.subplots(2, 3, figsize=(17, 9))

        # 1) kriva učenja
        if "ret" in self.log:
            r = np.array(self.log["ret"])
            ax[0, 0].plot(r, alpha=0.3, color=BOJA["agent"])
            k = max(1, len(r) // 50)
            glat = np.convolve(r, np.ones(k) / k, mode="valid")
            ax[0, 0].plot(np.arange(len(glat)) + k // 2, glat, color=BOJA["agent"], lw=2)
            # zumiraj na dio gdje se vidi napredak (početni pad je ogroman)
            ax[0, 0].set_ylim(np.percentile(r, 2), max(r.max(), 0) + abs(r.max()) * 0.05 + 1)
        ax[0, 0].set_title("Kriva učenja")
        ax[0, 0].set_xlabel("iteracija"); ax[0, 0].set_ylabel("prosječan povrat po danu")
        ax[0, 0].grid(alpha=0.3)

        # 2) temperatura istog dana
        if A is not None:
            ax[0, 1].plot(h, A["T"], color=BOJA["netreniran"], ls=":", label="netreniran")
        ax[0, 1].plot(h, B["T"], color=BOJA["agent"], label="agent")
        ax[0, 1].plot(h, C["T"], color=BOJA["termostat"], label="termostat")
        ax[0, 1].plot(h, B["T_spolja"], color="tab:green", ls="--", alpha=0.6, label="spolja")
        ax[0, 1].axhline(Room.T_TARGET, ls="--", color="gray", label="cilj")
        ax[0, 1].set_title("Temperatura istog dana")
        ax[0, 1].set_xlabel("sat"); ax[0, 1].set_ylabel("°C")
        ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

        # 3) CO2 i prisutnost
        ax[0, 2].plot(h, B["CO2"], color=BOJA["agent"], label="agent")
        ax[0, 2].plot(h, C["CO2"], color=BOJA["termostat"], label="termostat")
        ax[0, 2].axhline(Room.CO2_LIMIT, ls="--", color="gray", label="granica")
        ax2 = ax[0, 2].twinx()
        ax2.fill_between(h, B["ljudi"], step="mid", alpha=0.15, color="tab:orange")
        ax2.set_ylabel("broj ljudi", color="tab:orange")
        ax[0, 2].set_title("CO2 tokom dana")
        ax[0, 2].set_xlabel("sat"); ax[0, 2].set_ylabel("ppm")
        ax[0, 2].legend(fontsize=8, loc="upper left"); ax[0, 2].grid(alpha=0.3)

        # 4) akcije
        ax[1, 0].plot(h, B["a1"], color=BOJA["agent"], label="agent: klima")
        ax[1, 0].plot(h, B["a2"], color=BOJA["agent"], ls="--", label="agent: ventilacija")
        ax[1, 0].plot(h, C["a1"], color=BOJA["termostat"], alpha=0.5, label="termostat: klima")
        ax[1, 0].plot(h, C["a2"], color=BOJA["termostat"], ls="--", alpha=0.5,
                      label="termostat: ventilacija")
        ax[1, 0].set_title("Akcije tokom dana")
        ax[1, 0].set_xlabel("sat"); ax[1, 0].set_ylabel("vrijednost akcije")
        ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

        # 5) parametri i sigma
        if "W1" in self.log:
            W1 = np.array(self.log["W1"]); W2 = np.array(self.log["W2"])
            ax[1, 1].plot(np.linalg.norm(W1, axis=(1, 2)), label="||W1||")
            ax[1, 1].plot(np.linalg.norm(W2, axis=(1, 2)), label="||W2||")
        if "sigma" in self.log:
            sg = np.array(self.log["sigma"])
            axs = ax[1, 1].twinx()
            axs.plot(sg[:, 0], color="tab:purple", alpha=0.7, label="σ klima")
            axs.plot(sg[:, 1], color="tab:brown", alpha=0.7, label="σ ventilacija")
            axs.set_ylabel("σ"); axs.legend(fontsize=8, loc="lower right")
        ax[1, 1].set_title("Težine i naučena σ")
        ax[1, 1].set_xlabel("iteracija"); ax[1, 1].set_ylabel("norma težina")
        ax[1, 1].legend(fontsize=8, loc="upper left"); ax[1, 1].grid(alpha=0.3)

        # 6) poređenje povrata preko 30 novih dana
        ret_a = self.evaluate_returns("policy", self.policy, n=30)
        ret_t = self.evaluate_returns("thermostat", n=30)
        serije = [ret_t, ret_a]
        oznake = ["termostat", "agent"]
        boje = [BOJA["termostat"], BOJA["agent"]]
        if untrained is not None:
            serije.insert(0, self.evaluate_returns("policy", untrained, n=30))
            oznake.insert(0, "netreniran"); boje.insert(0, BOJA["netreniran"])
        bp = ax[1, 2].boxplot(serije, tick_labels=oznake, showmeans=True, patch_artist=True)
        for patch, c in zip(bp["boxes"], boje):
            patch.set_facecolor(c); patch.set_alpha(0.4)
        if untrained is not None:
            ax[1, 2].set_yscale("symlog")          # netrenirana je red veličine lošija
        ax[1, 2].set_title("Povrat na 30 novih dana")
        ax[1, 2].set_ylabel("povrat po danu (symlog)"); ax[1, 2].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        print(f"Grafikoni sačuvani: {path}")


# ===================================================================
if __name__ == "__main__":
    from train import Trainer

    os.makedirs(RESULTS, exist_ok=True)
    model_path = os.path.join(RESULTS, "trained.npz")
    log_path = os.path.join(RESULTS, "trained_log.npz")

    if os.path.exists(model_path) and os.path.exists(log_path):
        print("Učitavam istreniran model iz results/ ...")
        policy = Policy(seed=0).load(model_path)
        log = {k: v for k, v in np.load(log_path).items()}
    else:
        print("Nema sačuvanog modela, treniram ...")
        trainer = Trainer(seed=0)
        policy, log = trainer.train(verbose=True)
        trainer.save(model_path)

    untrained = Policy(seed=0, H=policy.H)
    ev = Evaluator(policy, log)

    ev.print_demo(untrained)

    print("\nPovrat preko 30 novih dana (prosjek ± std):")
    for ime, kind, pol in [("netreniran", "policy", untrained),
                           ("agent", "policy", policy),
                           ("termostat", "thermostat", None)]:
        r = ev.evaluate_returns(kind, pol, n=30)
        print(f"  {ime:11s}: {r.mean():9.1f} ± {r.std():6.1f}")

    print("\nNa šta odlazi kazna (prosjek po danu):")
    for ime, kind, pol in [("agent", "policy", policy), ("termostat", "thermostat", None)]:
        d = ev.decompose(kind, pol, n=30)
        print(f"  {ime:11s}: Σ(T-22)² {d['T2']:8.1f} | ΣCO2 prekor.² {d['CO2_prekoracenje2']:12.0f} "
              f"| Σa1² {d['a1_2']:7.1f} | Σa2² {d['a2_2']:6.1f}")

    ev.make_plots(untrained)