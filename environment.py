"""
Svijet u kojem agent živi.
Simulirana soba + on/off termostat kao mjerilo.

Ugovor prema ostalim članovima:
    Room.reset() -> s (np.array shape (2,), float64)
    Room.step(a) -> (s_next, r, done, info)
        a ∈ [-1, 1] realan koeficijent:
            a = -1  -> maksimalno hlađenje
            a =  0  -> klima isključena
            a = +1  -> maksimalno grijanje
            između -> proporcionalna snaga
       info = {"T_soba", "T_spolja", "t", "T_cilj", "a"}
"""

import numpy as np


class Room:
    """
    Fizika: Njutnov zakon hlađenja + proporcionalni doprinos klime.
        dT_soba/dt = -K * (T_soba - T_spolja) + G * a
    Diskretizacija: Ojler, dt = 6 min = 0.1 h. Epizoda = 24 h = 240 koraka.

    Stanje: s = [(T_soba - T_cilj)/3, (T_spolja - T_soba)/10]   (skalirano)
    Akcija: a ∈ [-1, 1], realan koeficijent (nije diskretan skup!)
    Nagrada: r = -(T_soba - T_cilj)^2 - W_STRUJA * a^2
    """

    # --- fizika ---
    DT = 0.1            # h, korak (6 min)
    K = 0.10            # 1/h, curenje toplote kroz zidove
    G = 1.50            # °C/h pri |a|=1, maksimalna snaga klime
    T_CILJ = 21.0       # °C, željena temperatura

    # --- skaliranje stanja (da tanh ne uđe u zasićenje) ---
    SKALA = np.array([3.0, 10.0])
    
    # --- dnevni ciklus spoljne temperature ---
    T_SPOLJA_SREDNJA = 17.5
    T_SPOLJA_AMPLITUDA = 7.5
    T_SPOLJA_FAZA_H = 15.0     # sat maksimuma (popodne)

    # --- epizoda ---
    STEPS_PER_DAY = 240
    W_STRUJA = 0.05

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.T_soba = None
        self.done = True

    # ---------- pomoćne ----------
    def _T_spolja(self, t):
        """Glatka sinusoida, period 24 h, maksimum u T_SPOLJA_FAZA_H."""
        phase = 2.0 * np.pi * (t - (self.T_SPOLJA_FAZA_H - 6.0)) / 24.0
        return self.T_SPOLJA_SREDNJA + self.T_SPOLJA_AMPLITUDA * np.sin(phase)

    def _state(self):
        Ts = self.T_soba
        Tsp = self._T_spolja(self.t)
        s = np.array([Ts - self.T_CILJ, Tsp - Ts], dtype=np.float64)
        return s / self.SKALA

    # ---------- ugovor ----------
    def reset(self, T0=None):
        """Novi dan sa nasumičnim početnim uslovima."""
        self.t = 0.0
        if T0 is None:
            self.T_soba = float(self.T_CILJ + self.rng.uniform(-3.0, 3.0))
        else:
            self.T_soba = float(T0)
        self.done = False
        return self._state()

    def step(self, action):
        """
        Odigraj jedan korak od 6 min.
        action: realan broj iz [-1, 1] - koeficijent grijanja/hlađenja.
        Vrati (s_next, r, done, info).
        """
        assert not self.done, "Epizoda je završena; pozovi reset()."

        # akcija je realan koeficijent; ograničavamo na fizički opseg
        a = float(np.clip(action, -1.0, 1.0))

        Tsp = self._T_spolja(self.t)

        # Ojler: eksplicitno, koristi trenutne vrijednosti.
        # Klima doprinosi proporcionalno koeficijentu a (nije diskretno!).
        dT = -self.K * (self.T_soba - Tsp) + self.G * a
        self.T_soba = float(self.T_soba + self.DT * dT)
        self.t += self.DT

        # nagrada: komfor + štednja (a^2 kažnjava i jako hlađenje i jako grijanje)
        r = -(self.T_soba - self.T_CILJ) ** 2 - self.W_STRUJA * a ** 2

        self.done = self.t >= self.DT * self.STEPS_PER_DAY - 1e-9

        info = {
            "T_soba": self.T_soba,
            "T_spolja": Tsp,
            "t": self.t,
            "T_cilj": self.T_CILJ,
            "a": a,        # klipovana akcija koja je zaista primijenjena
        }
        return self._state(), float(r), bool(self.done), info


class OnOffThermostat:
    """
    Običan on/off termostat sa histerezom - mjerilo za poređenje.
    Vraća diskretnu akciju {-1, 0, +1} (to je njegova priroda),
    ali Room.step prima bilo koji realan broj iz [-1, 1], pa je interfejs isti.
    """

    def __init__(self, histereza=0.5, iskljuci_u_zoni=False):
        self.histereza = histereza
        self.iskljuci_u_zoni = iskljuci_u_zoni
        self.a = 0.0

    def reset(self):
        self.a = 0.0

    def act(self, T_soba, T_cilj):
        if T_soba < T_cilj - self.histereza:
            self.a = 1.0
        elif T_soba > T_cilj + self.histereza:
            self.a = -1.0
        elif self.iskljuci_u_zoni:
            self.a = 0.0
        return self.a


def run_thermostat(room, n_episodes=1, seed=999, histereza=0.5, iskljuci_u_zoni=False):
    """Odigraj on/off termostat u istoj sobi. Vrati trajektorije."""
    trajs = []
    for i in range(n_episodes):
        room.rng = np.random.default_rng(seed + i)
        room.reset()
        th = OnOffThermostat(histereza=histereza, iskljuci_u_zoni=iskljuci_u_zoni)
        th.reset()
        Ts, Tsp, acts, ts = [], [], [], []
        done = False
        while not done:
            a = th.act(room.T_soba, room.T_CILJ)
            _, _, done, info = room.step(a)
            Ts.append(info["T_soba"]); Tsp.append(info["T_spolja"])
            acts.append(a); ts.append(info["t"])
        trajs.append({"T_soba": np.array(Ts), "T_spolja": np.array(Tsp),
                      "actions": np.array(acts), "t": np.array(ts)})
    return trajs


# ============================================================
# Brza provjera fizike (checklist prije predaje ostalima)
# ============================================================
if __name__ == "__main__":
    # 1) soba bez klime prati spoljnu
    room = Room(seed=0)
    room.reset()
    print("=== bez klime (a=0) ===")
    for _ in range(240):
        s, r, done, info = room.step(0.0)
        if int(info["t"] * 10) % 60 == 0:
            print(f"t={info['t']:5.1f}h  T_soba={info['T_soba']:6.2f}  "
                  f"T_spolja={info['T_spolja']:6.2f}  r={r:8.3f}")
        if done:
            break

    # 2) klima na maksimumu grije, na minimumu hladi
    for a_fix, label in [(1.0, "max grijanje"), (-1.0, "max hlađenje")]:
        room = Room(seed=0)
        room.reset()
        T0 = room.T_soba
        for _ in range(240):
            s, r, done, info = room.step(a_fix)
            if done:
                break
        print(f"=== {label} (a={a_fix:+.1f}) ===  "
              f"T_soba: {T0:.2f} -> {info['T_soba']:.2f}")

    # 3) realan koeficijent između -1 i 1
    print("=== realan koeficijent a=0.37 (blago grijanje) ===")
    room = Room(seed=0)
    room.reset()
    T0 = room.T_soba
    for _ in range(240):
        s, r, done, info = room.step(0.37)
        if done:
            break
    print(f"T_soba: {T0:.2f} -> {info['T_soba']:.2f}")

    # 4) termostat, obje varijante
    for zona in (False, True):
        th = run_thermostat(Room(seed=0), n_episodes=1, seed=999,
                            iskljuci_u_zoni=zona)[0]
        err = float(np.sum((th["T_soba"] - Room.T_CILJ) ** 2))
        cons = float(np.sum(th["actions"] ** 2))
        print(f"=== termostat, iskljuci_u_zoni={zona} ===")
        print(f"greška={err:.2f}, potrošnja Σa²={cons:.2f}, "
              f"T_soba min/max={th['T_soba'].min():.2f}/{th['T_soba'].max():.2f}")
