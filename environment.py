"""
Svijet u kojem agent živi.

Simulirana soba sa:
  - HVAC aktuatorom (a1 ∈ [-1, 1])
  - motorizovanom ventilacionom klapnom (a2 ∈ [0, 1])
  - ljudima koji generišu toplotu i CO2
  - dnevnim rasporedom prisutnih osoba

Ugovor prema ostalim članovima:
    Room.reset() -> s (np.array shape (4,), float64)
    Room.step(a) -> (s_next, r, done, info)
        a = [a1, a2]  (ili skalar - tada se a2 tretira kao 0)
            a1 ∈ [-1, 1]: HVAC koeficijent (grijanje/hlađenje)
            a2 ∈ [ 0, 1]: ventilaciona klapna
       info = {"T_soba", "T_spolja", "CO2_soba", "broj_ljudi",
               "t", "T_cilj", "a1", "a2"}
"""

import numpy as np


class Room:
    """
    Fizika: 

    dT/dt = -K*(T-T_out) - K_VENT*a2*(T-T_out) + G*a1 + PERSON_HEAT*n
    dCO2/dt = CO2_GEN*n - (BASE_VENT + VENT_CO2*a2)*(CO2-CO2_out)

    Diskretizacija: Ojler, dt = 6 min = 0.1 h. Epizoda = 24 h = 240 koraka.

    """

    # --- fizika ---
    DT = 0.1            # h, korak (6 min)
    K = 0.10            # 1/h, curenje toplote kroz zidove
    G = 1.50            # °C/h pri |a|=1, maksimalna snaga klime
    T_TARGET = 22.0       # °C, željena temperatura

    # --- ventilacija ---
    K_VENT = 0.20       # 1/h, dodatno curenje toplote
    VENT_CO2 = 2.00     # 1/h, dodatno curenje CO2
    BASE_VENT_CO2 = 0.20     # 1/h, minimalno curenje CO2

    # -- ljudi ---
    PERSON_HEAT = 0.05  # °C/h po osobi
    CO2_GEN = 200.0       # ppm/h po osobi

    CO2_OUT = 400.0        # ppm, spoljni CO2

    # --- skaliranje stanja (da tanh ne uđe u zasićenje) ---
    SCALE = np.array([5.0, 10.0, 500.0, 10.0])
    
    # --- dnevni ciklus spoljne temperature ---
    T_OUT_MEAN = 17.5
    T_OUT_AMPL = 7.5
    T_OUT_PHASE_H = 15.0     # sat maksimuma (popodne)

    # --- epizoda ---
    STEPS_PER_DAY = 240

    # -- nagrada ---
    W_TEMP = 1.0
    W_HVAC = 0.05
    W_VENT = 0.02
    W_CO2 = 0.20
    CO2_LIMIT = 1000.0
    CO2_REWARD_SCALE = 500.0

    def __init__(self, seed=None):

        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.T_room = None
        self.CO2_room = None
        self.ppl_cnt = 0
        self.done = True

    # ---------- pomoćne ----------
    def _T_out(self, t):
        """Glatka sinusoida, period 24 h, maksimum u T_OUT_PHASE_H."""

        phase = 2.0 * np.pi * (t - (self.T_OUT_PHASE_H - 6.0)) / 24.0
        
        return self.T_OUT_MEAN + self.T_OUT_AMPL * np.sin(phase)

    def _num_people(self, t):
        """Broj ljudi u sobi, diskretna funkcija vremena."""

        t_mod = t % 24.0
        
        if t_mod < 6.0:
            return 2
        elif t_mod < 8.0:
            return 3
        elif t_mod < 10.0:
            return 1
        elif t_mod < 16.0:
            return 0
        elif t_mod < 18.0:
            return 1
        elif t_mod < 20.0:
            return 2
        elif t_mod < 22.0:
            return 3
        else:
            return 2

    def _state(self):

        T_room = self.T_room

        T_out = self._T_out(self.t)

        return np.array([
            (T_room - self.T_TARGET)        / self.SCALE[0],
            (T_out - T_room)                / self.SCALE[1],
            (self.CO2_room - self.CO2_OUT)  / self.SCALE[2],
            self.ppl_cnt                    / self.SCALE[3]
        ], dtype=np.float64)


    # ---------- ugovor ----------
    def reset(self, T0=None, CO2_0=None):
        """Novi dan sa nasumičnim početnim uslovima."""

        self.t = 0.0
        
        if T0 is None:
            self.T_room = float(self.T_TARGET + self.rng.uniform(-3.0, 3.0))
        else:
            self.T_room = float(T0)

        if CO2_0 is None:
            self.CO2_room = float(self.CO2_OUT + self.rng.uniform(0.0, 200.0))
        else:
            self.CO2_room = float(CO2_0)

        self.ppl_cnt = self._num_people(self.t)
        
        self.done = False

        return self._state()

    def step(self, action):
        """
        Odigraj jedan korak od 6 min.
        action: [a1, a2] u opsezima [-1,1] i [0,1]  (ili skalar → a2=0).
        Vrati (s_next, r, done, info).
        """
        assert not self.done, "Epizoda je završena; pozovi reset()."

        # akcija je realan koeficijent; ograničavamo na fizički opseg
        arr = np.asarray(action, dtype=np.float64).ravel()

        if arr.size == 1:
            a1 = np.clip(arr[0], -1.0, 1.0)
            a2 = 0.0
        else:
            a1 = np.clip(arr[0], -1.0, 1.0)
            a2 = np.clip(arr[1], 0.0, 1.0)

        T_out = self._T_out(self.t)
        n = self.ppl_cnt

        dT = (- self.K * (self.T_room - T_out)
              - self.K_VENT * a2 * (self.T_room - T_out)
              + self.G * a1
              + self.PERSON_HEAT * n)

        vent = self.BASE_VENT_CO2 + self.VENT_CO2 * a2
        dCO2 = (self.CO2_GEN * n - vent * (self.CO2_room - self.CO2_OUT))

        self.T_room = float(self.T_room + self.DT * dT)
        self.CO2_room = float(max(self.CO2_room + self.DT * dCO2, self.CO2_OUT))
        self.t += self.DT
        self.ppl_cnt = self._num_people(self.t)

        # nagrada:

        r_temp = -self.W_TEMP * (self.T_room - self.T_TARGET) ** 2
        r_hvac = -self.W_HVAC * a1 ** 2
        r_vent = -self.W_VENT * a2 ** 2
        r_co2 = -self.W_CO2 *   (max(0.0, self.CO2_room - self.CO2_LIMIT) / self.CO2_REWARD_SCALE) ** 2

        r = r_temp + r_hvac + r_vent + r_co2

        self.done = self.t >= self.DT * self.STEPS_PER_DAY - 1e-9

        info = {
            "T_room":     self.T_room,
            "T_out":      T_out,
            "CO2_room":   self.CO2_room,
            "num_people": n,
            "t":          self.t,
            "T_target":   self.T_TARGET,
            "a1":         a1,
            "a2":         a2    
        }
        return self._state(), float(r), bool(self.done), info


class OnOffThermostat:
    """
    Običan on/off termostat sa histerezom - mjerilo za poređenje.
      - a1: {-1, 0, +1} na osnovu temperature
      - a2: { 0,  1  } na osnovu CO2 (uključi kad pređe limit + histerezu)
    """

    def __init__(self, hysteresis=0.5, off_in_zone=False,
                 co2_limit=1000.0, co2_hysteresis=100.0):
        self.hysteresis = hysteresis
        self.off_in_zone = off_in_zone
        self.co2_limit = co2_limit
        self.co2_hysteresis = co2_hysteresis
        self.a1 = 0.0
        self.a2 = 0.0

    def reset(self):
        self.a1 = 0.0
        self.a2 = 0.0

    def act(self, T_room, T_target, CO2_room):
        # --- HVAC ---
        if T_room < T_target - self.hysteresis:
            self.a1 = 1.0
        elif T_room > T_target + self.hysteresis:
            self.a1 = -1.0
        elif self.off_in_zone:
            self.a1 = 0.0

        # --- ventilacija ---
        if CO2_room > self.co2_limit + self.co2_hysteresis:
            self.a2 = 1.0
        elif CO2_room < self.co2_limit - self.co2_hysteresis:
            self.a2 = 0.0

        return np.array([self.a1, self.a2])



def run_thermostat(room, n_episodes=1, seed=999, hysteresis=0.5, off_in_zone=False):
    """Odigraj on/off termostat u istoj sobi. Vrati trajektorije."""
    trajs = []
    for i in range(n_episodes):
        room.rng = np.random.default_rng(seed + i)
        room.reset()
        th = OnOffThermostat(hysteresis=hysteresis, off_in_zone=off_in_zone)
        th.reset()
        T_rooms, T_outs, CO2s, acts, ts = [], [], [], [], []
        done = False
        while not done:
            a = th.act(room.T_room, room.T_TARGET, room.CO2_room)
            _, _, done, info = room.step(a)
            T_rooms.append(info["T_room"])
            T_outs.append(info["T_out"])
            CO2s.append(info["CO2_room"])
            acts.append([info["a1"], info["a2"]])
            ts.append(info["t"])
        trajs.append({
            "T_room":  np.array(T_rooms),
            "T_out":   np.array(T_outs),
            "CO2":     np.array(CO2s),
            "actions": np.array(acts),   # shape (240, 2)
            "t":       np.array(ts),
        })
    return trajs


# ============================================================
# Brza provjera fizike (checklist prije predaje ostalima)
# ============================================================
if __name__ == "__main__":
    print("=== PROŠIRENA SOBA: HVAC + ventilacija + CO2 + ljudi ===\n")

    # 1) Bez ikakve kontrole (a1=0, a2=0) - temperatura prati spolja, CO2 eksplodira
    room = Room(seed=0)
    room.reset()
    print("[1] Bez kontrole (a1=0, a2=0)")
    for _ in range(Room.STEPS_PER_DAY):
        s, r, done, info = room.step([0.0, 0.0])
        if int(round(info["t"] * 10)) % 60 == 0:
            print(f"  t={info['t']:5.1f}h  T={info['T_room']:6.2f}°C  "
                  f"CO2={info['CO2_room']:7.1f}ppm  n={info['num_people']}")
        if done: break

    # 2) Samo HVAC (termostatski), a2=0 - CO2 raste bez ventilacije
    print("\n[2] Samo HVAC (a1 iz termostata, a2=0):")
    room = Room(seed=0); room.reset()
    while not room.done:
        if room.T_room < Room.T_TARGET - 0.5: a1 = 1.0
        elif room.T_room > Room.T_TARGET + 0.5: a1 = -1.0
        else: a1 = 0.0
        _, _, done, info = room.step([a1, 0.0])
    print(f"  kraj: T={info['T_room']:.2f}°C  "
          f"CO2={info['CO2_room']:.0f}ppm  (vrlo visok CO2!)")

    # 3) Puna ventilacija, bez HVAC-a (a1=0, a2=1) - CO2 nizak, temperatura bježi
    print("\n[3] Puna ventilacija, bez HVAC (a1=0, a2=1):")
    room = Room(seed=0); room.reset()
    T_rooms, CO2s = [], []
    while not room.done:
        _, _, done, info = room.step([0.0, 1.0])
        T_rooms.append(info["T_room"]); CO2s.append(info["CO2_room"])
    print(f"  kraj: T={T_rooms[-1]:.2f}°C  CO2={CO2s[-1]:.0f}ppm")
    print(f"  T min/max: {min(T_rooms):.2f}/{max(T_rooms):.2f}  "
          f"CO2 max: {max(CO2s):.0f}ppm")

    # 4) Bazna linija: on/off termostat (sa i bez histerezne zone za HVAC)
    for zone in (False, True):
        trajs = run_thermostat(Room(seed=0), n_episodes=1, seed=999,
                               off_in_zone=zone)
        th = trajs[0]
        temp_err = float(np.sum((th["T_room"] - Room.T_TARGET) ** 2))
        over     = np.maximum(0.0, th["CO2"] - Room.CO2_LIMIT)
        co2_err  = float(np.sum(over ** 2))
        print(f"\n[4] On/off termostat (off_in_zone={zone}):")
        print(f"  temp greška Σ(T-22)² : {temp_err:8.2f}")
        print(f"  CO2 prekoračenje² Σ  : {co2_err:8.1f}")
        print(f"  T   min/max          : {th['T_room'].min():.2f}/{th['T_room'].max():.2f}")
        print(f"  CO2 min/max          : {th['CO2'].min():.0f}/{th['CO2'].max():.0f}")
        print(f"  prosječno a1², a2²   : "
              f"{np.mean(th['actions'][:,0]**2):.3f}, "
              f"{np.mean(th['actions'][:,1]**2):.3f}")