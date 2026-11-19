# Spisak svih izmjena

Dokument opisuje **šta je promijenjeno u odnosu na verziju u repou** (commit `1c38eab`), zašto, i koga se tiče. Podijeljeno po oblastima i članovima.

Oznake:
**[Č1]** okruženje · **[Č2]** mreža i izvodi · **[Č3]** trening, grafikoni, evaluacija · **[DOK]** dokumentacija

Najvažnija izmjena je u odjeljku 1: okruženje je imalo grešku zbog koje agent **nije mogao** dobro naučiti zadatak, bez obzira na mrežu i trening.

---

## 1. [Č1] Okruženje — `environment.py`

Izmijenjene su **tri konstante**. Sve ostalo (fizika, raspored ljudi, nasumični dani, `info` ključevi, on/off termostat) ostalo je nepromijenjeno.

### 1.1. `G`: 1.50 → **3.50** — glavna ispravka

**Problem:** ravnotežna temperatura pri punoj klimi je

$$T_{\text{ravnoteža}} = T_{\text{spolja}} + \frac{G}{K}$$

Sa starim vrijednostima `G = 1.50` i `K = 0.10` to daje samo $T_{\text{spolja}} + 15$.

Pošto `reset()` bira nasumične dane u kojima spoljna temperatura pada do **−3.4 °C**, cilj od 22 °C bio je **fizički nedostižan dijelom dana na 17 od 20 testnih dana**. Agent je tada kažnjavan za nešto na šta ne može uticati, a ta neizbježna kazna (Σ(T−22)² reda 1800) bila je toliko velika da je zasjenila sve ostalo, uključujući CO₂.

**Posljedica koju je to izazvala:** izgledalo je kao da agent ne želi da ventilira i da ne uči dobro. U stvari, nije imao šta da optimizuje.

**Provjereno:** pri `G = 1.50` povećavanje `W_CO2` na 5, 20 ili 50 **pogoršava** rezultat umjesto da pomogne.

**Sada:** `G/K = 35`, pa je max dostižna temperatura $T_{\text{spolja}} + 35$ i cilj je dostižan na svim danima.

**Za odbranu:** `G` je maksimalna brzina promjene temperature koju klima može postići. Vrijednost od 3.5 °C/h odgovara pravilno dimenzionisanom uređaju za sobu ove veličine. Konstante su pojednostavljene i to treba navesti u radu.

### 1.2. `SCALE`: `[5.0, 10.0, 500.0, 10.0]` → **`[1.5, 10.0, 700.0, 3.0]`**

Skale dijele ulaze prije mreže. Stare vrijednosti su loše koristile opseg:

| Ulaz | Stara skala | Stvarni raspon ulaza | Problem |
| :--- | ---: | :--- | :--- |
| $x_1$ temperatura | 5.0 | ±0.15 | premali signal |
| $x_3$ CO₂ | 500.0 | do 1.6 i preko | moglo preći 3 i zasititi tanh |
| $x_4$ broj ljudi | 10.0 | 0 do 0.3 | ljudi za mrežu skoro nevidljivi |

Nove skale drže svaki ulaz u opsegu u kojem $\tanh$ nije ravan. Mjereno: sa novim skalama agent dostiže bolji rezultat za 800 iteracija nego sa starim za 2000.

### 1.3. `W_CO2`: 0.20 → **10.0**

Kazna za CO₂ je bila oko **100 puta slabija** od kazne za temperaturu. Pri `W_CO2 = 0.20` i `CO2_REWARD_SCALE = 500`, prekoračenje od 500 ppm vrijedilo je koliko 0.45 °C odstupanja, pa je agent potpuno ignorisao ventilaciju i druga akcija je bila ukrasna.

Sada 500 ppm preko granice vrijedi otprilike koliko 3.2 °C odstupanja.

**Rezultat izmjene:** vrijeme iznad 1000 ppm palo je sa 42.9 % na 9.2 %, a korišćenje ventilacije (Σa₂²) poraslo sa 15 na 34.

### 1.4. Šta član 1 NIJE morao mijenjati

Dvije stvari koje sam ranije prijavio **on je već sam ispravio** prije ovih izmjena:
- ključevi u `info` sada slijede ugovor (`T_soba`, `T_spolja`, `CO2_soba`, `broj_ljudi`, `T_cilj`, `a1`, `a2`),
- `reset()` sada bira nasumičan dan (srednja spoljna temperatura, amplituda, faza), pa agent ne može naučiti jedan dan napamet.

---

## 2. [Č2] Mreža — `policy.py`

Fajl je **napisan iznova**, jer je stara verzija bila za prethodni, jednostavniji zadatak.

### 2.1. Struktura

| | Bilo | Sada |
| :--- | :--- | :--- |
| Ulaz | 2 broja (`x1`, `x2`) | **4 broja** (vektor stanja) |
| Skriveni sloj | `N = 4`, matrica `A`, bias `c` | **`H` neurona** (podrazumijevano 8), `W1`, `b1` |
| Izlaz | 1 akcija: `w`→μ, `v`→σ | **4 izlaza**: μ₁, σ₁, μ₂, σ₂ (`W2`, `b2`) |
| Akcije | 1 (klima) | **2** (klima + ventilacija) |
| Broj parametara | 21 | **9H + 4** (za H=8: **76**) |

### 2.2. Aktivacione funkcije

| Izlaz | Bilo | Sada | Izvod koji treba u backpropu |
| :--- | :--- | :--- | :--- |
| μ₁ (klima, [−1,1]) | tanh | tanh | $1 - \mu_1^2$ |
| μ₂ (ventilacija, [0,1]) | — | **sigmoid** | $\mu_2(1-\mu_2)$ |
| σ₁, σ₂ | `0.05 + 0.95·sigmoid` | **`SIGMA_MIN + softplus`** | $\text{sigmoid}(z)$ |

Softplus je izabran jer ne ograničava σ odozgo i ne prelijeva se za velike $z$.

### 2.3. Donja granica za σ

Dodano `SIGMA_MIN = 0.05`. U gradijentu se pojavljuje član $1/\sigma^2$, pa bi za $\sigma \to 0$ korak postao ogroman i trening bi se raspao. Konstanta ne mijenja izvode (izvod konstante je nula).

### 2.4. Početne vrijednosti

| Parametar | Bilo | Sada | Zašto |
| :--- | :--- | :--- | :--- |
| `W1` | $\mathcal{N}(0, 2^2)$ | $\mathcal{N}(0, 0.5^2)$ | σ=2 gura tanh odmah u zasićenje, pa gradijent iščezava |
| `W2` | $\mathcal{N}(0, 2^2)$ | $\mathcal{N}(0, 0.1^2)$ | mreža kreće sa μ₁≈0 (klima isključena) i μ₂≈0.5 |
| bias za σ | 0 | $\ln(e^{0.25}-1) \approx -1.26$ | bez toga je početni σ ≈ 0.74, što je previše za akciju u [0,1]; sada je ≈ 0.30 |

### 2.5. Izbačen fiksni σ

Zastavica `learn_sigma` i parametar `fixed_sigma` su uklonjeni. **σ se uvijek uči kao izlaz mreže.** Time nestaje grananje u `act` i u `grad`.

### 2.6. Vektorizacija i ostalo

- `forward(S)` radi i za jedno stanje `(4,)` i za cijeli batch `(n, 4)`, pa `grad` više nema petlju po koracima (brže i kraće).
- `grad(batch)` vraća listu `[gW1, gb1, gW2, gb2]`; zaštićen je slučaj praznog batcha.
- `update(batch, alpha, clip_norm=None)` ima opciono **ograničenje norme gradijenta** (koristi se `clip_norm=5.0`), kao zaštita od rijetkih ogromnih koraka.
- `log_prob_batch(S, U)` dodan zbog numeričke provjere gradijenta.
- `save`/`load` koriste ključeve `W1, b1, W2, b2, H`.
- `softplus` je implementiran preko `np.logaddexp(0, z)` radi numeričke stabilnosti.

### 2.7. Provjera gradijenta — `check_gradient.py`

Fajl je **napisan iznova**. Stara verzija je provjeravala samo tri skalarna parametra (`w1`, `w2`, `b`) i zahtijevala ručno prosljeđivanje argumenata.

Nova verzija:
- provjerava **sve parametre** (`W1`, `b1`, `W2`, `b2`), element po element,
- koristi centralne razlike na surogat-funkciji $L = \frac{1}{N}\sum_t G_t \log\pi(a_t|s_t)$,
- testira **tri veličine mreže**: H = 3, 8, 16,
- parametre prvo nasumično poremeti, da provjera ne bude trivijalna,
- štampa jasnu poruku `GRADIJENT JE ISPRAVAN` ili `GREŠKA U GRADIJENTU`.

**Rezultat:** najveća relativna greška je reda **10⁻⁹ do 10⁻¹¹** za sve parametre i sve veličine mreže. Ručni backprop je potvrđen.

### 2.8. Ispravke u `Izvodi.pdf` (rukom pisani izvod)

Izvod je u suštini tačan. Tri sitne greške u međukoracima (konačni rezultati su dobri):

1. **Strana 4:** piše `d/dz tanh(z) = 1 − tanh(z)`, treba **`1 − tanh²(z)`**. Rezultat `1 − h²` je ispravan.
2. **Strana 4:** u razvoju za $z^{(1)}$ stoji $W^{(2)}_{2j}$, a treba **$W^{(1)}_{2j}$** (težina prvog sloja).
3. **Strana 3, $\Delta_4$:** treba `sigmoid(z₄)`, kao u $\Delta_2$ (provjeriti rukopis).

Dodatno:
- σ je u rukopisu pisano kao δ, a δ se koristi i za greške backpropa — preporučujem da se σ piše jasno, jer je vjerovatno pitanje na odbrani.
- Prelazak $R(\tau) \to G_t$ napisan je kao jednakost; treba dodati rečenicu da nagrade iz prošlosti ne zavise od budućih akcija, pa im je doprinos u očekivanju nula.
- Strana 1 koristi $1/M$ (epizode), strana 6 koristi $1/N$ (koraci) — uskladiti oznaku (razlika je konstantan faktor koji upije α).

---

## 3. [Č3] Trening — `train.py`

### 3.1. Prilagođavanje novoj politici i okruženju

| | Bilo | Sada |
| :--- | :--- | :--- |
| Stanje | `x1, x2 = s[0], s[1]` (odbacivao 2 ulaza) | cijeli vektor `s` |
| Korak u batchu | `(x1, x2, u, m, sigma, r)` | `(s, u, m, sigma, r)` |
| Konstruktor politike | `Policy(seed, N=8)` | `Policy(seed, H=8)` |
| Log težina | `A, c, w, b, v, d` | `W1, b1, W2, b2` |
| σ u logu | skalar | vektor od 2 (po jedan za svaku akciju) |

### 3.2. Izbačeno

`learn_sigma`, `sigma0`, `sigma_decay`, `sigma_min` i cijeli raspored po kojem je σ ručno opadao. σ se sada uvijek uči.

### 3.3. Promijenjeni hiperparametri

| Parametar | Bilo | Sada | Zašto |
| :--- | ---: | ---: | :--- |
| `n_iters` | 500 | **1500** | pri 500 trening još nije konvergirao; mjereno 800 vs 1500 vs 2000, 1500 je dovoljno (oko 80 s) |
| `alpha_min` | 0.01 | **0.02** | sa 0.01 učenje se prerano uspori |
| `clip_norm` | — | **5.0** | nov parametar, ograničava normu gradijenta |

Nepromijenjeni: `batch_eps = 7`, `gamma = 0.95`, `alpha0 = 0.2`, `alpha_decay = 0.9995`.

### 3.4. Sitnije

- Dodan ispis proteklog vremena tokom treninga.
- `save` zapisuje cijeli log kroz `np.savez(**log)` umjesto nabrajanja ključeva.

---

## 4. [Č3] Evaluacija i grafikoni — `plots.py`

### 4.1. Ispravljene greške koje su rušile fajl

Stara verzija **nije mogla da se pokrene**. Pozivala je nepostojeća imena:

| Bilo (greška) | Sada |
| :--- | :--- |
| `policy.mu(x1, x2)` | `policy.mu(s)` |
| `room.T_soba` | `room.T_room` |
| `Room.T_CILJ` | `Room.T_TARGET` |
| `thermostat.act(T, T_cilj)` (2 argumenta) | `thermostat.act(T_room, T_TARGET, CO2_room)` (3 argumenta) |
| `info["a"]` | `info["a1"]` i `info["a2"]` |

### 4.2. Dodano

- **`decompose()`** — rastavlja kaznu na temperaturu, CO₂ i energiju, pa se vidi *na šta* tačno odlazi nagrada. Ovo je bilo presudno za pronalaženje greške u okruženju.
- **Grafikon CO₂** sa granicom od 1000 ppm i prikazom broja ljudi u pozadini.
- **Grafikon akcija** prikazuje obje akcije, i za agenta i za termostat.
- `plots.py` sada **sam trenira** ako nema sačuvanog modela, a inače učitava `results/trained.npz`.
- Demo tabela prikazuje i broj ljudi i obje akcije.

### 4.3. Popravljene skale na grafikonima

- **Kriva učenja:** početni pad je toliko veliki da je ostatak izgledao kao ravna linija. Sada je y-osa zumirana na 2. percentil.
- **Boxplot poređenja:** netrenirana mreža je red veličine lošija i gutala je razliku između agenta i termostata. Sada je `symlog` skala, pa se vide sve tri.

---

## 5. [Č3] Više seedova — `multi_seed.py`

**Promijenjena svrha fajla.** Stara verzija je poredila ručni σ sa naučenim σ. Pošto je ručni σ izbačen, to poređenje više nema smisla.

Nova verzija trenira **isti model sa pet različitih seedova** i poredi ga sa termostatom. Odgovara na pitanje koje će profesor skoro sigurno postaviti: *da li je rezultat slučajnost jednog pokretanja?*

**Rezultat:** agent **−21.5 ± 1.3** (najgori −22.8, najbolji −19.2), termostat −99.6. Nije slučajnost.

---

## 6. [DOK] Dokumentacija

### 6.1. `specifikacija_mreze_i_okruzenja.md`

- Numeracija je počinjala od „2. Akcija 2", pa je **dodan opis akcije 1 (klima)**.
- Uklonjena netačna tvrdnja da je otvorenost ventilacije **ulaz** (to je akcija).
- Uklonjeno netačno opravdanje za 8 neurona („veći broj neurona bi se raspršio pod uticajem šuma"). Šum u REINFORCE-u dolazi od povrata, ne od broja težina — Karpathyjev Pong ima 200 skrivenih neurona. Zamijenjeno poštenim argumentom.
- Mreža je prepisana kao **4 – H – 4** umjesto 4 – 8 – 4, sa formulom **9H + 4** za broj parametara.
- Dodan odjeljak **5: preporuke za početne vrijednosti**.
- Dodan odjeljak **6: sažetak formula za gradijent** sa tri ispravke iz `Izvodi.pdf`.
- Dodan odjeljak **7: ugovor prema ostalim modulima** (tačni pozivi `act`, `mu`, `grad`, `update`, oblik batcha), da članovi 2 i 3 ne moraju usaglašavati potpise funkcija.
- Dodan odjeljak **8: izmjene u okruženju** sa obrazloženjem.
- Dodan odjeljak **9: rezultati** sa tabelom i tumačenjem.
- Izbačeno poređenje sa ručnim σ.
- Ciljna temperatura usklađena na **22 °C** svuda (ranije se u razgovoru koristilo 21 °C).

### 6.2. `README.md` — nov fajl

Kratak pregled: spisak fajlova sa oznakom člana, redoslijed pokretanja, šta se uči i tabela rezultata.

---

## 7. Rezultati prije i poslije

Mjereno na **istih 30 novih dana** (seedovi 1000–1029) koje mreža nije vidjela tokom treninga, deterministički (a = μ, bez šuma).

### Prije izmjena u okruženju

| Kontroler | Povrat | Σ(T−22)² | Vrijeme >1000 ppm | Σa₁² | Σa₂² |
| :--- | ---: | ---: | ---: | ---: | ---: |
| agent | −1825 | 1812 | ~43 % | 155 | 12 |
| termostat | −1901 | 1888 | 34 % | 240 | 40 |

Agent je jedva bolji, i to samo na potrošnji. Ventilaciju gotovo ne koristi.

### Poslije izmjena

| Kontroler | Povrat | Σ(T−22)² | Vrijeme >1000 ppm | Σa₁² | Σa₂² |
| :--- | ---: | ---: | ---: | ---: | ---: |
| netrenirana mreža | −11621 ± 7409 | — | — | — | — |
| **agent** | **−22.5 ± 25.0** | **15.2** | **9.2 %** | **85** | 34 |
| termostat | −99.6 ± 21.3 | 70.4 | 34.2 % | 240 | 40 |
| ručni P-kontroler (najbolji od 12) | ≈ −20 | — | 0 % | — | — |

Agent je bolji od termostata **po sve tri osi istovremeno**: temperatura je oko 4.6 puta tačnija, vrijeme iznad granice CO₂ je 3.7 puta kraće, a potrošnja klime oko 65 % manja.

**Zašto:** termostat radi po principu sve ili ništa (a₁ skače između −1 i +1), pa temperatura stalno osciluje. Agent je naučio **proporcionalno** upravljanje — drži a₁ tačno na vrijednosti koja poništava gubitak toplote za trenutnu spoljnu temperaturu, i ventilira blago ali stalno (a₂ oko 0.4) umjesto u naletima.

**Ručni P-kontroler** je dodan kao provjera gornje granice: agent je blizu najboljeg što se ručnim podešavanjem postiže, dakle nije samo „bolji od lošeg termostata".

**Naučena σ** počinje oko 0.30 i spontano opada na oko 0.15, bez ručnog rasporeda. To je dokaz da mreža sama smanjuje istraživanje kada nađe dobru politiku, i dobra je slika za rad.

---

## 8. Šta ostaje otvoreno

| Stavka | Član |
| :--- | :--- |
| Obrazložiti konstante okruženja u radu (`K`, `G`, `CO2_GEN`…) i navesti da su pojednostavljene | Č1 |
| Unijeti tri ispravke u `Izvodi.pdf` i uskladiti oznake σ/δ | Č2 |
| Uporediti H = 8 i H = 16 na više seedova, pa opravdati izbor rezultatom | Č2/Č3 |
| Eventualno proširenje: vlažnost, prozor, roletne (4 akcije, 7 ulaza) — tek ako ostane vremena | Č1 |
| Prezentacija i izvještaj o toku učenja | Č3 |

**Napomena o `environment.py`:** to je fajl člana 1, a ja sam u njemu promijenio tri konstante. Posebno mu skrenite pažnju na `G`, jer mijenja fiziku sobe i mora znati da to objasni.
