# TEHNIČKA SPECIFIKACIJA OKRUŽENJA I NEURONSKE MREŽE (4 – H – 4)

> **Za člana 2 (mreža, `policy.py`): šta je izmijenjeno u odnosu na prethodnu verziju**
>
> 1. Skriveni sloj ima **H neurona** (parametar). Početna vrijednost je **H = 8**, ali kod mora raditi za bilo koje H. Ukupan broj parametara je **9H + 4** (za H = 8 to je 76).
> 2. **σ ima donju granicu:** `σ = σ_min + softplus(z)`, sa `σ_min = 0.05`. Izvod po z ostaje `sigmoid(z)`, pa se formule iz `Izvodi.pdf` ne mijenjaju.
> 3. Dodane su **preporuke za početne vrijednosti** težina (odjeljak 5), da trening krene stabilno.
> 4. Dopunjen je **opis akcije 1** (klima), koji je nedostajao, i uklonjena je netačna tvrdnja da je otvorenost ventilacije ulaz (to je akcija).
> 5. Opravdanje za veličinu mreže je prepisano (odjeljak 4). Nije tačno da veći broj neurona "raspršuje parametre pod uticajem šuma".
> 6. Dodan je **sažetak formula za gradijent** (odjeljak 6). Izvodi su provjereni u `Izvodi.pdf`, osim tri sitne ispravke navedene u odjeljku 6.
> 7. Ciljna temperatura u cijelom dokumentu je **22 °C** (kao u `environment.py`).
> 8. **Fiksni (ručno zadani) σ je izbačen.** σ se uvijek uči kao izlaz mreže, pa u kodu nema zastavice `learn_sigma`, ni parametara `sigma0`, `sigma_decay`, `fixed_sigma`.
> 9. Dodan je **ugovor prema ostalim modulima** (odjeljak 7), da članovi 2 i 3 ne moraju usaglašavati pozive.

---

## 1. Ulazni parametri (vektor stanja $s = [x_1, x_2, x_3, x_4]$)

Prije ulaska u mrežu, svi podaci se skaliraju da budu male vrijednosti oko nule. To sprječava zasićenje $\tanh$-a i stabilizuje ručni *backpropagation*.

* **$x_1$ — odstupanje temperature od cilja:** `(T_soba - T_cilj) / 5.0`
  * Glavni signal povratne sprege: pozitivno znači previše toplo, negativno previše hladno u odnosu na 22.0 °C.
* **$x_2$ — razlika spoljne i unutrašnje temperature:** `(T_spoljna - T_soba) / 10.0`
  * Prema Njutnovom zakonu hlađenja ova razlika određuje brzinu gubitka ili dobitka toplote kroz zidove i ventilaciju. Agent tako može djelovati unaprijed.
* **$x_3$ — nivo CO₂:** `(CO2_soba - 400.0) / 500.0`
  * Pokazuje koliko je vazduh ustajao u odnosu na svjež (400 ppm). Daje signal da je potrebno provjetravanje.
  * *Napomena:* bez ventilacije vrijednost može preći 3, pa je preporučeno ograničiti je na $[-3, 3]$ ako se primijeti zasićenje.
* **$x_4$ — broj prisutnih osoba:** `broj_ljudi / 10.0`
  * Ljudi generišu toplotu (oko 100 W po osobi) i CO₂, a od njih zavisi i koliko se komfor računa.
  * *Napomena:* u simulaciji je broj ljudi 0 do 3, pa je raspon ovog ulaza samo 0 do 0.3. Ako se pokaže da ima premalo uticaja, dogovoriti sa članom 1 skalu `/3`.

---

## 2. Akcije (2 akcije)

1. **Akcija 1 ($a_1 \in [-1, 1]$) — klima (HVAC):**
   * Vrijednost $-1$ znači puno hlađenje, $0$ isključeno, $+1$ puno grijanje. Okolina je povezuje s promjenom temperature preko koeficijenta $G$ (maksimalna snaga). Veća $|a_1|$ znači veću potrošnju energije, pa postoji kazna $-W_{HVAC}\,a_1^2$.

2. **Akcija 2 ($a_2 \in [0, 1]$) — otvorenost motorizovane ventilacione klapne:**
   * **Mehanizam:** metalna zakrilca u ventilacionoj cijevi pokreće mali aktuator. Služe za izbacivanje ustajalog vazduha (CO₂) i ubacivanje svježeg.
   * **Preslikavanje:** izlaz $[0, 1]$ linearno odgovara uglu $\theta = a_2 \cdot 90^\circ$ ($0$ = zatvoreno, $0.5$ = pola, $1$ = potpuno otvoreno).
   * **Inženjerski sukob:** otvaranje smanjuje CO₂, ali unosi spoljni topli ili hladni vazduh i remeti temperaturu. Agent mora uskladiti klapnu s radom klime uz minimalan utrošak energije.
   * *Napomena o vremenu:* korak simulacije je 6 minuta, pa $a_2$ znači prosječnu otvorenost u tom koraku. Kratko provjetravanje je jedan do dva koraka s visokim $a_2$.

**Granice akcija:** mreža daje neograničenu vrijednost iz Gausove raspodjele, a okolina je odsiječe (`clip`) na dozvoljeni opseg. U $\log\pi$ i u batchu se koristi **neodsječena** vrijednost $a$.

---

## 3. Izlazi neuronske mreže (4 izlaza: $\mu_1, \sigma_1, \mu_2, \sigma_2$)

Svaka akcija je nezavisna Gausova raspodjela, pa mreža ima **4 izlazna neurona**:

| Izlaz | Parametar | Aktivacija | Opseg | Uloga |
| :--- | :--- | :--- | :--- | :--- |
| 1 | $\mu_1$ (klima) | $\tanh(z_1)$ | $[-1, 1]$ | ciljana snaga grijanja ili hlađenja |
| 2 | $\sigma_1$ (klima) | $\sigma_{min} + \text{softplus}(z_2)$ | $[\sigma_{min}, \infty)$ | nivo istraživanja klime |
| 3 | $\mu_2$ (ventilacija) | $\text{sigmoid}(z_3)$ | $[0, 1]$ | ciljana otvorenost klapne |
| 4 | $\sigma_2$ (ventilacija) | $\sigma_{min} + \text{softplus}(z_4)$ | $[\sigma_{min}, \infty)$ | nivo istraživanja ventilacije |

$\text{softplus}(z) = \ln(1 + e^z)$, a $\sigma_{min} = 0.05$.

*Zašto softplus:* garantuje $\sigma > 0$ (uslov za Gausovu gustinu) bez eksplodiranja za velike $z$, a izvod mu je jednostavan: $\text{sigmoid}(z)$.

*Zašto donja granica $\sigma_{min}$:* u gradijentu se pojavljuje $1/\sigma^2$, pa bi za $\sigma \to 0$ korak postao ogroman i trening bi se raspao.

Akcije se uzorkuju kao $a_1 \sim \mathcal{N}(\mu_1, \sigma_1^2)$ i $a_2 \sim \mathcal{N}(\mu_2, \sigma_2^2)$.

---

## 4. Arhitektura mreže (4 – H – 4)

### Struktura i broj parametara

Višeslojni perceptron s jednim skrivenim slojem od **H neurona** ($\tanh$):

1. **Ulaz (4):** $[x_1, x_2, x_3, x_4]$.
2. **Skriveni sloj (H):** $W_1$ je $4 \times H$, $b_1$ je $1 \times H$.
3. **Izlazni sloj (4):** $W_2$ je $H \times 4$, $b_2$ je $1 \times 4$.

**Ukupno parametara: $4H + H + 4H + 4 = 9H + 4$** (za $H = 8$: 76).

**Zahtjev za implementaciju:** $H$ je parametar konstruktora (`Policy(H=8)`). Dimenzije ulaza (4) i izlaza (4) takođe ne smiju biti upisane na više mjesta. Promjena H ne smije tražiti izmjenu formula, samo dimenzija.

### Opravdanje (za odbranu)

1. **Zašto skriveni sloj?** Odluka o snazi klime zavisi od *kombinacije* ulaza (npr. hladno u sobi s puno ljudi drukčije je nego hladno u praznoj sobi). Mreža bez skrivenog sloja kombinuje ulaze samo linearno.
2. **Zašto H = 8?** Problem je mali (4 ulaza, 2 akcije), pa je 8 neurona dovoljno da uhvati nelinearne veze, a trening ostaje brz, a ručni backprop pregledan. Veća mreža se provjerava poređenjem H = 8 i H = 16 na više seedova, i uzima se ona koju rezultati opravdaju.
3. **Zašto se $\sigma$ uči kao izlaz mreže?** Omogućava *kontekstualnu eksploraciju*: u nepoznatom stanju (npr. naglo ulazi mnogo ljudi) mreža može povećati $\sigma$ i više istraživati, a u mirnom stanju smanjiti $\sigma$. Alternativa je ručno zadan raspored po kojem $\sigma$ opada kroz vrijeme, ali tada agent ne može prilagoditi istraživanje stanju. U ovom projektu se $\sigma$ **uvijek uči**; ručni raspored se pominje samo kao moguće proširenje.

---

## 5. Preporuke za početne vrijednosti

* **Težine skrivenog sloja $W_1$:** male slučajne vrijednosti, npr. $\mathcal{N}(0, 0.5^2)$, $b_1 = 0$.
* **Težine izlaznog sloja $W_2$:** male, npr. $\mathcal{N}(0, 0.1^2)$, tako da mreža krene s $\mu_1 \approx 0$ (klima isključena) i $\mu_2 \approx 0.5$ (klapna napola otvorena).
* **Bias izlaza za $\sigma$:** postaviti tako da početni $\sigma \approx 0.3$, tj. $b_2[\sigma] = \ln(e^{0.25} - 1) \approx -1.26$. Bez toga je početni $\sigma$ oko 0.74, što je mnogo za akciju u opsegu $[0, 1]$.
* **Seed:** koristiti `np.random.default_rng(seed)` radi ponovljivosti.

---

## 6. Sažetak formula za gradijent (provjeriti prema `Izvodi.pdf`)

Za jedan potez $t$ sa povratom $G_t$ i akcijom $a_i$ (neodsječenom):

$$\frac{\partial \log\pi}{\partial \mu_i} = \frac{a_i - \mu_i}{\sigma_i^2},\qquad
\frac{\partial \log\pi}{\partial \sigma_i} = \frac{(a_i - \mu_i)^2 - \sigma_i^2}{\sigma_i^3}$$

Greške izlaznog sloja $\Delta = [\Delta_1, \Delta_2, \Delta_3, \Delta_4]$:

* $\Delta_1 = \dfrac{a_1 - \mu_1}{\sigma_1^2}\,(1 - \mu_1^2)$  (tanh)
* $\Delta_2 = \dfrac{(a_1 - \mu_1)^2 - \sigma_1^2}{\sigma_1^3}\,\text{sigmoid}(z_2)$  (softplus)
* $\Delta_3 = \dfrac{a_2 - \mu_2}{\sigma_2^2}\,\mu_2(1 - \mu_2)$  (sigmoid)
* $\Delta_4 = \dfrac{(a_2 - \mu_2)^2 - \sigma_2^2}{\sigma_2^3}\,\text{sigmoid}(z_4)$  (softplus)

Zatim: $\delta^{(2)} = \Delta \cdot G_t$, $\;\delta^{(1)} = (\delta^{(2)} W_2^T) \odot (1 - h^2)$, $\;\partial J/\partial W_2 = h^T \delta^{(2)}$, $\;\partial J/\partial b_2 = \delta^{(2)}$, $\;\partial J/\partial W_1 = x^T \delta^{(1)}$, $\;\partial J/\partial b_1 = \delta^{(1)}$.

**Ispravke u rukom pisanom izvodu:**
1. `d tanh(z)/dz = 1 − tanh²(z)` (u međukoraku nedostaje kvadrat).
2. U razvoju za $z^{(1)}$ težine su $W^{(1)}$, ne $W^{(2)}$.
3. U $\Delta_4$ treba $\text{sigmoid}(z_4)$ (provjeriti rukopis).

**Povrat i korak učenja:** $G_t$ se računa unazad po epizodi (suma budućih nagrada), normalizuje se preko cijelog batcha (oduzme se srednja vrijednost, podijeli standardnom devijacijom), a ažuriranje je gradijentni uspon: $\theta \leftarrow \theta + \alpha\,\nabla J$.

**Obavezna provjera:** numerička provjera gradijenta (centralne razlike) za $H = 8$ i barem još jedno $H$, prije nego što se pokrene trening (`check_gradient.py`).

---

## 7. Ugovor prema ostalim modulima

Da članovi 2 (mreža) i 3 (trening, grafikoni) ne moraju usaglašavati pozive:

```python
Policy(seed=0, H=8)          # konstruktor
act(s)      -> (u, m, sigma) # s je vektor (4,); sva tri rezultata su vektori (2,)
mu(s)       -> m             # deterministička akcija, za evaluaciju
update(batch, alpha, clip_norm=None)
grad(batch) -> [gW1, gb1, gW2, gb2]
log_prob_batch(S, U) -> (n,) # za numeričku provjeru gradijenta
params()    -> [W1, b1, W2, b2]
save(path) / load(path)
```

`batch` je lista petorki `(s, u, m, sigma, G)`, gdje je `u` **neodsječena** akcija, a `G` već normalizovan povrat.

Okolina (`environment.Room`) vraća `info` sa ključevima `T_soba`, `T_spolja`, `CO2_soba`, `broj_ljudi`, `t`, `T_cilj`, `a1`, `a2`.

---

## 8. Izmjene u okolini (`environment.py`) i zašto

Prva verzija okoline imala je grešku zbog koje agent nije mogao dobro naučiti ni temperaturu ni ventilaciju. Tri konstante su izmijenjene:

| Konstanta | Bilo | Sada | Razlog |
| :--- | ---: | ---: | :--- |
| `G` (snaga klime) | 1.50 | **3.50** | **Cilj nije bio dostižan.** Ravnoteža pri punoj klimi je $T_{spolja} + G/K$, što je sa starim vrijednostima $T_{spolja} + 15$. Pošto `reset()` bira dane u kojima spoljna temperatura pada do $-3.4\,^\circ$C, cilj od 22 °C bio je fizički nedostižan dijelom dana na **17 od 20 dana**. Ta neizbježna kazna gušila je sve ostalo. Sada je $G/K = 35$. |
| `SCALE` | `[5, 10, 500, 10]` | **`[1.5, 10, 700, 3]`** | Ulazi $x_1$ i $x_4$ koristili su samo mali dio opsega (broj ljudi 0–3 dijeljen sa 10 daje najviše 0.3), pa su za mrežu bili skoro nevidljivi. Nove skale drže sve ulaze u opsegu u kojem $\tanh$ nije ravan. |
| `W_CO2` | 0.20 | **10.0** | Kazna za CO₂ bila je oko 100 puta slabija od kazne za temperaturu, pa je agent potpuno ignorisao ventilaciju i druga akcija je bila ukrasna. Sada 500 ppm preko granice vrijedi otprilike koliko 3.2 °C odstupanja. |

Provjereno je da bez prve izmjene nijedna druga ne pomaže: pri `G = 1.50` povećanje `W_CO2` na 5 ili 50 samo pogoršava rezultat, jer agent nema načina da istovremeno drži temperaturu.

---

## 9. Rezultati (1500 iteracija, batch 7 dana, H = 8)

Svi kontroleri su mjereni na **istih 30 novih dana** koje mreža nije vidjela tokom treninga, i pod **istom** funkcijom nagrade.

| Kontroler | Povrat | Σ(T−22)² | Vrijeme iznad 1000 ppm | Σa₁² (energija klime) | Σa₂² |
| :--- | ---: | ---: | ---: | ---: | ---: |
| netrenirana mreža | −11621 ± 7409 | — | — | — | — |
| **agent (REINFORCE)** | **−22.5 ± 25.0** | **15.2** | **9.2 %** | **85** | 34 |
| on/off termostat | −99.6 ± 21.3 | 70.4 | 34.2 % | 240 | 40 |
| ručni P-kontroler (najbolji od 12) | ≈ −20 | — | 0 % | — | — |

Agent je bolji od termostata **po sve tri osi istovremeno**: drži temperaturu oko 4.6 puta tačnije, provede 3.7 puta manje vremena iznad granice CO₂, i pritom troši oko 65 % manje energije na klimu.

**Zašto je bolji:** termostat radi po principu sve ili ništa (a₁ skače između −1 i +1, pa temperatura stalno osciluje oko cilja). Agent je naučio **proporcionalno** upravljanje: drži a₁ na vrijednosti koja tačno poništava gubitak toplote za trenutnu spoljnu temperaturu, i ventilira blago ali stalno (a₂ oko 0.4) umjesto u naletima. Rezultat je ravna linija na 22.0 °C i CO₂ koji rijetko prelazi granicu.

**Stabilnost:** `multi_seed.py` trenira isti model sa pet različitih seedova. Povrat je **−21.5 ± 1.3** (najgori −22.8, najbolji −19.2), dakle rezultat nije slučajnost jednog pokretanja. Za poređenje, termostat je −99.6.

**Naučena σ:** počinje oko 0.30 i spontano opada na oko 0.15 do kraja treninga, bez ručnog rasporeda. To je potvrda da mreža sama smanjuje istraživanje kada nađe dobru politiku.