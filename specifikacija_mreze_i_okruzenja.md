# TEHNIČKA SPECIFIKACIJA MIMO SISTEMA I NEURONSKE MREŽE (4 – 8 – 4)

## 1. Ulazni parametri (Vektor stanja $s = [x_1, x_2, x_3, x_4]$)

Prije ulaska u neuronsku mrežu, svi senzorski podaci se normalizuju (skaliraju) kako bi bili centrirani oko nule i malih vrijednosti. Time se osigurava numerička stabilnost i brža konvergencija tokom ručnog *backpropagation*-a.

* **$x_1$ — Odstupanje unutrašnje temperature od cilja:** `(T_soba - T_target) / 5.0`
  * **Konkretna upotreba:** Služi kao primarni senzor povratne sprege. Govori agentu da li je u prostoriji trenutno prevruće (pozitivan broj) ili prehladno (negativan broj) u odnosu na željenih 22.0°C.
* **$x_2$ — Termički pritisak okruženja:** `(T_spoljna - T_soba) / 10.0`
  * **Konkretna upotreba:** Mjeri razliku između spoljašnje i unutrašnje temperature. Prema Njutnovom zakonu hlađenja, ova razlika određuje brzinu kojom toplota prodire kroz zidove i ventilaciju. Omogućava agentu da reaguje proaktivno prije nego što se soba zagrije ili ohladi.
* **$x_3$ — Nivo zagađenja vazduha ($CO_2$):** `(CO2_soba - 400.0) / 500.0`
  * **Konkretna upotreba:** Mjeri koliko je vazduh u prostoriji ustajao u odnosu na idealan svjež vazduh (400 ppm). Kada ljudi borave u sobi, disanjem troše kiseonik i podižu $CO_2$, što mreži daje direktan signal da je potrebno provjetravanje.
* **$x_4$ — Dinamički broj prisutnih osoba:** `broj_ljudi / 10.0`
  * **Konkretna upotreba:** Normalizovan broj osoba u prostoriji (maksimalni kapacitet je 10 ljudi). Ovaj ulaz mreži javlja dvije ključne stvari: koliko dodatne toplote generišu ljudska tijela (oko 100W po osobi) i koliko će brzo rasti nivo $CO_2$, ali i koliko je stroga kazna za narušavanje komfora.

---

1. **Akcija 1 ($a_1 \in [-1.0, 1.0]$) — Snaga HVAC sistema (Klima uređaj):**
   * **Mehanizam:** Kontroliše termičku energiju u zatvorenom krugu (vrti i hladi/grije postojeći vazduh u sobi, ali ne dovodi svjež vazduh spolja).
   * **Opseg:** Negativne vrijednosti (`-1.0` do `0.0`) predstavljaju procenat snage hlađenja, pozitivne (`0.0` do `1.0`) procenat snage grijanja, dok `0.0` označava mirovanje.
    
2. **Akcija 2 ($a_2 \in [0.0, 1.0]$) — Otvorenost motorizovane ventilacione klapne:**
   * **Mehanizam:** Kontroliše metalna zakrilca (nalik venecijanerima) smještena unutar ventilacione cijevi, koja pokreće mali elektromotor (aktuator). Služi za izbacivanje ustajalog vazduha ($CO_2$) i ubacivanje svježeg vazduha spolja.
   * **Opseg i preslikavanje ugla:** Izlaz mreže (`0.0` do `1.0`) linearno se preslikava u fizički ugao zakretanja zakrilaca od **0° do 90°** ($\theta = a_2 \cdot 90^\circ$). Vrijednost `0.0` (0°) znači da su zakrilca spuštena i cijev je potpuno zatvorena, `0.5` (45°) da je otvorena dopola, a `1.0` (90°) da su zakrilca potpuno paralelna sa cijevi i vazduh slobodno struji.
   * **Inženjerski sukob:** Otvaranje klapne ($a_2$) pročišćava vazduh od $CO_2$, ali istovremeno unosi spoljni vreli ili ledeni vazduh, što narušava unutrašnju temperaturu. Agent mora naučiti da precizno dozira ugao klapne i uskladi ga sa radom klime ($a_1$) uz minimalan utrošak energije.


---

## 3. Izlazi neuronske mreže (4 izlaza: $\mu_1, \sigma_1, \mu_2, \sigma_2$)

Pošto radimo u dvodimenzionalnom kontinualnom prostoru akcija, stohastička politika modeluje svaku akciju kao nezavisnu Gausovu (Normalnu) raspodjelu. Zbog toga mreža na izlazu ima **4 neurona** (po dva parametra za svaku akciju):

| Izlazni neuron | Parametar | Aktivaciona funkcija | Opseg vrijednosti | Uloga u uzorkovanju akcije |
| :--- | :--- | :--- | :--- | :--- |
| **Neuron 1** | $\mu_1$ (Srednja vrijednost za klimu) | $\tanh(z_1)$ | $[-1.0, 1.0]$ | Određuje ciljanu snagu grijanja ili hlađenja. |
| **Neuron 2** | $\sigma_1$ (Devijacija za klimu) | $\text{Softplus}(z_2) = \ln(1 + e^{z_2})$ | $(0, +\infty)$ | Određuje nivo istraživanja (šuma) za rad klime u datom stanju. |
| **Neuron 3** | $\mu_2$ (Srednja vrijednost za ventilaciju) | $\text{Sigmoid}(z_3) = \frac{1}{1 + e^{-z_3}}$ | $[0.0, 1.0]$ | Određuje ciljani procenat otvorenosti ventilacije. |
| **Neuron 4** | $\sigma_2$ (Devijacija za ventilaciju) | $\text{Softplus}(z_4) = \ln(1 + e^{z_4})$ | $(0, +\infty)$ | Određuje nivo istraživanja (šuma) za rad ventilacije u datom stanju. |

*Napomena za aktivaciju devijacija ($\sigma_1, \sigma_2$):* Koristi se **Softplus** funkcija umjesto obične eksponencijalne ($e^z$) jer garantuje da je $\sigma$ uvijek strogo pozitivan broj (što je matematički uslov za Gausovu gustinu), a istovremeno sprječava eksplodiranje brojeva (*overflow*) kod velikih vrijednosti.

Konačne akcije se na svakom koraku uzorkuju prema formulama:
$$a_1 \sim \mathcal{N}(\mu_1, \sigma_1^2), \quad a_2 \sim \mathcal{N}(\mu_2, \sigma_2^2)$$

---

## 4. Arhitektura mreže (4 – 8 – 4) i inženjersko opravdanje

### Struktura slojeva i proračun parametara
Mreža je projektovana kao višeslojni perceptron (**MLP**) sa jednim skrivenim slojem od 8 neurona i ukupno **76 parametara** koji se uče putem ručnog *backpropagation*-a:

1. **Ulazni sloj (4 čvora):** Prima vektor stanja $[x_1, x_2, x_3, x_4]$.
2. **Skriveni sloj (8 neurona, $\tanh$ aktivacija):**
   * Matrica težina $W_1$: dimenzija $4 \times 8 = 32$ težine.
   * Vektor slobodnih članova $b_1$: $8$ parametara.
3. **Izlazni sloj (4 neurona):**
   * Matrica težina $W_2$: dimenzija $8 \times 4 = 32$ težine.
   * Vektor slobodnih članova $b_2$: $4$ parametra.
* **Ukupno parametara:** $32 + 8 + 32 + 4 = \mathbf{76\text{ parametara}}$.

### Zašto mreža izgleda baš ovako? (Ključni argumenti za odbranu)

1. **Zašto postoji skriveni sloj (a ne jednoslojna mreža)?**
   Uvođenjem 4 senzora i 2 suprotstavljene akcije, problem prestaje biti čisto linearan. Odluka o snazi klime sada zavisi od *kombinacije* više ulaza istovremeno (npr. proizvod otvorenosti ventilacije, spoljašnje temperature i broja ljudi). Jednoslojna mreža samo sabira ulaze i ne može da uhvati ovakve nelinearne interakcije. Skriveni sloj sa $\tanh$ aktivacijom omogućava mreži da kombinuje senzore i riješi sukob uređaja.
2. **Zašto baš 8 neurona (umjesto 64 ili 128)?**
   U čistom *Policy Gradient* (REINFORCE) algoritmu koji se piše od nule u NumPy-ju, gradijenti imaju visok statistički šum. Prevelik broj neurona (npr. 64) stvorio bi stotine parametara koji bi se raspršili pod uticajem šuma, usporavajući učenje i destabilizujući sistem. Osam neurona predstavlja idealan inženjerski kompromis – dovoljno kapaciteta za nelinearnost, a dovoljno malo parametara za brzu konvergenciju i stabilnost.
3. **Zašto se $\sigma_1$ i $\sigma_2$ uče kao izlazi mreže?**
   Učenje standardnih devijacija u zavisnosti od stanja omogućava **kontekstualnu eksploraciju**. Kada se sistem nađe u nepoznatom ili ekstremnom stanju (npr. nagli ulazak 10 ljudi u sobu), mreža može povećati $\sigma$ kako bi aktivnije istražila nova rješenja. Kada je prostorija stabilna na 22°C i 400 ppm, mreža samostalno smanjuje $\sigma$ blizu nule, pretvarajući agenta u precizan i pouzdan kontroler.
