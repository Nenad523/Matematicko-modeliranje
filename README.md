# Pametni termostat — REINFORCE sa Gausovom politikom

Sve implementirano od nule u NumPy-ju: mreža, backpropagation i optimizator.

## Fajlovi

| Fajl | Sadržaj | Član |
| :--- | :--- | :--- |
| `environment.py` | soba (temperatura, CO2, ljudi), on/off termostat za poređenje | 1 |
| `policy.py` | mreža 4–H–4, ručni backprop, Gausova politika | 2 |
| `check_gradient.py` | numerička provjera gradijenta (centralne razlike) | 2 |
| `train.py` | REINFORCE petlja: povrati, batch, gradijentni uspon | 3 |
| `plots.py` | evaluacija, demo tabela, šest grafikona | 3 |
| `multi_seed.py` | isti agent kroz pet seedova | 3 |
| `specifikacija_mreze_i_okruzenja.md` | specifikacija i nalazi | — |

## Redoslijed pokretanja

```bash
python check_gradient.py    # prvo: potvrdi da je backprop ispravan
python train.py             # trening (~100 s), čuva results/trained.npz
python plots.py             # evaluacija + results/grafikoni.png
python multi_seed.py        # pet seedova (~8 min)
```

`plots.py` sam trenira ako nema sačuvanog modela.

## Šta se uči

Stanje (4 broja): odstupanje temperature od cilja, razlika spolja i unutra, CO2, broj ljudi.
Akcije (2 broja): klima `a1 ∈ [-1, 1]`, ventilaciona klapna `a2 ∈ [0, 1]`.
Mreža daje `mu` i `sigma` za obje akcije, a akcija se uzorkuje iz Gausove raspodjele.
σ se **uči** (nije ručno zadan), uz donju granicu 0.05 da gradijent ne eksplodira zbog člana 1/σ².

## Rezultat

Mjereno na istih 30 novih dana, pod istom funkcijom nagrade:

| Kontroler | Povrat | Σ(T−22)² | Vrijeme >1000 ppm | Energija klime |
| :--- | ---: | ---: | ---: | ---: |
| netrenirana mreža | −11621 | — | — | — |
| **agent** | **−22.5** | **15.2** | **9.2 %** | **85** |
| on/off termostat | −99.6 | 70.4 | 34.2 % | 240 |

Agent je bolji po sve tri osi: tačnija temperatura, manje CO2 i manja potrošnja.
Razlog je što je naučio proporcionalno upravljanje, dok termostat radi
po principu sve ili ništa.

## Izmjene u okolini

Prva verzija nije mogla dobro raditi jer je cilj od 22 °C bio fizički nedostižan
na većini dana (klima preslaba). Izmijenjene su tri konstante u `environment.py`:
`G` 1.5 → 3.5, `SCALE` → `[1.5, 10, 700, 3]`, `W_CO2` 0.2 → 10.0.
Obrazloženje je u odjeljku 8 specifikacije.