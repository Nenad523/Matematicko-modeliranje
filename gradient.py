import numpy as np

def check(p, x1, x2, a, sigma, h=1e-5):

        '''
            Provjerava da li nas gradijent koji smo rucno izveli je ispravan.
        '''

        m = p.mu(x1, x2)
        analiticki = np.array(p.grad([(x1, x2, a, m, 1.0)], sigma))

        numericki = []
        for ime in ["w1", "w2", "b"]:
            staro = getattr(p, ime)

            setattr(p, ime, staro + h)
            f_plus = p.log_prob(x1, x2, a, sigma)

            setattr(p, ime, staro - h)
            f_minus = p.log_prob(x1, x2, a, sigma)

            setattr(p, ime, staro)          # vrati težinu na staru vrijednost

            numericki.append((f_plus - f_minus) / (2 * h))

        return analiticki, np.array(numericki)