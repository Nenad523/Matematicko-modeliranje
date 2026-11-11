
import numpy as np


class Policy:

    def __init__(self, seed=0):

        self.rng = np.random.default_rng(seed)

        # Parametri za mi
        self.w1 = self.rng.normal(0, 0.1)
        self.w2 = self.rng.normal(0, 0.1)
        self.b = 0.0

        # Parametri za sigma
        self.v1 = self.rng.normal(0, 0.1)
        self.v2 = self.rng.normal(0, 0.1)
        self.c = 0.0

    def mu(self, x1, x2):
        '''
            Racunanje prvog izlaza mi.
        '''

        z = self.w1 * x1 + self.w2 * x2 + self.b
        return np.tanh(z)

    def std(self, x1, x2):
        '''
            Racunjanje drugog izlaza sigma - standardna devijacija.
        '''

        z = self.v1 * x1 + self.v2 * x2 + self.c
        s = 1 / (1 + np.exp(-z))

        return 0.05 + 0.95 * s

    def act(self, x1, x2, learn_sigma=True, fixed_sigma=0.2):
        '''
            Uzorkuje akcije iz Gausove raspodjele sa parametrima mu i sigma.
        '''

        m = self.mu(x1, x2)

        if learn_sigma:
            sigma = self.std(x1, x2)
        else:
            sigma = fixed_sigma

        eps = self.rng.standard_normal()
        u = m + sigma * eps
        a = np.tanh(u)

        return a, u, m, sigma

    def log_prob(self, x1, x2, u, sigma):
        '''
            Racuna logaritam gustine Gausove raspodjele.
        '''

        m = self.mu(x1, x2)

        return (
            -(u - m)**2 / (2 * sigma**2)
            - np.log(sigma)
            - 0.5 * np.log(2 * np.pi)
        )

    def grad(self, batch, learn_sigma=True):
        '''
            Racuna gradijente funkcije cilja J po parametrima mreze.
            Ako je learn_sigma=True, racuna gradijente svih sest parametara.
            Ako je learn_sigma=False, racuna samo gradijente parametara mi.
        '''

        g1 = 0.0
        g2 = 0.0
        gb = 0.0

        gv1 = 0.0
        gv2 = 0.0
        gc = 0.0

        for (x1, x2, u, m, sigma, G) in batch:

            # Gradijenti za mi
            delta_mu = (
                (u - m) / sigma**2
            ) * (1 - m**2) * G

            g1 += delta_mu * x1
            g2 += delta_mu * x2
            gb += delta_mu

            # Gradijenti za sigma
            if learn_sigma:

                s = (sigma - 0.05) / 0.95

                delta_sigma = (
                    (u - m)**2 / sigma**3 - 1 / sigma
                ) * 0.95 * s * (1 - s) * G

                gv1 += delta_sigma * x1
                gv2 += delta_sigma * x2
                gc += delta_sigma

        N = len(batch)

        return (
            g1 / N, g2 / N, gb / N,
            gv1 / N, gv2 / N, gc / N
        )

    def update(self, batch, alpha, learn_sigma=True):
        '''
            Optimizacija parametara mreze kako bi se kretali u smjeru
            najbrzeg rasta funkcije cilja J.
        '''

        g_w1, g_w2, g_b, g_v1, g_v2, g_c = self.grad(
            batch, learn_sigma
        )

        self.w1 += alpha * g_w1
        self.w2 += alpha * g_w2
        self.b += alpha * g_b

        if learn_sigma:
            self.v1 += alpha * g_v1
            self.v2 += alpha * g_v2
            self.c += alpha * g_c

    def save(self, path):
        '''
            Cuvanje trenutnih parametara mreze u fajlu sa naznacenom putanjom.
        '''

        np.savez(
            path,
            w1=self.w1,
            w2=self.w2,
            b=self.b,
            v1=self.v1,
            v2=self.v2,
            c=self.c
        )

    def load(self, path):
        '''
            Ucitavanje parametara mreze iz fajla sa naznacenom putanjom.
        '''

        data = np.load(path)

        self.w1 = float(data["w1"])
        self.w2 = float(data["w2"])
        self.b = float(data["b"])

        if "v1" in data:
            self.v1 = float(data["v1"])
            self.v2 = float(data["v2"])
            self.c = float(data["c"])
