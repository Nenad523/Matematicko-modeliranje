
import numpy as np


class Policy:

    def __init__(self, seed=0, N=4):

        self.rng = np.random.default_rng(seed)
        self.N = N

        # Skriveni sloj: N čvorova, 2 ulaza
        self.A = self.rng.normal(0, 2, size=(N, 2))   # (N, 2)
        self.c = np.zeros(N)                              # (N,)

        # mi
        self.w = self.rng.normal(0, 2, size=N)         # (N,)
        self.b = 0.0

        # sigma
        self.v = self.rng.normal(0, 2, size=N)         # (N,)
        self.d = 0.0

    def hidden(self, x1, x2):
        s = self.A @ np.array([x1, x2]) + self.c
        return np.tanh(s)

    def mu(self, x1, x2):
        ''' Racunanje prvog izlaza mi. '''
        h = self.hidden(x1, x2)
        z = self.w @ h + self.b
        return np.tanh(z)

    def std(self, x1, x2):
        ''' Racunjanje drugog izlaza sigma - standardna devijacija. '''
        h = self.hidden(x1, x2)
        z = self.v @ h + self.d
        s = 1 / (1 + np.exp(-z))
        return 0.05 + 0.95 * s

    def act(self, x1, x2, learn_sigma=True, fixed_sigma=0.2):
        ''' Uzorkuje akcije iz Gausove raspodjele sa parametrima mu i sigma. '''

        m = self.mu(x1, x2)

        if learn_sigma:
            sigma = self.std(x1, x2)
        else:
            sigma = fixed_sigma

        eps = self.rng.standard_normal()
        u = m + sigma * eps

        return u, m, sigma

    def log_prob(self, x1, x2, u, sigma):
        ''' Racuna logaritam gustine Gausove raspodjele. '''

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
        N = self.N
        g_A = np.zeros((N, 2))
        g_c = np.zeros(N)
        g_w = np.zeros(N)
        g_b = 0.0
        g_v = np.zeros(N)
        g_d = 0.0

        for (x1, x2, u, m, sigma, G) in batch:
            # forward (ponovo, da dobijemo h)
            h = self.hidden(x1, x2)                    # (N,)
            z_mu = self.w @ h + self.b
            mu = np.tanh(z_mu)

            # μ grana
            delta_mu = (u - mu) / sigma**2 * G
            delta_zmu = delta_mu * (1 - mu**2)

            # σ grana
            delta_zsigma = 0.0
            if learn_sigma:
                delta_sigma = ((u - mu)**2 / sigma**3 - 1/sigma) * G
                s = (sigma - 0.05) / 0.95
                delta_zsigma = delta_sigma * 0.95 * s * (1 - s)

            # izlazne težine
            g_w += delta_zmu * h
            g_b += delta_zmu
            if learn_sigma:
                g_v += delta_zsigma * h
                g_d += delta_zsigma

            # skriveni sloj — SABIRANJE dvije grane
            delta_h = delta_zmu * self.w
            if learn_sigma:
                delta_h = delta_h + delta_zsigma * self.v

            delta_s = delta_h * (1 - h**2)              # (N,)

            g_A[:, 0] += delta_s * x1
            g_A[:, 1] += delta_s * x2
            g_c += delta_s

        n = len(batch)
        return (g_A/n, g_c/n, g_w/n, g_b/n, g_v/n, g_d/n)


    def update(self, batch, alpha, learn_sigma=True):
        '''
            Optimizacija parametara mreze kako bi se kretali u smjeru
            najbrzeg rasta funkcije cilja J.
        '''
        g_A, g_c, g_w, g_b, g_v, g_d = self.grad(batch, learn_sigma)

        self.A += alpha * g_A
        self.c += alpha * g_c
        self.w += alpha * g_w
        self.b += alpha * g_b

        if learn_sigma:
            self.v += alpha * g_v
            self.d += alpha * g_d

    def save(self, path):
        '''
            Cuvanje trenutnih parametara mreze u fajlu sa naznacenom putanjom.
        '''
        np.savez(path,
                 A=self.A, c=self.c,
                 w=self.w, b=self.b,
                 v=self.v, d=self.d,
                 N=self.N)

    def load(self, path):
        '''
            Ucitavanje parametara mreze iz fajla sa naznacenom putanjom.
        '''
        data = np.load(path)
        self.A = data["A"]
        self.c = data["c"]
        self.w = data["w"]
        self.b = float(data["b"])
        self.N = int(data["N"])
        if "v" in data:
            self.v = data["v"]
            self.d = float(data["d"])