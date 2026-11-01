import numpy as np

class Policy:

    def __init__(self, seed=0):

        self.rng = np.random.default_rng(seed)
        self.w1 = self.rng.normal(0, 0.1)
        self.w2 = self.rng.normal(0, 0.1)
        self.b = 0.0

    def mu(self, x1, x2):
        '''
            Ovo predstavlja forward pass u nasoj mrezi.
        '''

        z = self.w1 * x1 + self.w2 * x2 + self.b
        return np.tanh(z)


    def act(self, x1, x2, sigma):
        '''
            Uzorkuje akcije iz Gausove raspodjele sa parametrima mu i sigma.
        '''

        m = self.mu(x1, x2)
        eps = self.rng.standard_normal()
        a = m + sigma * eps
        return a, m


    def log_prob(self, x1, x2, a, sigma):
        '''
        Opisuje koliko je ova akcija “očekivana” za trenutnu politiku. Uzima stanje i akciju a, računa μ, pa vraća logaritam visine Gausovog zvona u tački a.
        '''

        m = self.mu(x1, x2)
        return -(a - m)**2 / (2 * sigma**2) - np.log(sigma) - (1 / 2) * np.log(2 * np.pi)


    def grad(self, batch, sigma):
        '''
            Racuna gradijent funkcije cilja J po parametrima mreze w1, w2 i b.
        '''

        g1 = 0.0
        g2 = 0.0
        gb = 0.0

        for (x1, x2, a, m, G) in batch:
            
            delta = (a - m) / sigma**2 * (1 - m**2) * G
            g1 += delta * x1
            g2 += delta * x2
            gb += delta

        N = len(batch)
        return g1 / N, g2 / N, gb / N


    def update(self, batch, sigma, alpha):
        '''
            Optimizacija parametara mreze kako bi se kretali u smjeru najbrzeg rasta funckije cilja J.
        '''

        g_w1, g_w2, g_b = self.grad(batch, sigma)
        
        self.w1 = self.w1 + alpha * g_w1
        self.w2 = self.w2 + alpha * g_w2
        self.b = self.b  + alpha * g_b


    def save(self, path):
        '''
            Cuvanje trenuthin parametara mreze u fajlu sa naznacenom putanjom.
        '''

        np.savez(path, w1 = self.w1, w2 = self.w2, b = self.b)

    def load(self, path):
        '''
            Ucitavanja parametara mreze iz fajla sa naznacenom putanjom.
        '''

        data = np.load(path)
        self.w1 = float(data["w1"])
        self.w2 = float(data["w2"])
        self.b = float(data["b"])


