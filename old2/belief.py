from basics import Distribution

class Gaussian(Distribution):

    def __init__(self, space, mu, sigma):
        self._space = space
        self._mu = mu
        self._sigma = sigma

    
