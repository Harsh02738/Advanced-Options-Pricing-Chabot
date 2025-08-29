import numpy as np
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S, K, T, r, sigma):
        """
        S: Spot price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate
        sigma: Volatility
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        return self.S * norm.cdf(self.d1()) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2())

    def put_price(self):
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) - self.S * norm.cdf(-self.d1())

    # Greeks
    def delta(self, option_type="call"):
        if option_type == "call":
            return norm.cdf(self.d1())
        else:
            return -norm.cdf(-self.d1())

    def gamma(self):
        return norm.pdf(self.d1()) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        return self.S * norm.pdf(self.d1()) * np.sqrt(self.T)

    def theta(self, option_type="call"):
        first_term = - (self.S * norm.pdf(self.d1()) * self.sigma) / (2 * np.sqrt(self.T))
        if option_type == "call":
            second_term = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2())
            return first_term + second_term
        else:
            second_term = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2())
            return first_term + second_term

    def rho(self, option_type="call"):
        if option_type == "call":
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2())
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2())
