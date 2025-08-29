import numpy as np

class MonteCarloOption:
    def __init__(self, S, K, T, r, sigma, num_paths=100000, num_steps=100):
        """
        Monte Carlo Option Pricing
        
        S: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free interest rate
        sigma: Volatility
        num_paths: Number of simulated paths
        num_steps: Time steps per path
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.num_paths = num_paths
        self.num_steps = num_steps

    def simulate_paths(self):
        dt = self.T / self.num_steps
        paths = np.zeros((self.num_steps + 1, self.num_paths))
        paths[0] = self.S

        for t in range(1, self.num_steps + 1):
            z = np.random.standard_normal(self.num_paths)
            paths[t] = paths[t-1] * np.exp(
                (self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z
            )
        return paths

    def european_call_price(self):
        paths = self.simulate_paths()
        payoff = np.maximum(paths[-1] - self.K, 0)
        return np.exp(-self.r * self.T) * np.mean(payoff)

    def european_put_price(self):
        paths = self.simulate_paths()
        payoff = np.maximum(self.K - paths[-1], 0)
        return np.exp(-self.r * self.T) * np.mean(payoff)
