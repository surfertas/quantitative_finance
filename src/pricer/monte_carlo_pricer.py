#!/usr/env/bin python
# author: Tasuku Miura
# date: 11/8/2017

import numpy as np


class MCPricer(object):
    # figure out way to get better estimate for r

    def __init__(self, S, r, distribution, T=365):
        self.S = S
        self.r = r
        self.distribution = distribution
        self.T = np.float(T)
        print("Initialized MC Pricer")

        self.df = None
        self.episodes = None

    def price(self, X, op_type="call"):
        assert(self.episodes != None)
        assert(self.df != None)
        n, t = self.episodes.shape
        if op_type == "call":
            return (np.sum((self.episodes[:, -1] - X)[np.where((self.episodes[:, -1] - X) > 0)]) / n) * self.df
        else:
            return (X - np.sum((self.episodes[:, -1])[np.where((X - self.episodes[:, -1]) > 0)]) / n) * self.df

    def simulate(self, n_samples, t, v):
        self.df = np.exp(-self.r * t / self.T)
        self.episodes = (
            self.S * (np.cumprod(
                np.exp((self.r - 0.5 * v**2) * (1 / self.T) + v * np.sqrt(1 / self.T) * np.random.randn(n_samples, t)),
                axis=1)
            )
        )

    def stats(self):
        mu = np.mean(self.episodes[:, -1])
        sq_err = lambda x: (x - mu)**2
        std = np.sqrt(np.mean(map(sq_err, self.episodes[:, -1])))
        mx = np.max(self.episodes[:, -1])
        mn = np.min(self.episodes[:, -1])
        print("Mean: {}, Std: {}, Min: {}, Max: {}".format(mu, std, mn, mx))
        return (mu, std, mn, mx)

    def reset_state(self):
        self.episodes = None

    def plot(self):
        plt.figure(1)
        plt.subplot(221)
        [plt.plot(episode) for episode in self.episodes]
        plt.subplot(222)
        sns.kdeplot(self.episodes[:, -1])
