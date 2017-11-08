#!usr/bin/env python

# author: Tasuku Miura
# date: 11/8/2017

# Quickly put together a black scholes pricer for experimentation.
# TODO: Improve code design as this was scrapped together in a rush.

import numpy as np
import scipy
from scipy.stats import norm


class BSPricer(object):

    def __init__(self, rf, T):
        self.N = lambda x: norm.cdf(x)
        self.d1 = None
        self.d2 = None
        self.r = rf           # risk free rate
        self.T = float(T)     # calendar year
        self.tt = None        # time to expiry (ratio t/T)

        # Input parameters
        self.S = None
        self.X = None
        self.v = None
        self.d = None
        self.t = None
        self.op_type = None

        # Greeks
        self.greeks = False
        self.delta = None
        self.gamma = None
        self.theta = None
        self.vega = None
        self.rho = None

    def __set_params(self, S, X, v, d, t, op_type):
        self.S = S.astype(np.float)
        self.X = X.astype(np.float)
        self.v = float(v)
        self.d = float(d)
        self.t = float(t)
        self.op_type = op_type

    def __set_d_values(self, d1, d2):
        self.d1 = d1
        self.d2 = d2

    def __generate_greeks(self):
        self.delta = (np.exp(-self.d * self.tt) * self.N(self.d1)
                      if self.op_type == "call"
                      else np.exp(-self.d * self.tt) * (self.N(self.d1) - 1))

        self.gamma = (np.exp(-self.d * self.tt) / (self.S * self.v * np.sqrt(self.tt))
                      * 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * self.d1**2))

        self.theta = (
            1 / self.T * (
                -(self.S * self.v * np.exp(-self.d * self.tt) / 2 * np.sqrt(self.tt)
                    * 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * self.d1**2)
                  )
                - self.r * self.X * np.exp(-self.r * (self.tt) * self.N(self.d2))
                + self.d * self.S * np.exp(-self.d * self.tt) * self.N(self.d1))
            if self.op_type == "call"
            else (
                1 / self.T * (
                    -(self.S * self.v * np.exp(-self.d * (self.tt)) / 2 * np.sqrt(self.tt)
                      * 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * self.d1**2)
                      )
                    + self.r * self.X * np.exp(-self.r * (self.tt) * self.N(-self.d2))
                    - self.d * self.S * np.exp(-self.d * (self.tt)) * self.N(-self.d1))))

        self.vega = (0.01 * self.S * np.exp(-self.d * (self.tt)) * np.sqrt(self.tt)
                     * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * self.d1**2))

        self.rho = (0.01 * self.X * (self.tt) * np.exp(-self.r * (self.tt)) * self.N(self.d2)
                    if self.op_type == "call"
                    else -0.01 * self.X * (self.tt) * np.exp(-self.r * (self.tt)) * self.N(-self.d2))

    def price(self, S, X, v, d, t, op_type, greeks=True):
        """
        :arg S - underlying stock price(s) type: np.array
        :arg X - strike price (s) type: np.array
        :arg v - volatility (decimal)
        :arg d - continuously compounded dividend yield (decimal)
        :arg t - time to expiration (decimal)
        :arg op_type - call or put
        :ret option price
        """
        assert(op_type == "call" or op_type == "put")
        assert(len(S.shape) == 1 and len(X.shape) == 1)  # Make sure dimensions (-1,)
        self.__set_params(S, X, v, d, t, op_type)

        self.tt = self.t / self.T
        ssqrt = self.v * np.sqrt(self.tt)
        d1 = (np.log(self.S / self.X) + self.tt * (self.r - self.d + 0.5 * self.v**2)) / ssqrt
        d2 = d1 - ssqrt

        if op_type == "call":
            price = (self.S * np.exp(-self.d * (self.tt)) * self.N(d1)
                     - self.X * np.exp(-self.r * (self.tt)) * self.N(d2))
        else:
            price = (self.X * np.exp(-self.r * (self.tt)) * self.N(-d2)
                     - self.S * np.exp(-self.d * (self.tt)) * self.N(-d1))

        self.__set_d_values(d1, d2)
        if greeks:
            self.greeks = greeks
            self.__generate_greeks()
        return price

    def payout(self, S, X, op_type):
        """
        :arg S - underlying stock price(s) type: np.array
        :arg X - strike price (s) type: np.array
        :arg op_type - call or put
        :ret option value type: list
        """
        if op_type == "call":
            return map(lambda s_x: np.max([s_x[0] - s_x[1], 0]), zip(S, X))
        else:
            return map(lambda s_x1: np.max([s_x1[1] - s_x1[0], 0]), zip(S, X))

    def print_greeks(self):
        if self.greeks:
            print("Delta: ", self.delta)
            print("Gamma: ", self.gamma)
            print("Theta: ", self.theta)
            print("Vega: ", self.vega)
            print("Rho: ", self.rho)
        else:
            print("Greeks were not set...")
