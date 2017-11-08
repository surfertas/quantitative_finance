from pricer.black_scholes_pricer import *
from pricer.monte_carlo_pricer import *


def test_bs_option_price():
    # Sanity check against http://www.option-price.com/
    s = np.array([100])
    x = np.array([108])
    t, T, r, d, v = (252, 365, 0.05, 0.05, 0.20)
    Pricer = BSPricer(r, T)
    c = Pricer.price(s, x, v, d, t, "call")
    p = Pricer.price(s, x, v, d, t, "put")
    assert(np.around(c, decimals=5)[0] == 3.48795)
    assert(np.around(p, decimals=5)[0] == 11.21650)


def test_bs_greeks():
    s = np.array([100])
    x = np.array([108])
    t, T, r, d, v = (252, 365, 0.05, 0.05, 0.20)
    Pricer = BSPricer(r, T)
    c = Pricer.price(s, x, v, d, t, "call")
    assert(np.around(Pricer.delta, decimals=5) == 0.34002)
    assert(np.around(Pricer.gamma, decimals=5) == 0.02158)
    assert(np.around(Pricer.theta, decimals=5) == -0.01815)
    assert(np.around(Pricer.vega, decimals=5) == 0.29793)
    assert(np.around(Pricer.rho, decimals=5) == 0.21067)


def test_mc_pricer():
    # Test that pricer is producing correct # trials.
    n_trials = 1000
    MC = MCPricer(100, 0.05, "normal", T=365)
    MC.simulate(n_trials, 252, 0.20)
    assert(len(MC.episodes) == n_trials)
