import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

T = 25


def dy_dt(C, t):
    return -C * 10 ** ((T - 25) / 5)


C0 = [0.6]
t = np.arange(25, 30, 0.01)
C = odeint(dy_dt, C0, t)
plt.figure()
plt.plot(t, C)
plt.show()
