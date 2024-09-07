import numpy as np
import matplotlib.pyplot as plt
t = range(1,120,1)
y = -np.log(0.5) / t - 0.4621 * 10 ** ((30 - 25) / 5)
plt.figure()
plt.plot(t,y)
plt.show()