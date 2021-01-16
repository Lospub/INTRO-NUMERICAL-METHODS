from scipy import optimize 
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(10*x) - x

x = np.linspace(-2, 2, 1000)
sol = optimize.root(f, [-0.84, -0.71, -0.3, 0, 0.26, 0.7, 0.86])

plt.plot(x, f(x), lw=3)
plt.plot(sol.x, f(sol.x), 'o')
plt.axhline(0, color='gray', lw=0.5)
plt.show()