import matplotlib.pyplot as plt
import numpy as np


x = [1,3,4,6]
y = [158000,141000,121000,109000]

#x = np.log(x) 
print(x)
y = np.log(y)
print(y)

A = np.vstack([x, np.ones(len(x))]).T

n, k_ = np.linalg.lstsq(A, y, rcond = None)[0]

_ = plt.plot(x, y, 'o', label='Original data', markersize=10)


_ = plt.plot(x, n*x + k_, 'r', label='Fitted line')

_ = plt.legend()

plt.show()