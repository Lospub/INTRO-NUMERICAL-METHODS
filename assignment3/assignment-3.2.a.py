from scipy import optimize 
import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    x1 = 81 + x
    x2 = 1 + x
    return x1 / x2

#x = np.linspace(-2, 2, 1000)
#x = range(70, f1(70))
#sol = optimize.root(f, [-0.84, -0.71, -0.3, 0, 0.26, 0.7, 0.86])

def f2(x):
    return x - np.sqrt(81)
x = float(70)
xvals = []
yvals = []
count = 0
while count <= 30:
    xvals.append(x)
    yvals.append(f2(x))
    x = f1(x)
    count += 1
    
xvals = np.array(range(0,count), dtype = float)
yvals = np.array(yvals, dtype = float)
    
#A = np.vstack([xvals, np.ones(len(xvals))]).T

plt.plot(xvals, yvals, 'r')
plt.ylabel("error")
plt.xlabel("iteration times")

#plt.plot(xvals, yvals, 'o')
plt.axhline(0, color='gray', lw=0.5)
plt.show()

#n, k_ = np.linalg.lstsq(A, yvals, rcond = None)[0]

#_ = plt.plot(xvals, yvals, 'o', label='Original data', markersize=10)


#_ = plt.plot(xvals, n*xvals + k_, 'r', label='Fitted line')

#_ = plt.legend()

plt.show()