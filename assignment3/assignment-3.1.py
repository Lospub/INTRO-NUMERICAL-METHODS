from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(10*x) - x

def derf(x):
    return 10*np.cos(10*x) - 1

interval_left = -1.75
interval_right = 1.75


xvals = np.linspace(interval_left, interval_right, num=100)
yvals = f(xvals)

pointlist = xvals
plt.figure()

plt.plot(xvals,yvals)
#plt.hlines(0, interval_left, interval_right)

plt.xlabel("$x$", fontsize=16)
plt.ylabel("$f(x)$", fontsize=16)

plt.xlim((interval_left, interval_right))
plt.ylim((-5,5))

# loop over all starting points defined in pointlist
for x0 in pointlist:
    # set desired precision and max number of iterations
    prec_goal = 1.e-10
    nmax = 1000
    # initialize values for iteration loop
    reldiff = 1
    xi = x0
    counter = 0
    # start iteration loop
    while reldiff > prec_goal and counter < nmax:
        # get the number of necessary iterations at that particular x0
        # compute relative difference
        reldiff = np.abs(f(xi)/derf(xi)/xi)
        # compute next xi
        x1 = xi - f(xi)/derf(xi)
        # trade output to input for next iteration step
        xi = x1
        # increase counter
        counter += 1
    # plot the number of necessary iterations at that particular x0
    if f(x0) == 0:
        plt.plot(x0, f(x0),'r.', markersize=5)
    # print test output
    #print(x0,counter,reldiff)
    
# save the figure to an output file
#plt.savefig('newton-method-demo-figure.jpg', bbox_inches='tight')
plt.grid()
plt.show()

# close plot for suppressing the figure output at runtime. Figure is still saved to file
#plt.close()        
