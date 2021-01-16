import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import math
import random

x = [1.02,0.95,0.87,0.77,0.67,0.56,0.44,0.3,0.16,0.01]
y = [0.39,0.32,0.27,0.22,0.18,0.15,0.13,0.12,0.13,0.15]

for i in range(len(x)):
    x[i] = x[i] + random.uniform(-0.005, 0.005)
    y[i] = y[i] + random.uniform(-0.005, 0.005)

a1 =  np.power(y, 2)
b1 = np.multiply(x, y)
x1 = np.power(x, 2)

A = np.vstack([a1, b1, x, y, np.ones(len(x))]).T

a, b, c, d, e = np.linalg.lstsq(A, x1, rcond = None)[0]

def computeEllipse(a, b, c, d, e):

    #Returns x-y arrays for ellipse coordinates.
    #Equation is of the form a*y**2 + b*x*y + c*x + d*y + e = x**2

    # Convert x**2 coordinate to +1
    a = -a
    b = -b
    c = -c
    d = -d
    e = -e
    # Rotation angle
    theta = 0.5 * math.atan(b / (1 - a))
    # Rotated equation
    sin = math.sin(theta)
    cos = math.cos(theta)
    aa = cos**2 + b * sin * cos + a * sin**2
    bb = sin**2 - b * cos * sin + a * cos**2
    cc = c * cos + d * sin
    dd = -c * sin + d * cos
    ee = e
    # Standard Form
    axMaj = 1 / math.sqrt(aa)
    axMin = 1 / math.sqrt(bb)
    scale = math.sqrt(cc**2 / (4 * aa) + dd**2 / (4 * bb) - ee)
    h = -cc / (2 * aa)
    k = -dd / (2 * bb)
    # Parametrized Equation
    t = np.linspace(0, 2 * math.pi, 1000)
    xx = h + axMaj * scale * np.sin(t)
    yy = k + axMin * scale * np.cos(t)
    # Un-rotated coordinates
    x = xx * cos - yy * sin
    y = xx * sin + yy * cos

    return x, y

lines = plt.plot(*computeEllipse(a, b, c, d, e))
ax = lines[0].axes
ax.plot(x, y, 'r.')

"""
y2 = []
y_ = []
xy = []
x_ = []
x2 = []
for i in range(len(x)):
    xy.append(b*x[i]*y[i])
    y2.append(a*(y[i]**2))
    y_.append(d*y[i])
    x_.append(c*x[i])
    x2.append(x[i]**2)

z = []
for i in range(len(x)):
    z.append(y2[i] + xy[i] + x_[i] + y_[i] + e - x2[i])
    
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.contour3D(x, y, z, c= z,  cmap='binary')
"""
"""
def get_transforms(A, B, C, D, E, F, G): 
    #Get transformation matrix and shift for a 3d ellipsoid 

    #Assume A*x**2 + C*y**2 + D*x + E*y + B*x*y + F + G*z**2 = 0, 
    #use principal axis transformation and verify that the inputs 
    #correspond to an ellipsoid. 

    #Returns: (d, V, s) tuple of arrays 
        #d: shape (3,) of semi-major axes in the canonical form 
           #(X/d1)**2 + (Y/d2)**2 + (Z/d3)**2 = 1 
        #V: shape (3,3) of the eigensystem 
        #s: shape (3,) shift from the linear terms 
     

    # construct original matrix 
    M = np.array([[A, B/2, 0], 
                  [B/2, C, 0], 
                  [0, 0, G]]) 
    # construct original linear coefficient vector 
    b0 = np.array([D, E, 0]) 
    # constant term 
    c0 = F 

    # compute eigensystem 
    D, V = np.linalg.eig(M) 
    if (D <= 0).any(): 
        raise ValueError("Parameter matrix is not positive definite!") 

    # transform the shift 
    b1 = b0 @ V 

    # compute the final shift vector 
    s = b1 / (2 * D) 

    # compute the final constant term, also has to be positive 
    c2 = (b1**2 / (4 * D)).sum() - c0 
    if c2 <= 0: 
        print(b1, D, c0, c2) 
        raise ValueError("Constant in the canonical form is not positive!")

    # compute the semi-major axes 
    d = np.sqrt(c2 / D) 

    return d, V, s 

def get_ellipsoid_coordinates(A, B, C, D, E, F, G, n_theta=20, n_phi=40): 
    #Compute coordinates of an ellipsoid on an ellipsoidal grid 

    #Returns: x, y, z arrays of shape (n_theta, n_phi) 
    

    # get canonical grid 
    theta,phi = np.mgrid[0:np.pi:n_theta*1j, 0:2*np.pi:n_phi*1j] 
    r2 = np.array([np.sin(theta) * np.cos(phi), 
                   np.sin(theta) * np.sin(phi), 
                   np.cos(theta)]) # shape (3, n_theta, n_phi) 

    # get transformation data 
    d, V, s = get_transforms(A, B, C, D, E, F, G)  # could be *args I guess 

    # shift and transform back the coordinates 
    r1 = d[:,None,None]*r2 - s[:,None,None]  # broadcast along first of three axes
    r0 = (V @ r1.reshape(3, -1)).reshape(r1.shape)  # shape (3, n_theta, n_phi) 

    return r0  # unpackable to x, y, z of shape (n_theta, n_phi)

args = -a,b,c,d,e,1,1
x,y,z = get_ellipsoid_coordinates(*args) 
np.allclose(a*y**2+b*x*y+c*x+d*y+e-x**2-z, 0)
    
# create 3d axes
fig = plt.figure() 
ax = fig.add_subplot(111, projection='3d')

# plot the data
ax.plot_wireframe(x, y, z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# scaling hack
bbox_min = np.min([x, y, z])
bbox_max = np.max([x, y, z])
ax.auto_scale_xyz([bbox_min, bbox_max], [bbox_min, bbox_max], [bbox_min, bbox_max])

plt.show()
"""
    
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
 
y2 = []
y_ = []
xy = []
x_ = []
x2 = []
for i in range(len(x)):
    xy.append(b*x[i]*y[i])
    y2.append(a*(y[i]**2))
    y_.append(d*y[i])
    x_.append(c*x[i])
    x2.append(x[i]**2)

z = []
for i in range(len(x)):
    z.append(y2[i] + xy[i] + x_[i] + y_[i] + e - x2[i])

# Make data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
for i in range(len(x)):
    x[i] = x[i] * np.outer(np.cos(u), np.sin(v))
    y[i] = y[i] * np.outer(np.sin(u), np.sin(v))
    z[i] = z[i] * np.outer(np.ones(np.size(u)), np.cos(v))
 
# Plot the surface
ax.plot_surface(x, y, z, color='b')
 
plt.show()
"""