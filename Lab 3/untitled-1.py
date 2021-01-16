import numpy as np
import copy
import random
from scipy.linalg import hilbert
from scipy.optimize import lsq_linear

b = np.ones(10)
A = hilbert(10)
A = np.delete(A, 5, 1)
A = np.delete(A, 4, 1)

# Use this test instance while developing the algorithms. 
# This instance is much easier to debug than the ill-conditioned problem above. 
#A = np.array([[1, 2, 2], [4, 4, 2], [4, 6, 4]], dtype=float)
#b = np.array([3, 6, 10], dtype=float)
n = A.shape[1]
m = A.shape[0]

Q0, R0 = np.linalg.qr(A, mode='complete')
d = Q0.T.dot(b)
d1 = d[:n]
d2 = d[m-n+1:]
expected_r1 = np.linalg.norm(d2)
print(expected_r1)

def qr_MGS(A):
    Q = np.zeros(A.shape, dtype=float)
    R = np.zeros((A.shape[1], A.shape[1]), dtype=float)
    for k in range(0, A.shape[1]):
        #R[k, k] = np.linalg.norm(A[:, k])
        R[k, k] = np.sqrt(np.dot(A[:, k], A[:, k]))
        if R[k, k] == 0:
            break
        Q[:, k] = A[:, k]/R[k, k]
        for j in range(k + 1, A.shape[1]):
            R[k, j] = np.dot((Q[:,k].T),A[:, j])
            A[:, j] = A[:, j] - np.dot(R[k, j],Q[:, k])
    return Q,R

def back_substituion(U, b):
    x = np.zeros((b.shape[0]))
    for j in range(U.shape[1]-1,-1,-1):
        # singular matrix
        if U[j][j] == 0:
            break 
        x[j] = b[j]/ U[j][j]
    
        for i in range(0,j):
            b[i] = b[i] - U[i][j] * x[j]
    return x.transpose()
'''
def back_substituion(U, b): # solve Ux = y and return x
    n = len(U)
    x = np.zeros_like(b, dtype=float)
    for j in range(n - 1, -1, -1):
        if U[j][j] == 0:
            break
        x[j] = b[j] / U[j][j]
        
        for i in range(j):
            b[i] = b[i] - U[i][j] * x[j]
    return x
'''

Q, R = qr_MGS(copy.deepcopy(A))
c = Q.T.dot(b)
c1 = c[:n]
c2 = c[m-n+1:]
expected_r = np.linalg.norm(c2)
r = np.sqrt(np.dot(c2, c2))
#r1 = b - np.dot(R, x)
print(expected_r)
#print(r1)

Qt = Q.T
Qtb = Qt.dot(b)
Qtb = Qtb[:n]
x = back_substituion(R, Qtb)
print(x)
r_ = np.linalg.norm(A.dot(x)-b)
print(r_)

