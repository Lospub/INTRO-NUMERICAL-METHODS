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

def qr_CGS(A):
    R = np.zeros((A.shape[0],A.shape[0]))
    Q = np.zeros(A.shape)
    for k in range(0, A.shape[1]):
        Q[k] = A[k]
        for j in range(k):
            R[j, k] = np.dot((Q[j].T), A[k])
            Q[k] = Q[k] - np.dot(R[j, k],Q[j])
        #R[k, k] = np.sqrt(np.dot(Q[k], Q[k]))
        R[k, k] = np.linalg.norm(Q[k])
        if R[k, k] == 0:
            break
        Q[k] = Q[k]/R[k, k]
    return Q,R
        
        
def forward_substituion(L, b):
    x = np.zeros((b.shape[0]))
    for j in range(0, L.shape[1]):
        # singular matrix
        if L[j][j] == 0:
            break 
        x[j] = b[j]/ L[j][j]
    
        for i in range(j, L.shape[0]):
            b[i] = b[i] - L[i][j] * x[j]
    return x

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

Q, R = qr_CGS(copy.deepcopy(A));
c = Q.T.dot(b)
c1 = c[:n]
c2 = c[m-n+1:]
expected_r = np.linalg.norm(c2)
r = np.sqrt(np.dot(c2, c2))
print(expected_r)
print(r)

Qt = Q.T
Qtb = Qt.dot(b)
Qtb = Qtb[:n]
x = back_substituion(R, Qtb)
print(x)
r_ = np.linalg.norm(A.dot(x)-b)
print(r_)
