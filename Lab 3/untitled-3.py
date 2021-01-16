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
    R = np.zeros((A.shape[0],A.shape[0]))
    Q = np.zeros(A.shape)
    for k in range(0, A.shape[1]):
        R[k, k] = np.sqrt(np.dot(A[:, k], A[:, k]))
        if R[k, k] == 0:
            break
        Q[:, k] = A[:, k]/R[k, k]
        for j in range(k, A.shape[1]):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] = A[:, j] - R[k, j]*Q[:, k]
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

def lu_factor_v2(A):
    L = np.eye(A.shape[0])
    U = A.copy()
    for k in range(U.shape[1]):
        if U[k][k] == 0:
            break
       # for i in range(k+1,A.shape[0]):
       #     M[i][k] = A[i][k]/A[k][k]
       #     A[i][k] = 0
       # for j in range(k+1,A.shape[1]):
       #     for i in range(k+1,A.shape[1]):
       #         A[i][j] = A[i][j] - M[i][k]*A[k][j]
        
        temp = U[k+1:,k]/U[k,k]
        L[k+1:,k] = temp
        # adding the dimention to increase the shape of the temp, to make temp*U[k] caculable
        # here is the citation for adding dimention£º
        # https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
        temp = np.expand_dims(temp, axis=1)
        U[k+1:] = U[k+1:] - np.multiply(temp,U[k])
        # print(U[k])
        # print(U[k+1:, k]/U[k, k] * U[k])
        # print(U[k+1:, k]/U[k, k])
    #print(U)
    #print(L)    
        
    return L, U
    
L, U = lu_factor_v2(copy.deepcopy(A))
y = forward_substituion(L, copy.deepcopy(b))
x = back_substituion(U, y)
print('x: ', x)

Q, R = qr_MGS(copy.deepcopy(A));
c = Q.T.dot(b)
c1 = c[:n]
c2 = c[m-n+1:]
expected_r = np.linalg.norm(c2)
r = np.sqrt(np.dot(c2, c2))
r1 = b - np.dot(R, x)
print(expected_r)
print(r1)

def qr_factorization(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]

        for i in range(j - 1):
            q = Q[:, i]
            R[i, j] = q.dot(v)
            v = v - R[i, j] * q

        norm = np.linalg.norm(v)
        Q[:, j] = v / norm
        R[j, j] = norm
    return Q, R

Q2, R2 = qr_factorization(copy.deepcopy(A))

a = Q2.T.dot(b)
a1 = a[:n]
a2 = a[m-n+1:]
expected_r2 = np.linalg.norm(a2)
r2 = np.sqrt(np.dot(a2, a2))
#r1 = b - np.dot(R, x)
print(expected_r2)
#print(r1)