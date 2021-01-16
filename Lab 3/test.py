import numpy as np
import copy
import random
from scipy.linalg import hilbert
from scipy.optimize import lsq_linear

b = np.ones(10)
A = hilbert(10)
A = np.delete(A, 5, 1)
A = np.delete(A, 4, 1)
n = A.shape[1]
m = A.shape[0]
print('Condition number of A: ', np.linalg.norm(A) * np.linalg.norm(np.linalg.pinv(A)))

# Use this test instance while developing the algorithms. 
# This instance is much easier to debug than the ill-conditioned problem above. 
A_test = np.array([[1, 2, 2], [4, 4, 2], [4, 6, 4]], dtype=float)
b_test = np.array([3, 6, 10], dtype=float)

Q, R0 = np.linalg.qr(A, mode='complete')
c = Q.T.dot(b)
c1 = c[:n]
c2 = c[m-n+1:] # The +1 here accounts for the fact that numpy arrays start at 0!
print(c1, c2)
# Compute the expected residual from the QR factorization "c_2"
expected_r = np.linalg.norm(c2)


