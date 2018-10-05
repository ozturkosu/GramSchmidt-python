import numpy as np
from numpy import linalg as LA
from scipy import linalg

import matplotlib.pyplot as plt

def cgs(A):

    m, n = A.shape
    R = np.zeros((n, n))
    Q = np.empty((m, n))
    R[0, 0] = linalg.norm(A[:, 0])
    Q[:, 0] = A[:, 0] / R[0, 0]
    for k in range(1, n):
        R[:k-1, k] = np.dot(Q[:m, :k-1].T, A[:m, k])
        z = A[:m, k] - np.dot(Q[:m, :k-1], R[:k-1, k])
        R[k, k] = linalg.norm(z) ** 2
        Q[:m, k] = z / R[k, k]
    return Q, R

def modified_gs(A):

    m, n = A.shape
    R = np.zeros((n, n))
    Q = np.empty((m, n))

    v = A.copy()

    for i in range(n):
        R[i,i]= linalg.norm(v[:,i])
        Q[:,i] = v[:,i]/R[i,i]

        for j in range(i+1,n):
            R[i,j] = np.sum(Q[:,i]*v[:,i])
            v[:,j]=v[:,j]-R[i,j]*Q[:,i]

    return Q, R


gs_norms = []
modgs_norms = []
doublecgs_norms = []


for n in range(2,13):

    H = linalg.hilbert(n)

    Q, R  = cgs(H)
    #the_norm = -np.log10(LA.norm(np.eye(n) - np.dot(Q.T, Q) , 'fro'))
    the_norm = -np.log10(LA.norm(np.eye(n) - np.dot(np.transpose(Q), Q) , 'fro'))
    
    CheckH=np.dot(Q,R)

    gs_norms.append(the_norm)

    Q, R  = cgs(Q)
    #the_norm = -np.log10(LA.norm( np.eye(n) - np.dot(Q.T, Q) , 'fro'))
    the_norm = -np.log10(LA.norm(np.eye(n) - np.dot(np.transpose(Q), Q) , 'fro'))
    
    doublecgs_norms.append(the_norm)

    Q, R  = modified_gs(H)
    #the_norm = -np.log10(LA.norm( np.eye(n) - np.dot(Q.T, Q) , 'fro'))
    the_norm = -np.log10(LA.norm(np.eye(n) - np.dot(np.transpose(Q), Q) , 'fro'))

    modgs_norms.append(the_norm)

    print('H')
    print(H)
    print('Q')
    print(Q)
    print('R')
    print(R)

    print('CheckH')
    print(CheckH)
#

    # print(the_norm )

plt.plot(np.arange(2,13), gs_norms, label='CGS')
plt.plot(np.arange(2,13), doublecgs_norms, label='double-CGS')

plt.plot(np.arange(2,13), modgs_norms, label='modified-CGS')

plt.legend()

plt.show()


# print('QR')
# print( np.dot(Q, R) )
