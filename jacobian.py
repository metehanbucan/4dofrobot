import numpy as np
from math_utils import vec_to_so3, matrix_exp6, vec_to_se3, skew

def trans_to_rp(T):
    R = T[0:3, 0:3]
    p = T[0:3,3]
    return R,p

def adjoint(T):
    R,p = trans_to_rp(T)
    skewp = vec_to_so3(p)
    adT = np.zeros((6,6))
    adT[0:3,0:3] = R
    adT[3:6, 3:6] = R
    adT[3:6, 0:3] = skewp @ R
    return adT

def jacobian_space(SList, ThetaList):
    n = len(ThetaList)
    Js = np.zeros((6,n))
    Js[:,0] = SList[:,0]
    T = np.eye(4)
    for i in range (1,n):
        T = T @ matrix_exp6(vec_to_se3(SList[:, i-1] * ThetaList[i-1]))
        Js[:,i] = adjoint(T) @ SList[:,i]
    return Js

def ad(V):

    w = V[:3]
    v = V[3:]

    adV = np.zeros((6,6))

    adV[:3,:3] = skew(w)
    adV[3:,3:] = skew(w)
    adV[3:,:3] = skew(v)

    return adV

