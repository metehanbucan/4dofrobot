from math_utils import *
import numpy as np
from jacobian import trans_to_rp, jacobian_space

def analytic_ik(x,y,z, L1, L2, L3):
    r1 = np.sqrt(x**2 + z**2)
    theta1 = np.atan2(-z,x)

    y = y-L1
    r = np.sqrt(r1**2 + y**2)

    phi = np.atan(y / r1)

    alpha = np.arccos((r**2 + L2**2 - L3**2) / (2 * L2 * r))
    theta2 = phi - alpha

    beta = np.arccos((L3**2 + L2**2 - r**2) / (2*L2*L3))
    theta3 = np.pi - beta

    return radian_to_angle(theta1),radian_to_angle(theta2),radian_to_angle(theta3)

def radian_to_angle(rad):
    return rad * 180 / np.pi


def matrix_log6(T):
    R,p = trans_to_rp(T)
    omgmat = matrix_log3(R)
    omega = so3_to_vec(omgmat)
    if(np.linalg.norm(omega) < 1e-6):
        se3mat = np.zeros((4,4))
        se3mat[0:3,3] = p
        return se3mat

    theta = np.linalg.norm(omega)
    omgmat_unit = omgmat / theta
    G_inv = (np.eye(3) / theta - 0.5 * omgmat_unit + (1/theta - 0.5 / np.tan(theta / 2)) * (omgmat_unit @ omgmat_unit))
    v = G_inv @ p
    se3mat = np.zeros((4,4))
    se3mat[0:3,0:3] = omgmat
    se3mat[0:3, 3] = v * theta
    return se3mat

def se3_to_vec(se3mat):
    vec = np.zeros((6))
    vec[0:3] = so3_to_vec(se3mat[0:3,0:3])
    vec[3:6] = se3mat[0:3,3]
    return vec

def IKinSpace(SList, M, Tsd, ThetaList0, eomg, ev, maxiter):
    thetaList = ThetaList0.copy()
    angleList = []
    for i in range (maxiter):
        Tsb = FKinSpace(SList, thetaList, M)
        Tbd = np.linalg.inv(Tsb) @ Tsd
        Vb = se3_to_vec(matrix_log6(Tbd))
        err_omg = np.linalg.norm(Vb[0:3])
        err_v = np.linalg.norm(Vb[3:6])
        if(err_omg < eomg and err_v < ev):
            for i in thetaList:
                angleList.append(radian_to_angle(i))
            return angleList,True
        J = jacobian_space(SList, thetaList)
        thetaList += np.linalg.pinv(J) @ Vb
    
    for i in thetaList:
                angleList.append(radian_to_angle(i))
    return angleList, False
