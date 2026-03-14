import numpy as np

def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def vec_to_so3(vec):
    return skew(vec)

def so3_to_vec(so3):
    return np.array([so3[2,1], so3[0,2], so3[1,0]])

def matrix_exp3(so3):
    omega = so3_to_vec(so3)
    theta = np.linalg.norm(omega)
    if(theta < 1e-6):
        return np.eye(3)
    omegahat = so3 / theta
    return(
        np.eye(3) + np.sin(theta) * omegahat + (1-np.cos(theta)) * np.dot(omegahat, omegahat)
    )

def vec_to_se3(vec):
    se3mat = np.zeros((4,4))
    omega = vec[0:3]
    v = vec[3:6]
    se3mat[0:3,0:3] = vec_to_so3(omega)
    se3mat[0:3,3] = v
    return se3mat

def matrix_exp6(se3mat):
    omega = se3mat[0:3, 0:3]
    v = se3mat[0:3,3]
    theta = np.linalg.norm(so3_to_vec(omega))
    if (theta < 1e-6):
        T = np.eye(4)
        T[0:3,3] = v
        return T
    omega_unit = omega / theta
    R = matrix_exp3(omega)
    G = (np.eye(3) * theta + (1 - np.cos(theta)) * omega_unit + (theta - np.sin(theta)) * np.dot(omega_unit, omega_unit))
    P = np.dot(G , v / theta)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = P
    return T

def FKinSpace(SList, ThetaList, M):
    T = np.eye(4)
    length = len(ThetaList)
    for i in range (length):
        T = T @ matrix_exp6(vec_to_se3(SList[:, i] * ThetaList[i]))
    return T @ M

def angle_to_radian(angle):
    return (angle / 180) * np.pi

def matrix_log3(R):
    acos_input = (np.trace(R)-1)/2.0
    theta = np.arccos(np.clip(acos_input, -1, 1))
    if abs(theta) < 1e-6:
        return np.zeros((3,3))
    
    return theta / (2 * np.sin(theta)) * (R - R.T)