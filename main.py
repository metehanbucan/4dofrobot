import numpy as np
from math_utils import *
from ik import analytic_ik, IKinSpace

L1 = 9.4
L2 = 13.5
L3 = 14.7

M = np.array([
    [1,0,0,L2+L3],
    [0,1,0,L1],
    [0,0,1,0],
    [0,0,0,1]
])

S1 = np.array([0,1,0,0,0,0])
S2 = np.array([0,0,1,L1,0,0])
S3 = np.array([0,0,1,L1,-L2,0])

theta1 = angle_to_radian(0)
theta2 = angle_to_radian(-148.5058203393992)
theta3 = angle_to_radian(139.31693409731233)

SList = np.array([S1 , S2, S3]).T
ThetaList = np.array([theta1, theta2, theta3])

T = FKinSpace(SList, ThetaList, M)
print(T)

new_theta_list = analytic_ik(3, 0, 0, L1, L2, L3)
print(new_theta_list)

angles,result = IKinSpace(SList, M, T, np.array([0, angle_to_radian(-30), angle_to_radian(70)]),  1e-3, 1e-3, 50)

print(angles, result)
