import numpy as np
import matplotlib.pyplot as plt
from math_utils import *
from ik import analytic_ik, IKinSpace, se3_to_vec, matrix_log6
from trajectory import cubic_trajectory, cubic_time_scaling

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


q0 = np.array([0,0,0])
qf = np.array([0, angle_to_radian(-148.5058203393992), angle_to_radian(139.31693409731233)])

T = 5

t = np.linspace(0,T,100)

traj = []

for i in range(3):

    ti,qi = cubic_trajectory(q0[i],qf[i],T)

    traj.append(qi)

traj = np.array(traj)

for i in range(3):

    plt.plot(t,traj[i],label=f"joint {i+1}")

plt.legend()
plt.xlabel("time")
plt.ylabel("joint angle")
plt.title("Joint Trajectory")
plt.show()


#end effector trajectory
xs=[]
ys=[]

for i in range(len(t)):

    theta1 = traj[0][i]
    theta2 = traj[1][i]
    theta3 = traj[2][i]
    thetaList = np.array([theta1,theta2,theta3])

    fk = FKinSpace(SList,thetaList,M)
    x= fk[0,3]
    y= fk[1,3]

    xs.append(x)
    ys.append(y)

print(fk[0,3], fk[1,3])
plt.plot(xs,ys)
plt.title("End Effector Path")
plt.axis("equal")
plt.show()

#end effector space trajectory
TStart = M
TEnd = FKinSpace(SList, ThetaList, M)


Trel = np.linalg.inv(TStart) @ TEnd
Vb = se3_to_vec(matrix_log6(Trel))

Tf = 5
N = 100
ts = np.linspace(0,Tf,N)
traj = []

for t in ts:
    s = cubic_time_scaling(t,Tf)
    Tt = TStart @ matrix_exp6(vec_to_se3(Vb * s))
    traj.append(Tt)

joint_traj = []
q_guess = np.array([0,np.pi/6,-np.pi/6])
for T in traj:
    q_sol, success = IKinSpace(SList, M, T, q_guess, 200, 0.001, 50)
    joint_traj.append(q_sol)
    q_guess = q_sol

xs, ys, zs = [], [], []

for T in traj:
    xs.append(T[0,3])
    ys.append(T[1,3])
    zs.append(T[2,3])

plt.plot(xs,ys)
plt.title("End Effector Trajectory")
plt.axis("equal")
plt.show()