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

plt.plot(xs,ys)
plt.title("End Effector Path")
plt.axis("equal")
plt.show()

#end effector space trajectory
q_start = np.array([0, np.pi/6, -np.pi/9])
q_end = np.array([np.pi/4, np.pi/4, 0]) # Örnek bir bitiş noktası

# Başlangıç ve bitiş matrislerini gerçek konumlarına çekelim (Space Frame)
TStart = FKinSpace(SList, q_start, M)
TEnd = FKinSpace(SList, q_end, M)

# --- TWIST HESABI (Body Twist Yaklaşımı) ---
# TEnd = TStart * exp([Vb]) -> Vb = log(inv(TStart) * TEnd)
Trel = np.linalg.inv(TStart) @ TEnd
Vb_mat = matrix_log6(Trel)
Vb = se3_to_vec(Vb_mat)

Tf = 5
N = 100
ts = np.linspace(0, Tf, N)
traj = []

# --- YÖRÜNGE OLUŞTURMA ---
for t in ts:
    s = cubic_time_scaling(t, Tf)
    # TStart'tan başlayarak Body Twist ile ilerle
    # Bu, uç işlevcinin (end-effector) kendi eksenine göre düz hat çizmesini sağlar
    Tt = TStart @ matrix_exp6(vec_to_se3(Vb * s))
    traj.append(Tt)

# --- TERS KİNEMATİK ---
joint_traj = []
q_guess = q_start.copy() # İlk tahmin başlangıç konumu olmalı

for T_target in traj:
    # SList (Space Screw Axes) kullanıyorsan IKinSpace kullanmalısın
    # e_omega (tolerans) ve e_v parametrelerine dikkat et
    #q_sol, success = IKinSpace(SList, M, T_target, q_guess, 0.001, 0.5, 200)
    x = T_target[0,3]
    y = T_target[1,3]
    z = T_target[2,3]
    q_sol, success = analytic_ik(x,y,z,L1,L2,L3)
    
    if success:
        joint_traj.append(q_sol)
        q_guess = q_sol # Bir sonraki adım için "sıcak başlangıç"
        print(f"Uyarı: {T_target[:3,3]} noktasına ULAŞILDI!")
    else:
        # Hata alıyorsan muhtemelen hedef nokta robotun erişim alanı dışındadır
        print(f"Uyarı: {T_target[:3,3]} noktasına ulaşılamadı!")
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

