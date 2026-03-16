import numpy as np
import matplotlib.pyplot as plt
from math_utils import *
from ik import analytic_ik, IKinSpace, se3_to_vec, matrix_log6
from trajectory import cubic_trajectory, cubic_time_scaling
from jacobian import adjoint, ad

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
Xstart = TStart[0,3]
Ystart = TStart[1,3]
Zstart = TStart[2,3]
Xend = TEnd[0,3]
Yend = TEnd[1,3]
Zend = TEnd[2,3]

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
    x = Xstart + (Xend - Xstart)*s
    y = Ystart + (Yend - Ystart)*s
    z = Zstart + (Zend - Zstart)*s
    
    #Tt = TStart @ matrix_exp6(vec_to_se3(Vb * s))
    traj.append(np.array([x,y,z]))

# --- TERS KİNEMATİK ---
joint_traj = []
q_guess = q_start.copy() # İlk tahmin başlangıç konumu olmalı

for T_target in traj:
    # SList (Space Screw Axes) kullanıyorsan IKinSpace kullanmalısın
    # e_omega (tolerans) ve e_v parametrelerine dikkat et
    #q_sol, success = IKinSpace(SList, M, T_target, q_guess, 0.001, 0.5, 200)
    x = T_target[0]
    y = T_target[1]
    z = T_target[2]
    q_sol, success = analytic_ik(x,y,z,L1,L2,L3)
    
    if success:
        joint_traj.append(q_sol)
        q_guess = q_sol # Bir sonraki adım için "sıcak başlangıç"
        print(f"Uyarı: {T_target[0], T_target[1], T_target[2]} noktasına ULAŞILDI!")
    else:
        # Hata alıyorsan muhtemelen hedef nokta robotun erişim alanı dışındadır
        print(f"Uyarı: {T_target[0], T_target[1], T_target[2]} noktasına ulaşılamadı!")
    q_guess = q_sol

xs, ys, zs = [], [], []

for T in traj:
    xs.append(T[0])
    ys.append(T[1])
    zs.append(T[2])

plt.plot(xs,ys)
plt.title("End Effector Trajectory")
plt.axis("equal")
plt.show()

#RNEA Dynamic

def recursive_newton_euler_algorithm(Slist, Mlist, Glist, theta, thetadot, thetaddot, g):
    n = len(theta)
    Alist = [np.zeros(6) for _ in range(n+1)]
    T = np.eye(4)
    for i in range (n):
        T = T @ np.linalg.inv(Mlist[i])
        A = adjoint(np.linalg.inv(T)) @ Slist[:,i]
        Alist[i] = A

    
    AdTi = [None]*(n+1)
    V = [np.zeros(6) for _ in range(n+1)]
    Vd = [np.zeros(6) for _ in range(n+1)]

    Vd[0][3:] = -g

    for i in range(n):
        T = matrix_exp6(vec_to_se3(-Alist[i] * theta[i])) @ Mlist[i]
        AdTi[i] = adjoint(T)
        V[i+1] = AdTi[i] @ V[i] + Alist[i] * thetadot[i]

        Vd[i+1] = AdTi[i] @ Vd[i] + ad(V[i+1]) @ (Alist[i] * thetadot[i]) + Alist[i] * thetaddot[i]

    F = [np.zeros(6) for _ in range(n)]

    tau = np.zeros(n)
    Fplus = np.zeros(6)
    # backward recursion
    for i in reversed(range(n)):

        Flink = (
            Glist[i] @ Vd[i+1]
            - ad(V[i+1]).T @ (Glist[i] @ V[i+1])
        )
        F[i] = Flink + Fplus

        tau[i] = F[i] @ Alist[i]

        if i != 0:

            Fplus = AdTi[i].T @ F[i]

    return tau

m1 = m2 = m3 = 1
L1 = L2 = L3 = 1
q = [0,0,0]
qd = [0,0,0]
qdd = [0,0,0]

M_10 = np.array([[1,0,0,0],
                 [0,1,0,-L1/2],
                 [0,0,1,0],
                 [0,0,0,1]])
M_21 = np.array([[1,0,0,-L2/2],
                 [0,1,0,-L1/2],
                 [0,0,1,0],
                 [0,0,0,1]])
M_32 = np.array([[1,0,0,-(L3+L2) / 2],
                 [0,1,0,0],
                 [0,0,1,0],
                 [0,0,0,1]])
M_43 = np.array([[1,0,0,-L3/2],
                 [0,1,0,0],
                 [0,0,1,0],
                 [0,0,0,1]])

MList = np.array([M_10, M_21, M_32, M_43])

S1 = np.array([0,1,0,0,0,0])
S2 = np.array([0,0,1,L1,0,0])
S3 = np.array([0,0,1,L1,-L2,0])

SList = np.array([S1 , S2, S3]).T

I1 = np.diag([
1/12*m1*L1**2,
0,
1/12*m1*L1**2
])

I2 = np.diag([
0,
1/12*m2*L2**2,
1/12*m2*L2**2
])

I3 = np.diag([
0,
1/12*m3*L3**2,
1/12*m3*L3**2
])

def spatial_inertia(m,I):

    G = np.zeros((6,6))

    G[:3,:3] = I
    G[3:,3:] = m*np.eye(3)

    return G

G1 = spatial_inertia(m1,I1)
G2 = spatial_inertia(m2,I2)
G3 = spatial_inertia(m3,I3)

Glist = [G1,G2,G3]

q = np.array([0,0,0])
qd = np.zeros(3)
qdd = np.zeros(3)

g = np.array([0,-9.81,0])


tau = recursive_newton_euler_algorithm(SList, MList, Glist, q, qd, qdd, g)

print("sonuc:" ,tau)


def mass_matrix(q):

    n = len(q)
    M = np.zeros((n,n))

    for i in range(n):

        qd = np.zeros(n)
        qdd = np.zeros(n)
        qdd[i] = 1

        tau = recursive_newton_euler_algorithm(
            SList, MList, Glist, q, qd, qdd, np.zeros(3)
        )

        M[:,i] = tau

    return M

M = mass_matrix([0.3,0.2,0.1])
print(M)
print(M.T)