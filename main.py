import numpy as np
import matplotlib.pyplot as plt
from math_utils import *
from ik import analytic_ik, IKinSpace, se3_to_vec, matrix_log6
from trajectory import cubic_trajectory, cubic_time_scaling
from jacobian import adjoint, ad
import modern_robotics as mr

L1 = 0.094
L2 = 0.135
L3 = 0.147

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


q0 = np.array([0, np.pi/2, np.pi/4])
qf = np.array([-np.pi/4, np.pi/4, -np.pi/2])

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
q_start = np.array([0, np.pi/2, np.pi/4])
q_end = np.array([-np.pi/4, np.pi/4, -np.pi/2]) # Örnek bir bitiş noktası

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
    s= cubic_time_scaling(t, Tf)
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
n = len(traj)
for i in range (n):
    # SList (Space Screw Axes) kullanıyorsan IKinSpace kullanmalısın
    # e_omega (tolerans) ve e_v parametrelerine dikkat et
    #q_sol, success = IKinSpace(SList, M, T_target, q_guess, 0.001, 0.5, 200)
    x = traj[i][0]
    y = traj[i][1]
    z = traj[i][2]
    q_sol, success = analytic_ik(x,y,z,L1,L2,L3)
    
    if success:
        joint_traj.append(q_sol)
        q_guess = q_sol # Bir sonraki adım için "sıcak başlangıç"
        print(f"Uyarı: {x, y, z} noktasına ULAŞILDI!")
    else:
        # Hata alıyorsan muhtemelen hedef nokta robotun erişim alanı dışındadır
        print(f"Uyarı: {x,y,z} noktasına ulaşılamadı!")
    q_guess = q_sol

joint_traj = np.array(joint_traj)
dt = Tf / (N-1)
djoint_traj = np.gradient(joint_traj,dt,axis=0)
ddjoint_traj = np.gradient(djoint_traj,dt,axis=0)

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
n = len(traj)
m1= 1
m2=1
m3 =1

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



g = np.array([0,-9.81,0])
tau_history = []
mr_tau_history = []
newMList = []
Ftip = np.zeros(6)
for m in MList:
    newMList.append(np.linalg.inv(m))

newMList = np.array(newMList)
for i in range (n):
    q = joint_traj[i]
    qd = djoint_traj[i]
    qdd = ddjoint_traj[i]
    tau = recursive_newton_euler_algorithm(SList, MList, Glist, q, qd, qdd, g)
    tau_history.append(tau)
    tau_mr = mr.InverseDynamics(q, qd, qdd, g, Ftip, newMList, Glist, SList)
    mr_tau_history.append(tau_mr)
    print(f"--- ADIM {i} ---")
    print(f"Kendi RNEA Torkum : {tau}")
    print(f"MR Kütüphane Torku: {tau_mr}")
    print("Fark:", tau - tau_mr)
    print("-------------------")

tau_history = np.array(tau_history)
mr_tau_history = np.array(mr_tau_history)

plt.subplot(2, 1, 2)
for i in range(3):
    plt.plot(ts, tau_history[:, i], label=f'Tau {i+1}')
plt.xlabel('Zaman (s)')
plt.ylabel('Tork (Nm)')
plt.title('Zamana Bağlı Gerekli Eklem Torkları (Inverse Dynamics)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# --- İLERİ DİNAMİK VE SİMÜLASYON (Euler Integration) ---

# Başlangıç durumu (Robot duruyor)
q_sim = joint_traj[0].copy()
qdot_sim = djoint_traj[0].copy()

simulated_joint_traj = []
Ftip = np.zeros(6)

sub_steps = 10
dt_sub = dt / sub_steps

# DİKKAT: Yerçekimi eksi olmalı! (RNEA hesabınla aynı)
g = np.array([0, -9.81, 0]) 

# 2. PD KONTROLCÜ KAZANÇLARI (Mıknatıs gibi yörüngeye yapıştıracak)
Kp = np.array([5.0, 5.0, 5.0]) 
Kd = np.array([0.2, 0.2, 0.2])      # Türevsel Kazanç (Sanal Amortisör)

for i in range(N):
    simulated_joint_traj.append(q_sim.copy())
    
    # 1. Hedef konum ve hız (İdeal RNEA yörüngesinden)
    q_des = joint_traj[i]
    qdot_des = djoint_traj[i]
    tau_ideal = tau_history[i]
    
    # 2. İNTEGRASYON VE ANLIK KONTROL DÖNGÜSÜ
    for _ in range(sub_steps):
        # DİKKAT: PD Kontrolcüyü alt döngünün İÇİNE aldık!
        # Robot her milisaniye nerede olduğuna bakıp torku anlık güncelleyecek.
        error = q_des - q_sim
        error_dot = qdot_des - qdot_sim
        
        # İdeal Tork + Düzeltici Tork
        current_tau = tau_ideal + (Kp * error) + (Kd * error_dot)
        
        # Fizik motoru ivmeyi hesaplıyor
        qddot_sim = mr.ForwardDynamics(q_sim, qdot_sim, current_tau, g, Ftip, newMList, Glist, SList)
        
        # Konum ve hızı Euler ile güncelliyoruz
        qdot_sim = qdot_sim + (qddot_sim * dt_sub)
        q_sim = q_sim + (qdot_sim * dt_sub)
        
    if np.any(np.isnan(q_sim)) or np.any(np.isinf(q_sim)):
        print(f"HATA: Simülasyon {i}. adımda patladı!")
        break

simulated_joint_traj = np.array(simulated_joint_traj)

# --- KARŞILAŞTIRMA GRAFİĞİ ---
plt.figure(figsize=(10, 6))
colors = ['blue', 'orange', 'green']
plot_len = len(simulated_joint_traj)

for i in range(3):
    # Planlanan ideal yörünge (Kalın ve saydam)
    plt.plot(ts[:plot_len], joint_traj[:plot_len, i], color=colors[i], linewidth=6, alpha=0.3, label=f'Planlanan q{i+1}')
    
    # Simüle edilen yörünge (İnce ve kesik çizgi)
    plt.plot(ts[:plot_len], simulated_joint_traj[:, i], '--', color=colors[i], linewidth=2, label=f'Simüle q{i+1}')

plt.xlabel('Zaman (s)')
plt.ylabel('Eklem Açıları (rad)')
plt.title('Kapalı Çevrim (Closed-Loop PD) Simülasyonu')
plt.legend()
plt.grid(True)
plt.show()