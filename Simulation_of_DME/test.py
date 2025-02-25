import sys
sys.path.append("..")
import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from scipy.linalg import *
import numpy as np
from tqdm import trange
import scienceplots

def master_equation(I,Rse,omega_0,rf,T,Arf):
    a = round(I + 1 / 2)
    b = round(I - 1 / 2)

    # --------------------------------Generate the angular momentum operators-----------------------------------#
    U = alkali_atom_uncoupled_to_coupled(round(2 * I))
    ax, ay, az, bx, by, bz = spin_operators_of_2or1_alkali_metal_atoms(1, I)
    Sx = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmax().full()))
    Sx = U.T.conjugate() @ Sx @ U
    Sy = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmay().full()))
    Sy = U.T.conjugate() @ Sy @ U
    Sz = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmaz().full()))
    Sz = U.T.conjugate() @ Sz @ U

    # --------------------------------Characterize interactions envolved-----------------------------------#
    # omega_0 = 0.01
    Rsd = 0.
    # sx=1  #/np.sqrt(2)
    # sz=0    #1/np.sqrt(2)
    # --------------------------------Define the initial state-----------------------------------#
    theta = np.pi/4
    phi = 0
    a_theta = spin_Jx(a) * np.sin(theta) * np.cos(phi) + spin_Jy(a) * np.sin(theta) * np.sin(phi) + spin_Jz(a) * np.cos(
        theta)
    b_theta = spin_Jx(b) * np.sin(theta) * np.cos(phi) + spin_Jy(b) * np.sin(theta) * np.sin(phi) + spin_Jz(b) * np.cos(
        theta)
    qa, va = np.linalg.eig(np.array(a_theta.full()))
    qb, vb = np.linalg.eig(np.array(b_theta.full()))
    v = block_diag(va, vb)
    q = np.hstack((qa, qb))
    Rho_ini = np.zeros(2 * (a + b + 1))

    # # -----------------spin temperature state-----------------#
    P = 0.9999999
    beta = np.log((1 + P) / (1 - P))
    for i in np.arange(0, 2 * (a + b + 1), 1):
        Rho_ini = Rho_ini + np.exp(beta * q[i]) * v[:, [i]] * v[:, [i]].T.conjugate()
    Rho_ini = Rho_ini / np.trace(Rho_ini)

    # -----------------eigenstates-----------------#

    # Rho_ini = np.outer(np.array([0, 1, 0, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0, 0, 0]))

    # --------------------------------------Evolution under hyperfine effect, etc.--------------------------------#
    # omega_rev=1/100
    omega_rev=0.15
    Rhot = Rho_ini
    dt = 0.01
    hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
    MSx = np.zeros(round(T / dt))
    for n in trange(0, round(T / dt), 1):
        # if n<round(T / dt)/50:
        #     omega_1=Arf*np.sin(rf*n*dt)
        #     probability=np.diagonal(Rhot)
        # else: omega_1=0
        # H = omega_0 * Sz+omega_1*Sy # 投影定理
        H = omega_0 * az-omega_rev* az@az-omega_0*bz+omega_rev* bz@bz # 投影定理
        q, v = np.linalg.eig(H)
        evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)
        # -----------------Evolution-----------------#
        x1 = Rhot @ Sx
        x2 = Rhot @ Sy
        x3 = Rhot @ Sz
        AS = 3 / 4 * Rhot - (Sx @ x1 + Sy @ x2 + Sz @ x3)
        alpha = Rhot - AS
        mSx = np.trace(x1)
        mSy = np.trace(x2)
        mSz = np.trace(x3)
        mSS = mSx * Sx + mSy * Sy + mSz * Sz
        ER = -Rsd * AS
        # OP = Rop * (2 * alpha @ (sx*Sx+sz*Sz) - AS)
        # Rhot =Rhot+ (H@Rhot-Rhot@H)/1j*dt # Zeeman effect

        Rhot = evolving_B @ Rhot @ evolving_B.T.conjugate()  # Zeeman effect

        Rhot = Rse * (alpha + 4 * alpha @ mSS - Rhot) * dt + (
                ER ) * dt + Rhot
        Rhot = hyperfine * Rhot
        # -----------------Observables-----------------#
        MSx[n] = mSx

    return MSx


T1=1000
dt=0.01
# master_equation(I,Rse,omega_0,rf,T,Arf):

Msy1=master_equation(3/2,0.,0.1,2,T1,100/1000)
# Msy2,probability2=master_equation(3/2,0.,1,1,T1,10/1000)
# Msy3,probability3=master_equation(3/2,0.,1,1,T1,100/1000)
# Msy4,probability4=master_equation(3/2,0.,1,1,T1,200/1000)
tt=np.arange(0,T1,dt)
m=[1,2,3,4,5,6,7,8]

plt.style.use(['science'])
with plt.style.context(['science']):
    fig = plt.figure()
    plt.rc('font',family='Times New Roman')
    ax1 = fig.add_subplot(111)
    pp,=ax1.plot(tt, Msy1)
    ax1.set_ylabel('$\langle Sx \\rangle$')
    ax1.set_xlabel('$t $')
    ax1.axes.xaxis.set_ticklabels([])

plt.savefig('FID_.png', dpi=1000)
plt.show()
# plt.style.use(['science','nature'])
# with plt.style.context(['science','nature']):
#     fig = plt.figure(figsize=(3.35, 8))
#     plt.rc('font',family='Times New Roman')
#     ax1 = fig.add_subplot(411)
#     pp,=ax1.plot(tt, Msy1)
#     ax1.set_ylabel('$Sx$', fontsize=9)
#     ax1.legend(["$A_{rf}=\\frac {1 \omega_0} {1000}$,$T_{rf}=\\frac {20} { \omega_0}  $"])
#     ax2 = fig.add_subplot(412)
#     pp,=ax2.plot(tt, Msy2)
#     ax2.set_ylabel('$Sx$', fontsize=9)
#     ax2.legend(["$A_{rf}=\\frac {10 \omega_0} {1000}$, $T_{rf}=\\frac {20} { \omega_0}  $"])
#     ax3 = fig.add_subplot(413)
#     pp,=ax3.plot(tt, Msy3)
#     ax3.set_ylabel('$Sx$', fontsize=9)
#     ax3.legend(["$A_{rf}=\\frac {100 \omega_0} {1000}$, $T_{rf}=\\frac {20} { \omega_0} $ "])
#     ax4 = fig.add_subplot(414)
#     pp,=ax4.plot(tt, Msy4)
#     ax4.legend(["$A_{rf}=\\frac {200 \omega_0} {1000}$, $T_{rf}=\\frac {20} { \omega_0} $"])
#     ax4.set_xlabel('$t (1/\omega_0)$', fontsize=9)
#     ax4.set_ylabel('$Sx$', fontsize=9)
# plt.savefig('FID_.png', dpi=1000)
# plt.show()
# plt.style.use(['science','nature'])
# with plt.style.context(['science','nature']):
#     fig = plt.figure(figsize=(3.35, 8))
#     plt.rc('font',family='Times New Roman')
#     ax1 = fig.add_subplot(411)
#     ax1.bar(m, probability1)
#     ax1.set_xticks(m,['$|22\\rangle$','$|21\\rangle$','$|20\\rangle$','$|2,-1\\rangle$','$|2,-2\\rangle$','$|11\\rangle$','$|10\\rangle$','$|1,-1\\rangle$'])
#     ax1.set_ylabel('$P_m$', fontsize=9)
#     # ax1.legend(["$A_{rf}=\\frac {1 \omega_0} {1000}$,$T_{rf}=\\frac {20} { \omega_0}  $"])
#     ax2 = fig.add_subplot(412)
#     ax2.bar(m, probability2)
#     ax2.set_xticks(m,['$|22\\rangle$','$|21\\rangle$','$|20\\rangle$','$|2,-1\\rangle$','$|2,-2\\rangle$','$|11\\rangle$','$|10\\rangle$','$|1,-1\\rangle$'])   
#     ax2.set_ylabel('$P_m$', fontsize=9)
#     # ax2.legend(["$A_{rf}=\\frac {10 \omega_0} {1000}$, $T_{rf}=\\frac {20} { \omega_0}  $"])
#     ax3 = fig.add_subplot(413)
#     ax3.bar(m, probability3)
#     ax3.set_xticks(m,['$|22\\rangle$','$|21\\rangle$','$|20\\rangle$','$|2,-1\\rangle$','$|2,-2\\rangle$','$|11\\rangle$','$|10\\rangle$','$|1,-1\\rangle$']) 
#     ax3.set_ylabel('$P_m$', fontsize=9)
#     # ax3.legend(["$A_{rf}=\\frac {100 \omega_0} {1000}$, $T_{rf}=\\frac {20} { \omega_0} $ "])
#     ax4 = fig.add_subplot(414)
#     ax4.bar(m, probability4)
#     ax4.set_xticks(m,['$|22\\rangle$','$|21\\rangle$','$|20\\rangle$','$|2,-1\\rangle$','$|2,-2\\rangle$','$|11\\rangle$','$|10\\rangle$','$|1,-1\\rangle$']) 
#     # ax4.legend(["$A_{rf}=\\frac {200 \omega_0} {1000}$, $T_{rf}=\\frac {20} { \omega_0} $"])
#     ax4.set_xlabel('$m$', fontsize=9)
#     ax4.set_ylabel('$P_m$', fontsize=9)
# plt.savefig('FID_.png', dpi=1000)
# plt.show()

# plt.style.use(['science','nature'])
# with plt.style.context(['science','nature']):
#     fig = plt.figure(figsize=(3.35, 4))
#     plt.rc('font',family='Times New Roman')
#     ax1 = fig.add_subplot(211)
#     ax1.bar(m, probability1)
#     ax1.set_xticks(m,['$|22\\rangle$','$|21\\rangle$','$|20\\rangle$','$|2,-1\\rangle$','$|2,-2\\rangle$','$|11\\rangle$','$|10\\rangle$','$|1,-1\\rangle$'])
#     ax1.set_ylabel('$P_m$', fontsize=9)
#     ax2 = fig.add_subplot(212)
#     ax2.plot(tt, Msy1)
#     ax2.set_ylabel('$Sx$', fontsize=9)
#     ax2.set_xlabel('$t$', fontsize=9)
#     ax2.legend(["$A_{rf}=\\frac {100 \omega_0} {1000}$,$T_{rf}=\\frac {20} { \omega_0}  $"])

#     plt.savefig('FID2.png', dpi=1000)
# plt.show()