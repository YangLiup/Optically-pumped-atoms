import sys
sys.path.append(r"D:\python\pythonProject\Optically_pumped_atoms\my_functions")

import matplotlib.pyplot as plt
from spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from scipy.linalg import *
import numpy as np
from matplotlib import ticker
from tqdm import trange
import scienceplots

T1=100000
T2=8000
T3=800

def master_equation(I,Rse,omega_0,T,N):
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
    Rop =0 #0.1
    Rsd =0 #50/1e4
    sx=1/np.sqrt(2)
    sz=1/np.sqrt(2)
    # --------------------------------Define the initial state-----------------------------------#
    theta = np.pi / 4
    phi = np.pi / 4
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
    P = 0.99
    beta = np.log((1 + P) / (1 - P))
    for i in np.arange(0, 2 * (a + b + 1), 1):
        Rho_ini = Rho_ini + np.exp(beta * q[i]) * v[:, [i]] * v[:, [i]].T.conjugate()
    Rho_ini = Rho_ini / np.trace(Rho_ini)

    # -----------------eigenstates-----------------#

    # Rho_ini = np.outer(np.array([0, 1, 0, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0, 0, 0]))

    # --------------------------------------Evolution under hyperfine effect, etc.--------------------------------#
    Rhot = Rho_ini
    dt = 0.01
    t = np.arange(0, T, dt)
    hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
    MSx = np.zeros(round(T / dt))
    MSz = np.zeros(round(T / dt))
    V = np.zeros(round(T / dt))

    H = omega_0 * (az - bz)  # 投影定理
    q, v = np.linalg.eig(H)
    evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)
    for n in trange(0, round(T / dt), 1):
        # -----------------Evolution-----------------#
        if n==round(T / dt)/N:
            Rop = 0.0
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
        OP = Rop * (2 * alpha @ (sx*Sx+sz*Sz) - AS)

        Rhot = evolving_B @ Rhot @ evolving_B.T.conjugate()  # Zeeman effect

        Rhot = Rse * (alpha + 4 * alpha @ mSS - Rhot) * dt + (
                ER + OP) * dt + Rhot
        Rhot = hyperfine * Rhot
        # -----------------Observables-----------------#
        MSz[n] = mSz
        MSx[n] = np.sqrt(mSx**2+mSy**2)
        # Vx = np.trace(Rhot @ ((ax + bx) @ (ax + bx) + (ay + by) @ (ay + by) + (az + bz) @ (az + bz)))
        # V[n] = Vx

    # ---------------------------Bloch Equation-----------------------------------
    Rop =0 #0.1
    transverse = np.zeros(round(T / dt))
    longitude = np.zeros(round(T / dt))
    Px = P * np.sin(theta)
    Pz = P * np.cos(theta)
    Py = 0
    for n in np.arange(0, round(T / dt), 1):
        if n==round(T / dt)/N:
            Rop = 0.0
        transverse[n] = np.sqrt(Px**2+Py**2)
        longitude[n] = Pz

        eta=(5+3*P**2)/(1-P**2)
        kappa=1/8*5/eta*(1+0.03885952*P**2+0.25893367*P**4+0.16693826*P**6-0.33048905*P**8+0.14710879*P**10)

        qnm = 2 * (3 + P ** 2) / (1 + P ** 2)
        Qnm = 2 * (3 + P ** 4) / ((1 + P ** 2) ** 2)
        Gamma = 4*eta/(1+eta)**3/kappa*omega_0**2* qnm / Qnm
        # xiao
        # qnm = 2 * (3 + P ** 2) / (1 + P ** 2)
        # Qnm = 2 * (3 + P ** 4) / ((1 + P ** 2) ** 2)
        # Gamma = (4 * (-4 + qnm) * (4 + qnm) * omega_0 ** 2 / 3 / qnm ** 2 /qnm*6) *qnm/6* qnm / Qnm

        T2 = (1 - ((qnm - Qnm) / qnm) * Pz ** 2 / P ** 2) * Gamma
        T1 = T2-Gamma/qnm*Qnm

        Pz = Pz + (-1 / Qnm * (Rsd + Rop) * Pz + Rop *
                ((1 / Qnm - 1 / qnm) * (Px *sx+Pz*sz)*(Pz-Px *sx*sz-Pz*sz*sz) / P ** 2)+Rop *sz*
                ((1 / Qnm  * (Px *sx+Pz*sz)**2/ P ** 2)+(1 / qnm  * (1-(Px *sx+Pz*sz)**2/ P ** 2)))
                - Pz * T1) * dt
        Px = Px + (-1 / Qnm * (Rsd + Rop) * Px +Rop *
                ((1 / Qnm - 1 / qnm) * (Px *sx+Pz*sz)*(Px-Px *sx*sx-Pz*sz*sx) / P ** 2)+Rop *sx*
                ((1 / Qnm  * (Px *sx+Pz*sz)**2/ P ** 2)+(1 / qnm  * (1-(Px *sx+Pz*sz)**2/ P ** 2)))
                - 1 / qnm * omega_0 * 4 * Py - Px * T2) * dt
        Py = Py + (-1 / Qnm * (Rsd + Rop) * Py + Rop * (
                1 / Qnm - 1 / qnm) * (Px*sx+Pz*sz) / P ** 2 * Py + 1 / qnm * omega_0 * 4 * Px - Py * T2) * dt
        P = np.sqrt(Px ** 2 + Py ** 2 + Pz ** 2)
    Px=2*MSx
    Pz=2*MSz
    return Px, Pz, transverse, longitude
dt=0.01
Px1,Pz1,transverse1,longitude1=master_equation(3/2,1,0.01,T1,10)
Px2,Pz2,transverse2,longitude2=master_equation(3/2,1,0.05,T2,10)
Px3,Pz3,transverse3,longitude3=master_equation(3/2,1,0.25,T3,10)


tt1=np.arange(0,T1,dt)
tt2=np.arange(0,T2,dt)
tt3=np.arange(0,T3,dt)

plt.style.use(['science'])
with plt.style.context(['science']):
    plt.rc('font',family='Times New Roman')
    fig = plt.figure(figsize=(3.2, 5))
    # plt.ylabel('Polarization ', fontsize=8)
    # plt.xlabel('$t$ ($1/R_{\mathrm{se}}$)', fontsize=8)
    # plt.xticks(ticks=[])
    # plt.yticks(ticks=[])
    ax1 = fig.add_subplot(311)
    ax1.plot(tt1, Px1)
    ax1.plot(tt1, Pz1)
    ax1.plot(tt1, transverse1)
    ax1.plot(tt1, longitude1)
    # ax1.set_xlim([0,50000])
    # ax1.set_ylim([0,1])

    ax1.set_ylabel('Polarization', fontsize=8)
    # ax1.axes.xaxis.set_ticklabels([])
    ax1.text(90000, 0.65, '(a)',fontsize=8)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.tick_params(axis='both', which='minor', labelsize=8)

    ax2 = fig.add_subplot(312)
    ax2.plot(tt2, Px2)
    ax2.plot(tt2, Pz2)
    ax2.plot(tt2, transverse2)
    ax2.plot(tt2, longitude2)
    ax2.legend(["$P_{\perp}^{\mathrm{DME}}$", "$P_z^{\mathrm{DME}}$", "$P_{\perp}^{\mathrm{BE}}$", "$P_z^{\mathrm{BE}}$"],
               loc='lower right', prop={'size': 8})

    # ax2.set_xlabel('$t$ ($1/R_{\mathrm{se}}$)', fontsize=9)
    ax2.set_ylabel('Polarization ', fontsize=8)
    # ax2.set_xlim([0,10000])
    ax2.text(7200, 0.65, '(b)',fontsize=8)
    
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.tick_params(axis='both', which='minor', labelsize=8)

    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.tick_params(axis='both', which='minor', labelsize=8)

    ax3 = fig.add_subplot(313)
    ax3.plot(tt3, Px3)
    ax3.plot(tt3, Pz3)
    ax3.plot(tt3, transverse3)
    ax3.plot(tt3, longitude3)

    # ax3.set_xlabel('$t$ ($1/R_{\mathrm{se}}$)', fontsize=8)
    ax3.set_ylabel('Polarization ', fontsize=8)
    # ax3.set_xlim([0,2000])
    ax3.text(720, 0.65, '(c)', fontsize=8)

    ax3.tick_params(axis='both', which='major', labelsize=8)
    ax3.tick_params(axis='both', which='minor', labelsize=8)
    ax3.set_xlabel('$t$ ($1/R_{\mathrm{se}}$)', fontsize=8)
    

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((0,0)) 
    ax3.xaxis.set_major_formatter(formatter)
    ax2.xaxis.set_major_formatter(formatter)
    ax1.xaxis.set_major_formatter(formatter)
    # ax2.axes.yaxis.set_ticklabels([])
    # ax4.axes.yaxis.set_ticklabels([])
    plt.tight_layout()
    plt.savefig('Evolution.png', dpi=1000)
plt.show()