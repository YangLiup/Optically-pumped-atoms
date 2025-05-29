# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
# -*- coding:utf-8 -*-
import sys
sys.path.append(r"D:\python\pythonProject\Optically_pumped_atoms\my_functions")

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from Generate_a_squeezed_state_by_QND import Generate_a_squeezed_state_by_QND
from alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from scipy.linalg import *
from spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from ptr import ptr
from scipy.linalg import *
import scienceplots
from matplotlib.ticker import FuncFormatter
import random
from tqdm import trange

# ax, ay, az, bx, by, bz, a1x, a2x, a1y, a2y, a1z, a2z, Fx, Fy, Fz

# ----------------------Squeezing----------------------------#
# N is the number of atoms, T is the squeezing time, F is the spin of atom, s is the spin of light and alpha is the coupling constant
T = 1
F= 10
s = 10
alpha = 0.25
dt = 0.01
# ----------------------squeezing----------------------#

sx = np.array(spin_Jx(s).full())
sy = np.array(spin_Jy(s).full())
sz = np.array(spin_Jz(s).full())
qy, vy = np.linalg.eig(sy)
Fx = np.array(spin_Jx(F).full())
qF, vF = np.linalg.eig(Fx)
Fy = np.array(spin_Jy(F).full())
Fz = np.array(spin_Jz(F).full())
# initiation
H = alpha * np.kron(Fz, sz)


plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    # plt.rc('font',family='Times New Roman')
    data=np.array([])
    N=500
    x=np.zeros(N)
    for j in trange(0, N, 1):
        XiF_ini = np.array(spin_coherent(F, np.pi / 2, 0).full())
        # XiF_ini = vF[:,3]
        ini_Rho_atom = np.outer(XiF_ini, XiF_ini)
        Xis_ini = np.array(spin_coherent(s, np.pi / 2, 0).full())
        Rhos_ini = np.outer(Xis_ini, Xis_ini)
        Rho_ini = np.kron(ini_Rho_atom, Rhos_ini)
        Rhot = Rho_ini
        n = round(T / dt)
        C_1z1z = np.array([None] * n)
        C_1z2z = np.array([None] * n)

        for i in np.arange(0, n, 1):
            Rhot = dt * (H @ Rhot - Rhot @ H) / 1j + Rhot
            k=random.randint(0,20)
            read = np.kron(np.eye(2*F+1), np.outer(vy[:,k] , vy[:,k].T.conjugate()))
            Rho_r = read @ Rhot @ read.T.conjugate()
            Rho_r = Rho_r / Rho_r.trace()
            Rho_atom = ptr(Rho_r, 2 * s + 1, (2*F+1) )
            Rhot = np.kron(Rho_atom, Rhos_ini)    
            if j==N-1:
                C_1z2z[i] = np.trace(Rho_atom @ Fz@Fz )-np.trace(Rho_atom @ Fz )**2   
                C_1z1z[i] = np.trace(Rho_atom @ Fz)
        x[j]=np.trace(Rho_atom @ Fz)

    tt = np.arange(0, T, dt)
    fig = plt.figure(figsize=(3.35, 5))
    ax_1  = fig.add_subplot(211)
    # for j in np.arange(0, N, 1):       
    #         ax_1.plot(tt, data[(j)*n:(j+1)*n],linewidth='0.5')
    ax_1.hist(x,bins=20,histtype='stepfilled',density=True)
    ax_1.set_xlabel('$\tilde J_x$', fontsize=9)
    ax_1.tick_params(axis='x', labelsize='9' )
    ax_1.tick_params(axis='y', labelsize='9' )

    ax_2  = fig.add_subplot(212)
    down=np.array(list(C_1z1z))-np.sqrt(np.array(list(C_1z2z)))/2
    up=np.array(list(C_1z1z))+np.sqrt(np.array(list(C_1z2z)))/2
    ax_2.plot(tt,np.array(list(C_1z1z)))
    ax_2.fill_between(tt, down,up,facecolor = 'red', alpha = 0.5)
    ax_2.text(8.5, 0.1, '$\mathbb{E}[\tilde J_x]$',fontsize=8)
    ax_2.annotate('$ \sqrt{\mathrm{var} ( J_x )}$', xy=(0.5, 1), xytext=(2, 1),
            arrowprops=dict(arrowstyle='->', color='red'),fontsize=6)

    ax_2.plot(tt, np.zeros(n),color='black',linestyle='dashed')
    ax_2.set_xlabel('t ($1/\Phi$)', fontsize=9)
    ax_2.set_ylabel('$\tilde J_x$', fontsize=9)
    ax_2.tick_params(axis='x', labelsize='9' )
    ax_2.tick_params(axis='y', labelsize='9' )
    # ax_2.plot(tt, np.zeros(n),color='black',linestyle='dashed',linewidth='0.5')
    # ax_2.set_xlabel('t ($1/\Phi$)', fontsize=10)
    # ax_2.set_ylabel('$\\tilde{J_x}$', fontsize=10)
    # ax_2.tick_params(axis='x', labelsize='10' )
    # ax_2.tick_params(axis='y', labelsize='10' )
    plt.savefig('squeezing.png', dpi=1000)

    plt.show()
