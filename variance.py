import sys
# sys.path.append(r"D:\Optically-pumped-atoms\my_functions")
sys.path.append(r"/Users/liyang/Documents/GitHub/Optically_polarized_atoms/my_functions")

import matplotlib.pyplot as plt
from spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from scipy.linalg import *
import numpy as np
from matplotlib import ticker
from tqdm import trange
import scienceplots

I=3/2
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

# --------------------------------Define the initial state-----------------------------------#
theta = 0
phi = 0
a_theta = spin_Jx(a) * np.sin(theta) * np.cos(phi) + spin_Jy(a) * np.sin(theta) * np.sin(phi) + spin_Jz(a) * np.cos(
    theta)
b_theta = spin_Jx(b) * np.sin(theta) * np.cos(phi) + spin_Jy(b) * np.sin(theta) * np.sin(phi) + spin_Jz(b) * np.cos(
    theta)
qa, va = np.linalg.eig(np.array(a_theta.full()))
qb, vb = np.linalg.eig(np.array(b_theta.full()))
v = block_diag(va, vb)
q = np.hstack((qa, qb))
V=np.zeros(len(np.arange(1e-5,0.99,0.01)))
j=0
for P in np.arange(1e-5,0.99,0.01):
# # -----------------spin temperature state-----------------#
    Rho_ini = np.zeros(2 * (a + b + 1))
    beta = np.log((1 + P) / (1 - P))
    for i in np.arange(0, 2 * (a + b + 1), 1):
        Rho_ini = Rho_ini + np.exp(beta * q[i]) * v[:, [i]] * v[:, [i]].T.conjugate()
    Rho_ini = Rho_ini / np.trace(Rho_ini)
    V[j]=np.trace(ax@ax@Rho_ini)
    j=j+1


plt.style.use(['science'])
with plt.style.context(['science']):
    plt.rc('font',family='Times New Roman')
    fig = plt.figure()
    
    ax1 = fig.add_subplot(111)
    ax1.plot(np.arange(1e-5,0.99,0.01),V)
    ax1.set_ylabel('$\langle F^a_x \\rangle$', fontsize=8)
    ax1.set_xlabel('$\omega_0 t$', fontsize=8)
    # ax1.set_xlim(1500,2000)
    # ax1.set_ylim(-1e-3,-5e-4)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.tick_params(axis='both', which='minor', labelsize=8)
    plt.grid()
    plt.savefig('Evolution.png', dpi=1000)
plt.show()