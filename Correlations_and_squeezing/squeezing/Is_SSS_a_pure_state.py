# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
# -*- coding:utf-8 -*-
import sys
sys.path.append(r"D:\Optically-pumped-atoms\my_functions")
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from Generate_a_squeezed_state_by_QND import Generate_a_squeezed_state_by_QND
from alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from scipy.linalg import *
from spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from sympy.physics.quantum.spin import JzKet, JxKet
from sympy.physics.quantum.represent import represent
from ptr import ptr
from scipy.linalg import *
from tqdm import trange
from matplotlib.ticker import FuncFormatter

# ax, ay, az, bx, by, bz, a1x, a2x, a1y, a2y, a1z, a2z, Fx, Fy, Fz
def squeezing(dt):
# ----------------------Squeezing----------------------------#
# N is the number of atoms, T is the squeezing time, F is the spin of atom, s is the spin of light and alpha is the coupling constant
    N = 2
    I = 3 / 2
    T = 2
    a = round(I + 1 / 2)
    b = round(I - 1 / 2)
    s = 5
    alpha = 0.25
    S = 1 / 2
    U = alkali_atom_uncoupled_to_coupled(round(2 * I))
    # ----------------------squeezing----------------------#

    a = round(I + 1 / 2)
    b = round(I - 1 / 2)
    a1x, a2x, a1y, a2y, a1z, a2z, b1x, b2x, b1y, b2y, b1z, b2z, Fx, Fy, Fz = spin_operators_of_2or1_alkali_metal_atoms(N, I)
    sx = spin_Jx(s)
    sy = spin_Jy(s)
    sz = spin_Jz(s)
    qy, vy = sy.eigenstates()

    Sx = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmax().full()))
    Sx = U.T.conjugate() @ Sx @ U
    Sy = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmay().full()))
    Sy = U.T.conjugate() @ Sy @ U
    Sz = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmaz().full()))
    Sz = U.T.conjugate() @ Sz @ U

    S1x = np.kron(Sx, np.eye(2 * (a + b + 1)))
    S2x = np.kron(np.eye(2 * (a + b + 1)), Sx)
    S1y = np.kron(Sy, np.eye(2 * (a + b + 1)))
    S2y = np.kron(np.eye(2 * (a + b + 1)), Sy)
    S1z = np.kron(Sz, np.eye(2 * (a + b + 1)))
    S2z = np.kron(np.eye(2 * (a + b + 1)), Sz)

    Sz=S1z+S2z
    H = alpha * np.kron(Sz, sz.full())

    XiF_ini = np.vstack((np.array(spin_coherent(a, np.pi / 2, 0).full()), np.array(zero_ket(2 * b + 1).full())))
    if N == 2:
        XiF_ini = np.kron(XiF_ini, XiF_ini)

    ini_Rho_atom = np.outer(XiF_ini, XiF_ini)
    Xis_ini = np.array(spin_coherent(s, np.pi / 2, 0).full())
    Rhos_ini = np.outer(Xis_ini, Xis_ini)
    Rho_ini = np.kron(ini_Rho_atom, Rhos_ini)
    Rhot = Rho_ini
    n = round(T / dt)
    q, v = np.linalg.eig(H)
    evolving = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)
    # evolving
    for i in trange(0, n, 1):
        Rhot = evolving @ Rhot @ evolving.T.conjugate()
    read = tensor(tensor(qeye(2 * (a + b + 1)), qeye(2 * (a + b + 1)), vy[1] * vy[1].dag())).full()
    Rho_r = read @ Rhot @ read.T.conjugate()
    Rho_r = Rho_r / Rho_r.trace()
    return np.trace(Rho_r@Rho_r)
dt_range=np.array([0.02,0.01,0.001])
trace=np.zeros(3)
i=0
for dt in dt_range:
    trace[i]=squeezing(dt)
    i=i+1

plt.figure()
dt_range=np.array([2e-2,1e-2,1e-3])*1e4
plt.scatter(dt_range,trace)
plt.xticks(dt_range)
plt.xlabel('dt(1e-4)')
plt.ylabel('$\\text{tr}(\\rho^2)$')
plt.savefig('squeezing.png', dpi=600)
plt.show()
