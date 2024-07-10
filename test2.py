import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from sympy.physics.quantum.spin import Rotation
from sympy import pi
from my_functions.master_relaxationrate import masterequation
from scipy.linalg import *
import numpy as np
from my_functions.test7_copy import gammam

I=3/2
omega_0=0.01
a = round(I + 1 / 2)
b = round(I - 1 / 2)

PP1,DD1=masterequation(3/2,round(1e7),0.01,0.99)

# I=3/2
omega_e=omega_0*4
q = 2 * (3 + PP1 ** 2) / (1 + PP1** 2)
Gamma1 =  -np.real(((-36 * q) +np.sqrt(q*(1296*q+6912*1j*omega_e-576*q*omega_e**2)))/96/q/omega_e**2)


plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    fig = plt.figure()
    plt.rc('font',family='Times New Roman')

    ax2 = fig.add_subplot(111)
    ax2.plot(PP1, DD1)
    ax2.plot([],[])
    ax2.plot([],[])
    ax2.plot([],[])
    ax2.plot([],[])
    ax2.plot(PP1, Gamma1,linestyle='dotted')
    ax2.set_xlim([0.02,1])
    ax2.set_ylim([0,0.05])

    ax2.set_ylabel('$\Gamma^+_t$ $(\omega_e^2/R_{\\rm{se}})$', fontsize=10)
    ax2.tick_params(axis='x', labelsize='10' )
    ax2.tick_params(axis='y', labelsize='10' )

plt.savefig('Gamma_.png', dpi=1000)
