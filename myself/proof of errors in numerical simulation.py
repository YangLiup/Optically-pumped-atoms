import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from sympy.physics.quantum.spin import Rotation
from sympy import pi
from NewGamma import Gamma
from scipy.linalg import *
import numpy as np
from my_functions.test7_copy import gammam
from matplotlib import ticker


DD1,z,PP1=Gamma(3/2,0.01,0.01,5000)
DD2,zz,PP2=Gamma(3/2,0.01,0.005,5000)
DD3,zzz,PP3=Gamma(3/2,0.01,0.001,5000)

happer1=5/108*np.ones(len(PP1))
happer2=5/108*np.ones(len(PP2))
happer3=5/108*np.ones(len(PP3))
fig = plt.figure()
plt.rc('font',family='Times New Roman')
ax1 = fig.add_subplot(111)
ax1.plot(PP1,(DD1-happer1)/happer1,linewidth=0.5)
ax1.plot(PP2,(DD2-happer2)/happer2,linewidth=0.5)
ax1.plot(PP3,(DD3-happer3)/happer3,linewidth=0.5)
ax1.legend(["$ dt=0.01$", "$ dt=0.005$", "$dt=0.001$"],
               loc='lower right', prop={'size': 9})


plt.savefig('Gamma_.png', dpi=1000)
plt.show()