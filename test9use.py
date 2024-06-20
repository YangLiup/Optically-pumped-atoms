
import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from sympy.physics.quantum.spin import Rotation
from sympy import pi
from test9 import gammam
from scipy.linalg import *
import numpy as np

Fm1=gammam(0.5,5/2,7, 1,0.95)
Fm2=gammam(0.5,5/2,7, 1,0.95)
tt=np.arange(0, 10001, 1)
plt.style.use(['science','ieee'])
with plt.style.context(['science','ieee']):
    fig1 = plt.figure(figsize=(2.8, 2.5))
    p1,=plt.plot(tt*0.001, Fm1,linestyle='solid',color='black')



    p4, = plt.plot(tt*0.001,Fm2,linestyle='solid',color='red')

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('imag/Gamma_.png', dpi=600)
plt.show()