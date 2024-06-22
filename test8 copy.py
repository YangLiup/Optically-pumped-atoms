
import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from sympy.physics.quantum.spin import Rotation
from sympy import pi
from test7 import gammam
from scipy.linalg import *
import numpy as np

bound=1001
Fm1=gammam(3/2,5,1,0.0001)

h=3/4
qq0=2*(19)/(3)
hh=qq0*4*(5/2)*4/(3*36*(qq0-6))

qqq0=2*(11)/(1)
hhh=qqq0*4*(7/2)*6/(3*64*(qqq0-8))

PP=np.arange(0, bound, 1)/1000
plt.style.use(['science','ieee'])
with plt.style.context(['science','ieee']):
    fig1 = plt.figure(figsize=(2.8, 2.5))
    p1,=plt.plot(PP, -Fm1,linestyle='solid',color='black')



    p4, = plt.plot(PP, h*np.ones(bound),linestyle='solid',color='red')

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('imag/Gamma_.png', dpi=600)
plt.show()