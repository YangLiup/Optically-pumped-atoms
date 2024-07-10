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

# xiao's fast relaxation rate 
# h=3/4
# qq0=2*(19)/(3)
# hh=qq0*4*(5/2)*4/(3*36*(qq0-6))

# qqq0=2*(11)/(1)
# hhh=qqq0*4*(7/2)*6/(3*64*(qqq0-8))

# Mr zhao's fast relaxation rate 

# varpsilon1=(5+P**2)/(1+P**2)
# q1 = 2 * (3 + P ** 2) / (1 + P ** 2)
# g1=P**2/(4*(1+P**2))
# h = (q1/8)/(varpsilon1)* 5

# varpsilon2=(35+42*P**2+3*P**4)/(3+10*P**2+3*P**4)
# q2 = 2 * (19 + 26 * P ** 2 + 3 * P ** 4) / (3 + 10 * P ** 2 + 3 * P ** 4)
# g2 = 8*P**2* (7 + 3 * P ** 2 ) / 27/(3 + 10 * P ** 2 + 3 * P ** 4)
# hh =(19/27-g2-g2) /varpsilon2*(35/3)

# varpsilon3=(21+63*P**2+27*P**4+P**6)/(1+7*P**2+7*P**4+P**6)
# q3 = 2 * (11 + 35 * P ** 2 + 17 * P ** 4 + P ** 6) / (1 + 7 * P ** 2 + 7 * P ** 4 + P ** 6)
# g3 = P**2* (21 + 30 * P ** 2 +5*P**4) / 16/(1+ 7 * P ** 2 + 7 * P ** 4+P**6)
# hhh =(22/32-2*g3)/varpsilon3*(21)


P1,Gamma1,PP1,DD1=masterequation(5/2)

z=gammam(5/2,PP1)


plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    fig = plt.figure(figsize=(3.35, 4))
    plt.rc('font',family='Times New Roman')

    ax2 = fig.add_subplot(211)
    ax2.plot(PP1, DD1)
    ax2.plot(P1, Gamma1)
    ax2.text(0.45, 0.05, '(b)',fontsize=8)
    ax2.set_xlim([0.001,1])

    ax3 = fig.add_subplot(212)
    p21=ax3.plot(PP1, -z)
    ax3.set_xlim([0,1])
plt.savefig('Gamma_.png', dpi=1000)