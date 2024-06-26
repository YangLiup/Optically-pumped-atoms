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
bound=1000
PP,z=gammam(3/2, bound)
PP,zz=gammam(5/2,bound)
PP,zzz=gammam(7/2, bound)

P=np.arange(0, bound, 1)/1000

# xiao's fast relaxation rate 
h=3/4
qq0=2*(19)/(3)
hh=qq0*4*(5/2)*4/(3*36*(qq0-6))

qqq0=2*(11)/(1)
hhh=qqq0*4*(7/2)*6/(3*64*(qqq0-8))

plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    fig = plt.figure(figsize=(3.2, 4))
    ax2 = fig.add_subplot(111)
    p21=ax2.plot(PP, -z)
    p22=ax2.plot(PP, -zz)
    p23=ax2.plot(PP, -zzz)
    ax2.legend(["$ I=3/2$", "$ I=5/2$", "$ I=7/2$"],
               loc='upper left', prop={'size': 9})
    ax2.plot([],[])
    ax2.plot([],[])
    ax2.plot([],[])
    ax2.plot([],[])
    p24=ax2.plot(PP, h*np.ones(bound),linestyle='dotted')
    p25=ax2.plot(PP, hh*np.ones(bound),linestyle='dotted')
    p26=ax2.plot(PP, hhh*np.ones(bound),linestyle='dotted')
    ax2.set_xlim([0,1])

    # plt.ylim([0.65, 1])
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    # plt.xlabel('Frequency (Hz)', fontsize=12)
    # plt.ylabel(' PSD ($N \chi_a^2/$Hz)', fontsize=12)

    ax2.set_xlabel('$P$', fontsize=10)
    ax2.set_ylabel('$\Gamma_t^- \\approx \Gamma_z^-$ ($R_{\\rm{se}}$) ', fontsize=10)
    ax2.tick_params(axis='x', labelsize='10' )
    ax2.tick_params(axis='y', labelsize='10' )
    # ax2.set_xlim([0,1000])
    

    plt.savefig('Gamma.png', dpi=1000)
plt.show()