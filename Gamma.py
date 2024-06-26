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
z,Fm1=gammam(3/2, bound)
zz,Fm2=gammam(5/2,bound)
zzz,Fm3=gammam(7/2, bound)

PP=np.arange(0, bound, 1)/1000
P=np.arange(0, bound, 1)/1000

# xiao's fast relaxation rate 
# h=3/4
# qq0=2*(19)/(3)
# hh=qq0*4*(5/2)*4/(3*36*(qq0-6))

# qqq0=2*(11)/(1)
# hhh=qqq0*4*(7/2)*6/(3*64*(qqq0-8))

# Mr zhao's fast relaxation rate 

varpsilon1=(5+P**2)/(1+P**2)
q1 = 2 * (3 + P ** 2) / (1 + P ** 2)
g1=P**2/(4*(1+P**2))
h = (q1/8)/varpsilon1* 5

varpsilon2=(35+42*P**2+3*P**4)/(3+10*P**2+3*P**4)
q2 = 2 * (19 + 26 * P ** 2 + 3 * P ** 4) / (3 + 10 * P ** 2 + 3 * P ** 4)
g2 = 8*P**2* (7 + 3 * P ** 2 ) / 27/(3 + 10 * P ** 2 + 3 * P ** 4)
hh =(19/27-g2-g2) /varpsilon2*(35/3)

varpsilon3=(21+63*P**2+27*P**4+P**6)/(1+7*P**2+7*P**4+P**6)
q3 = 2 * (11 + 35 * P ** 2 + 17 * P ** 4 + P ** 6) / (1 + 7 * P ** 2 + 7 * P ** 4 + P ** 6)
g3 = P**2* (21 + 30 * P ** 2 +5*P**4) / 16/(1+ 7 * P ** 2 + 7 * P ** 4+P**6)
hhh =(22/32-2*g3)/varpsilon3*(21)


P1,Gamma1,PP1,DD1=masterequation(3/2)
P2,Gamma2,PP2,DD2=masterequation(5/2)
P3,Gamma3,PP3,DD3=masterequation(7/2)


plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    fig = plt.figure(figsize=(3.2, 4))
    ax1 = fig.add_subplot(211)
    ax1.plot(PP1, DD1)
    ax1.plot(PP2, DD2)
    ax1.plot(PP3, DD3)
    ax1.plot([],[])
    ax1.plot([],[])
    ax1.plot([],[])
    ax1.plot([],[])
    ax1.plot(P1, Gamma1,linestyle='dotted')
    ax1.plot(P2, Gamma2,linestyle='dotted')
    ax1.plot(P3, Gamma3,linestyle='dotted')
    ax1.set_xlim([0.01,1])

    ax1.set_ylabel('$\Gamma^+_t$ $(\omega_e^2/R_{\\rm{se}})$', fontsize=10)
    ax1.tick_params(axis='x', labelsize='10' )
    ax1.tick_params(axis='y', labelsize='10' )
    ax1.axes.xaxis.set_ticklabels([])

    ax2 = fig.add_subplot(212)
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