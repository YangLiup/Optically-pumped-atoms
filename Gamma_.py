
import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from sympy.physics.quantum.spin import Rotation
from sympy import pi
from my_functions.test7_copy import gammam
from scipy.linalg import *
import numpy as np

bound=1000
z,Fm1=gammam(3/2, bound)
zz,Fm2=gammam(5/2,bound)
zzz,Fm3=gammam(7/2, bound)
h=3/4
qq0=2*(19)/(3)
hh=qq0*4*(5/2)*4/(3*36*(qq0-6))

qqq0=2*(11)/(1)
hhh=qqq0*4*(7/2)*6/(3*64*(qqq0-8))

PP=np.arange(0, bound, 1)/1000
plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    plt.rc('font',family='Times New Roman')
    fig1 = plt.figure()
    # p1,=plt.plot(PP, -z,linestyle='solid',color='black')
    # p2,=plt.plot(PP, -zz,linestyle='dashed',color='black' )
    # p3,=plt.plot(PP, -zzz,linestyle='dotted',color='black')
    p1,=plt.plot(PP, -z)
    p2,=plt.plot(PP, -zz)
    p3,=plt.plot(PP, -zzz)
    # plt.text(PP[850], -z[790], '${I=\\frac 3 2}$')
    # plt.text(PP[850], -zz[800], '${I=\\frac 5 2}$')
    # plt.text(PP[850], -zzz[950], '${I=\\frac 7 2}$')
    plt.plot([],[])
    plt.plot([],[])
    plt.plot([],[])
    plt.plot([],[])
    p4, = plt.plot(PP, h*np.ones(bound),linestyle='dotted')
    p5, = plt.plot(PP, hh*np.ones(bound),linestyle='dotted')
    p6, = plt.plot(PP, hhh*np.ones(bound),linestyle='dotted')
    # plt.scatter(PP[0],-z[0])
    # # plt.text(PP[0], z[0], 'start')

    # plt.scatter(PP[0],-zz[0])
    # # plt.text(PP[0], zz[0], 'start')

    # plt.scatter(PP[0],-zzz[0])
    # plt.text(PP[0], zzz[0], 'start')

    plt.xlabel('$P$', fontsize=10)
    plt.ylabel('$\Gamma_t^- \\approx \Gamma_z^-$ ($R_{\\rm{se}}$) ',fontsize=10)
    plt.legend( [p1,p2,p3],["$ I=3/2$", "$ I=5/2$", "$ I=7/2$"],
               loc='upper left', prop={'size': 9})
    # plt.ylim([0.65, 1])
    plt.xlim([0, 1])
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    # plt.xlabel('Frequency (Hz)', fontsize=12)
    # plt.ylabel(' PSD ($N \chi_a^2/$Hz)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('imag/Gamma_.png', dpi=1000)
plt.show()