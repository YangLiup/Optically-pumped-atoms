
import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from sympy.physics.quantum.spin import Rotation
from sympy import pi
from test7 import gammam
from scipy.linalg import *
import numpy as np

Fm1=gammam(3/2, 5, 1)
Fm2=gammam(5/2, 7, 1)
Fm3=gammam(7/2, 9, 1)
h=3/4
qq0=2*(19)/(3)
hh=qq0*4*(5/2)*4/(3*36*(qq0-6))

qqq0=2*(11)/(1)
hhh=qqq0*4*(7/2)*6/(3*64*(qqq0-8))

PP=np.arange(0, 1001, 1)/1000
plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    fig1 = plt.figure()
    p1,=plt.plot(PP, -Fm1,linestyle='solid',color='tomato')
    p2,=plt.plot(PP, -Fm2,linestyle='dashed',color='tomato' )
    p3,=plt.plot(PP, -Fm3,linestyle='dotted',color='tomato')


    p4, = plt.plot(PP, h*np.ones(1001),linestyle='solid',color='black')
    p5, = plt.plot(PP, hh*np.ones(1001),linestyle='dashed',color='black')
    p6, = plt.plot(PP, hhh*np.ones(1001),linestyle='dotted',color='black')
    plt.xlabel('$P$', fontsize=11)
    plt.ylabel('$\Gamma_z^-$ ($R_{\\rm{se}}$) ',fontsize='11')
    plt.legend( [p1,p2,p3],["$ I=3/2$", "$ I=5/2$", "$ I=7/2$"],
               loc='upper right', prop={'size': 10})
    plt.ylim([0.65, 0.9])
    # plt.xlim([0, 1])
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    # plt.xlabel('Frequency (Hz)', fontsize=12)
    # plt.ylabel(' PSD ($N \chi_a^2/$Hz)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('imag/Power.png', dpi=600)