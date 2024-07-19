import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from sympy.physics.quantum.spin import Rotation
from sympy import pi
from my_functions.Master_equation import master_equation
from scipy.linalg import *
import numpy as np
from my_functions.Master_equation import master_equation

T1=5000
T2=5000
dt=0.01
Px1,Pz1,transverse1,longitude1=master_equation(3/2,1,0.01,T1,10)
Px2,Pz2,transverse2,longitude2=master_equation(3/2,1,0.1,T2,10)
tt1=np.arange(0,T1,dt)
tt2=np.arange(0,T2,dt)
plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    plt.rc('font',family='Times New Roman')
    fig = plt.figure(figsize=(3.35, 4))
    ax1 = fig.add_subplot(211)
    ax1.plot(tt1, Px1)
    ax1.plot(tt1, Pz1)
    ax1.plot(tt1, transverse1)
    ax1.plot(tt1, longitude1)

    ax1.plot([],[])
    ax1.plot([],[])
    ax1.plot([],[])

    # ax1.set_xlim([0.01,1])
    # ax1.set_ylim([0,1])

    ax1.set_ylabel('Polarization', fontsize=10)
    ax1.tick_params(axis='x', labelsize='10' )
    ax1.tick_params(axis='y', labelsize='10' )
    ax1.axes.xaxis.set_ticklabels([])

    ax2 = fig.add_subplot(212)
    ax2.plot(tt2, Px2)
    ax2.plot(tt2, Pz2)
    ax2.plot(tt2, transverse2)
    ax2.plot(tt2, longitude2)
    ax2.legend(["$P_x^{\mathrm{DM}}$", "$P_z^{\mathrm{DM}}$", "$P_x^{\mathrm{NB}}$", "$P_z^{\mathrm{NB}}$"],
               loc='upper right', prop={'size': 9})
    # ax2.plot([],[])
    # ax2.plot([],[])
    # ax2.plot([],[])
    # ax2.plot([],[])
    # p24=ax2.plot(PP, h*np.ones(bound),linestyle='dotted')
    # p25=ax2.plot(PP, hh*np.ones(bound),linestyle='dotted')
    # p26=ax2.plot(PP, hhh*np.ones(bound),linestyle='dotted')
    # ax2.set_xlim([0,1])
    # ax2.set_ylim([0,2])

    # plt.ylim([0.65, 1])
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    # plt.xlabel('Frequency (Hz)', fontsize=12)
    # plt.ylabel(' PSD ($N \chi_a^2/$Hz)', fontsize=12)

    ax2.set_xlabel('$t$ ($1/R_{\mathrm{se}}$)', fontsize=10)
    ax2.set_ylabel('Polarization ', fontsize=10)
    ax2.tick_params(axis='x', labelsize='10' )
    ax2.tick_params(axis='y', labelsize='10' )
    # ax2.set_xlim([0,1000])
    

    plt.savefig('Evolution.png', dpi=1000)
plt.show()