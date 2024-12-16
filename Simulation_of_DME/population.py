import sys
sys.path.append("..")
import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from scipy.linalg import *
import numpy as np
from tqdm import trange
import scienceplots


m=[1,2,3,4,5,6,7,8]
plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    fig = plt.figure(figsize=(3.35, 4))
    plt.rc('font',family='Times New Roman')
    ax1 = fig.add_subplot(211)
    ax1.bar(m, [1,0,0,0,0,0,0,0])
    ax1.set_xticks(m,['$|22\\rangle$','$|21\\rangle$','$|20\\rangle$','$|2,-1\\rangle$','$|2,-2\\rangle$','$|11\\rangle$','$|10\\rangle$','$|1,-1\\rangle$'])
    ax1.set_ylabel('Population for $|\psi\\rangle_z$', fontsize=9)
    ax2 = fig.add_subplot(212)
    ax2.bar(m, [1/16,1/4,3/8,1/4,1/16,0,0,0])
    ax2.set_xticks(m,['$|22\\rangle$','$|21\\rangle$','$|20\\rangle$','$|2,-1\\rangle$','$|2,-2\\rangle$','$|11\\rangle$','$|10\\rangle$','$|1,-1\\rangle$'])
    ax2.set_ylabel('Population for $|\psi\\rangle_x$', fontsize=9)
    plt.savefig('Population.png', dpi=1000)
plt.show()