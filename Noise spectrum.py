# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
import numpy as np
import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from scipy.linalg import *
import matplotlib.pyplot as plt
import scienceplots

omega = np.arange(-10, 10, 0.1)
gammap = 2
gamman = 10
Omega = 5
Lorentzianpp = 1 / (gammap + (omega - Omega) ** 2)
Lorentzianpn = 1 / (gammap + (omega + Omega) ** 2)
Lorentziannp = 1 / (gamman + (omega - Omega) ** 2)
Lorentziannn = 1 / (gamman + (omega + Omega) ** 2)

plt.style.use(['science'])
with plt.style.context(['science']):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 2)
    p1, = plt.plot(omega, 3 * Lorentzianpp + Lorentzianpn + 3 * Lorentziannp + Lorentziannn)
    ax2 = fig.add_subplot(2, 2, 1)
    p2, = plt.plot(omega, 2 * Lorentzianpp + 2 * Lorentzianpn)
    p3, = plt.plot(omega, 2 * Lorentziannp + 2 * Lorentziannn)
    ax3 = fig.add_subplot(2, 2, 2)
    p4, = plt.plot(omega, Lorentzianpp - Lorentzianpn)
    p5, = plt.plot(omega, +Lorentziannp - Lorentziannn)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax1.set_title("Total spectrum", fontsize=8)
    ax2.set_title("Symmetric part", fontsize=8)
    ax3.set_title("Antisymmetric part", fontsize=8)

    ax3.legend([p4, p5],
               ["Narrow linewidth", "Broad linewidth"]
               , loc='upper left', prop={'size': 6})

    ax1.yaxis.set_major_formatter(plt.NullFormatter())
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    ax2.yaxis.set_major_formatter(plt.NullFormatter())
    ax2.xaxis.set_major_formatter(plt.NullFormatter())
    ax3.yaxis.set_major_formatter(plt.NullFormatter())
    ax3.xaxis.set_major_formatter(plt.NullFormatter())
    # plt.ylim(-0.5, 5.2)
    plt.savefig('Noise spectrum.png', dpi=600)
