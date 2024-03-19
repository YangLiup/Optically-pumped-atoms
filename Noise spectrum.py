# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

omega = np.arange(-50, 50, 0.1)
gammap = 10
gamman = 500
Omega = 10
Fz = 0.9192388155425111
Fx2 = 1.862499999999999
Lorentzianpp = gammap / (gammap + (omega - Omega) ** 2)
Lorentzianpn = gammap / (gammap + (omega + Omega) ** 2)
Lorentziannp = gamman / (gamman + (omega - Omega) ** 2)
Lorentziannn = gamman / (gamman + (omega + Omega) ** 2)

plt.style.use(['science'])
with plt.style.context(['science']):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    p1, = plt.plot(omega, Fx2*(Lorentzianpp + Lorentzianpn), color='brown')
    ax1.set_title("Symmetric part", fontsize=8)
    ax1.set_xlabel('Frequency(Hz)', fontsize=10)
    ax1.set_ylabel(' PSD($n^2 l^2 \chi_a^2$)', fontsize=10)
    plt.savefig('Noise spectrum1.png', dpi=600)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    p2, = plt.plot(omega, Fz/2*(Lorentzianpp - Lorentzianpn), color='olive')
    ax2.set_xlabel('Frequency(Hz)', fontsize=10)
    # ax2.set_ylabel('PSD($n^2 l^2 \chi_a^2$)', fontsize=10)
    ax2.set_title("Antisymmetric part", fontsize=8)
    plt.savefig('Noise spectrum2.png', dpi=600)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1, 1, 1)
    p4, = plt.plot(omega, Fx2*(Lorentzianpp + Lorentzianpn)+Fz/2*(Lorentzianpp - Lorentzianpn), color='purple')
    ax3.set_xlabel('Frequency(Hz)', fontsize=10)
    # ax3.set_ylabel('PSD($n^2 l^2 \chi_a^2$)', fontsize=10)
    ax3.set_title("Total spectrum", fontsize=8)
    plt.savefig('Noise spectrum3.png', dpi=600)

