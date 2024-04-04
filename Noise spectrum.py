# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

omega = np.arange(-120, 120, 0.1)
q=5.2
Omega = 100/q*4
gammap = 10/q
gamman = 10000*(2/3+4/3/16-4*0.25/4/1.25)


Fz = 0.9192388155425111
Fx2 = 1.862499999999999
Lorentzianpp = gammap / (gammap + (omega - Omega) ** 2)
Lorentzianpn = gammap / (gammap + (omega + Omega) ** 2)
Lorentziannp = gamman / (gamman + (omega - Omega) ** 2)
Lorentziannn = gamman / (gamman + (omega + Omega) ** 2)

plt.style.use(['science'])
with plt.style.context(['science']):
    fig1 = plt.figure()
    p1, = plt.plot(omega, Fx2*(Lorentzianpp + Lorentzianpn))
    p2, = plt.plot(omega, -Fz/2*(Lorentzianpp - Lorentzianpn))
    p3, = plt.plot(omega, Fx2*(Lorentzianpp + Lorentzianpn)-Fz/2*(Lorentzianpp - Lorentzianpn))


    plt.legend([p1, p2, p3], ["Symmetric part", "Antisymmetric part", "Total"], loc='upper left',
               prop={'size': 10})
    plt.xlabel('Frequency(Hz)', fontsize=12)
    plt.ylabel(' PSD($n^2 l^2 \chi_a^2$)', fontsize=12)
    plt.ylim([-0.5,4])


    plt.savefig('Noise spectrum.png', dpi=600)


