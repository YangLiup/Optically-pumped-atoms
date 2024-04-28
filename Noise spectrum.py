# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

omega = np.arange(-1000,1000 , 0.1)
q=5.2
Omega = 100/q*4
gammap = 2*10/q
gamman = 1000*(2/3+4/3/16-4*0.25/4/1.25)


Fz = 1.3
Fx2 = 1.3
Fxn2=10
Lorentzianpp = gammap / (gammap**2 + (omega - Omega) ** 2)
Lorentzianpn = gammap / (gammap**2 + (omega + Omega) ** 2)
Lorentziannp = gamman / (gamman**2 + (omega - Omega) ** 2)
Lorentziannn = gamman / (gamman**2 + (omega + Omega) ** 2)

plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    fig1 = plt.figure()
    # p3, = plt.plot(omega, Fx2*(Lorentzianpp + Lorentzianpn)-Fz/2*(Lorentzianpp - Lorentzianpn))
    # p1, = plt.plot(omega, Fx2*(Lorentzianpp + Lorentzianpn))
    # p2, = plt.plot(omega, -Fz/2*(Lorentzianpp - Lorentzianpn))
    p1, = plt.plot(omega, 9/169*Fxn2*(Lorentziannp + Lorentziannn)+100/169*Fx2*(Lorentzianpp + Lorentzianpn),color='maroon')
    # p2, = plt.plot(omega, -Fz/2*(Lorentzianpp - Lorentzianpn))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    # plt.legend([p1, p2, p3], ["Symmetric part", "Antisymmetric part", "Total"], loc='upper center',
    #            prop={'size': 10})
    # plt.xlabel('Frequency (Hz)', fontsize=12)
    # plt.ylabel(' PSD ($ N \chi_+^2$/Hz)', fontsize=12)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel(' PSD ($N \chi_a^2/$Hz)', fontsize=12)

    plt.savefig('Noise spectrum.png', dpi=600)


