# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

omega = np.arange(0,800, 0.1)
q=5.2
Omega = 100/q*4/2/np.pi
gammap = 2*10/q/2/np.pi
gamman = 1000*(2/3+4/3/16-4*0.25/4/1.25)/2/np.pi


Fz = 1.3
Fx2 = 1.3
Fxn2=10
Lorentzianpp = gammap / (gammap**2 + (omega - Omega) ** 2)/2/np.pi
Lorentzianpn = gammap / (gammap**2 + (omega + Omega) ** 2)/2/np.pi
Lorentziannp = gamman / (gamman**2 + (omega - Omega) ** 2)/2/np.pi
Lorentziannn = gamman / (gamman**2 + (omega + Omega) ** 2)/2/np.pi

plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    fig1 = plt.figure()
    # p3, = plt.plot(omega, 10*Fx2*(Lorentzianpp + Lorentzianpn)-10*Fz/2*(Lorentzianpp - Lorentzianpn))
    # p1, = plt.plot(omega, 10*Fx2*(Lorentzianpp + Lorentzianpn))
    # p2, = plt.plot(omega, -10*Fz/2*(Lorentzianpp - Lorentzianpn))
    p1, = plt.plot(omega, np.log10(2*9/169*Fxn2*(Lorentziannp + Lorentziannn)+2*100/169*Fx2*(Lorentzianpp + Lorentzianpn)))
    p2, = plt.plot(omega, np.log10(Fx2*(Lorentzianpp + Lorentzianpn)))
    p3, = plt.plot(omega, np.log10(Fxn2*(Lorentziannp + Lorentziannn)))

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    plt.legend([p1, p2, p3], ["$\chi_b=-\chi_a$", "$\chi_b=\chi_a$", "$\chi_b=-\eta \chi_a$"], loc='upper right',
               prop={'size': 10})
    # plt.xlabel('Frequency (Hz)', fontsize=12)
    # plt.ylabel(' PSD ($ N \chi_+^2$/Hz)', fontsize=12)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel(' Log$[S(\\nu$)] (arb. units)', fontsize=12)
    plt.xlim([0,800])
    # plt.ylim([-2.5,0])
    plt.savefig('Noise spectrum.png', dpi=600)


