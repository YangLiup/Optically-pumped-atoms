# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

omega = np.arange(-100,100, 0.01)
P=2/3
q=2*(3+P**2)/(1-P**2)
Omega = 300/q*4/2/np.pi
gammap = 300/q/2/np.pi
gamman = 10000*(2/3+4/3/16-P**2/(1+P**2))/2/np.pi
eta=(3*P**2+5)/(1-P**2)
Fz = 1/2 *q*P
az=eta/(1+eta)*Fz
bz=1/(1+eta)*Fz

cxx= Omega**2*az**2/(4*gammap*(gammap**2+Omega**2))
cxy= cxx*gammap/Omega
Lorentzianpp = gammap / (gammap**2 + (omega - Omega) ** 2)/2/np.pi
Lorentzianpn = gammap / (gammap**2 + (omega + Omega) ** 2)/2/np.pi
Lorentzianppd =(omega-Omega) / (gammap**2 + (omega - Omega) ** 2)/2/np.pi
Lorentzianpnd = (omega+Omega) / (gammap**2 + (omega + Omega) ** 2)/2/np.pi

Lorentzianpp0 = gammap / (gammap**2 )/2/np.pi
Lorentzianpn0 = gammap / (gammap**2 + (2*Omega) ** 2)/2/np.pi
Lorentzianppd0 =0
Lorentzianpnd0 = (2*Omega) / (gammap**2 + (2* Omega) ** 2)/2/np.pi

chip2=(eta-1)**2/(eta+1)**2
chin2=4/(eta+1)**2
plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    plt.rc('font',family='Times New Roman')
    fig1 = plt.figure()
    # p3, = plt.plot(omega, 10*Fx2*(Lorentzianpp + Lorentzianpn)-10*Fz/2*(Lorentzianpp - Lorentzianpn))
    # p1, = plt.plot(omega, 10*Fx2*(Lorentzianpp + Lorentzianpn))
    # p2, = plt.plot(omega, -10*Fz/2*(Lorentzianpp - Lorentzianpn))
    p1, = plt.plot(omega, (cxx*(Lorentzianpp + Lorentzianpn)+cxy*(Lorentzianpnd - Lorentzianppd))/(cxx*(Lorentzianpp0 + Lorentzianpn0)+cxy*(Lorentzianpnd0 - Lorentzianppd0))
)
    p2, = plt.plot(omega, (cxx*(Lorentzianpp + Lorentzianpn))/(cxx*(Lorentzianpp0 + Lorentzianpn0)+cxy*(Lorentzianpnd0 - Lorentzianppd0))
)
    p3, = plt.plot(omega, (cxx*Lorentzianpnd - cxy*(Lorentzianppd))/(cxx*(Lorentzianpp0 + Lorentzianpn0)+cxy*(Lorentzianpnd0 - Lorentzianppd0))
)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    plt.legend([p1, p2, p3], ["Total", "Absorptive", "Dispersive"], loc='upper right',
               prop={'size': 9})

    plt.xlabel('$\\nu$ (Hz)', fontsize=10)
    plt.ylabel(' $S(\\nu)$ (arb. units)', fontsize=10)
    plt.xlim([0,100])
    # plt.ylim([-2,6])
    plt.savefig('imag/Noise spectrum_lightshift.png', dpi=1000)


