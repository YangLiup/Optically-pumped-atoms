# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

omega = np.arange(-0 ,500, 0.01)
P=0.99
q=2*(3+P**2)/(1+P**2)
Omega = 400/q*4/2/np.pi
gammap = 1000/q/2/np.pi
# gamman = 30000/2/np.pi
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

chip2=(eta-1)**2/(eta+1)**2
chin2=4/(eta+1)**2
plt.style.use(['science'])
with plt.style.context(['science']):
    plt.rc('font',family='Times New Roman')
    fig1 = plt.figure()
    # p3, = plt.plot(omega, 10*Fx2*(Lorentzianpp + Lorentzianpn)-10*Fz/2*(Lorentzianpp - Lorentzianpn))
    # p1, = plt.plot(omega, 10*Fx2*(Lorentzianpp + Lorentzianpn))
    # p2, = plt.plot(omega, -10*Fz/2*(Lorentzianpp - Lorentzianpn))
    p1, = plt.plot(omega, (cxx*(Lorentzianpp + Lorentzianpn)+cxy*(Lorentzianpnd - Lorentzianppd))/np.max(cxx*(Lorentzianpp + Lorentzianpn)+cxy*(Lorentzianpnd - Lorentzianppd))
)
    p2, = plt.plot(omega, (cxx*(Lorentzianpp + Lorentzianpn))/np.max(cxx*(Lorentzianpp + Lorentzianpn)+cxy*(Lorentzianpnd - Lorentzianppd))
)
    p3, = plt.plot(omega, (cxx*Lorentzianpnd - cxy*(Lorentzianppd))/np.max(cxx*(Lorentzianpp + Lorentzianpn)+cxy*(Lorentzianpnd - Lorentzianppd))
)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    plt.legend([p1, p2, p3], ["Total", "Absorptive", "Dispersive"], loc='upper right',
               prop={'size': 9})

    plt.xlabel('$\\nu$ (Hz)', fontsize=9)
    plt.ylabel(' $S^{\mathrm{ls}}(\\nu)$ (arb. units)', fontsize=9)
    plt.xlim([0,500])
    # plt.ylim([-2,6])
    plt.savefig('Noise spectrum_lightshift.png', dpi=1000)
plt.show()

