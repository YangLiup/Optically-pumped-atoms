# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

omega = np.arange(0,100000, 0.01)
P=2/3
q=2*(3+P**2)/(1-P**2)
Omega = 400/q*4/2/np.pi
gammap = 2*100/q/2/np.pi
gamman = 10000*(2/3+4/3/16-P**2/(1+P**2))/2/np.pi
eta=(3*P**2+5)/(1-P**2)
Fz = 1/2 *q*P
az=eta/(1+eta)*Fz
bz=1/(1+eta)*Fz 
Fxa2=0
Fxb2=0
Z=0
for m in [2,1,0,-1,-2]:
    Fxa2=Fxa2+(6-m**2)/2*(1+P)**m/(1-P)**m
for m in [1,0,-1]:
    Fxb2=Fxb2+(2-m**2)/2*(1+P)**m/(1-P)**m
for m in [2,1,0,-1,-2,1,0,-1]:
    Z=Z+(1+P)**m/(1-P)**m
Fxa2= Fxa2/Z
Fxb2= Fxb2/Z

Fx2= Fxa2+Fxb2
Fxn2= Fxa2+eta**2*Fxb2
Lorentzianpp = gammap / (gammap**2 + (omega - Omega) ** 2)/2/np.pi
Lorentzianpn = gammap / (gammap**2 + (omega + Omega) ** 2)/2/np.pi
Lorentziannp = gamman / (gamman**2 + (omega - Omega) ** 2)/2/np.pi
Lorentziannn = gamman / (gamman**2 + (omega + Omega) ** 2)/2/np.pi

Lorentzianpp0 = gammap / (gammap**2 )/2/np.pi
Lorentzianpn0 = gammap / (gammap**2 + (2*Omega) ** 2)/2/np.pi
Lorentziannp0 = gamman / (gamman**2 )/2/np.pi
Lorentziannn0 = gamman / (gamman**2 + (2* Omega) ** 2)/2/np.pi

chip2=(eta-1)**2/(eta+1)**2
chin2=4/(eta+1)**2
plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    plt.rc('font',family='Times New Roman')
    fig1 = plt.figure()
    # p3, = plt.plot(omega, 10*Fx2*(Lorentzianpp + Lorentzianpn)-10*Fz/2*(Lorentzianpp - Lorentzianpn))
    # p1, = plt.plot(omega, 10*Fx2*(Lorentzianpp + Lorentzianpn))
    # p2, = plt.plot(omega, -10*Fz/2*(Lorentzianpp - Lorentzianpn))
    p1, = plt.loglog(omega, 100*(chin2**Fxn2*(Lorentziannp + Lorentziannn)+chip2*Fx2*(Lorentzianpp + Lorentzianpn)))
    p2, = plt.loglog(omega, 100*(Fx2*(Lorentzianpp + Lorentzianpn)))
    p3, = plt.loglog(omega, 100*(Fxn2*(Lorentziannp + Lorentziannn)))

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    plt.legend([p1, p2, p3], ["$\chi_b=-\chi_a$", "$\chi_b=\chi_a$", "$\chi_b=-\eta \chi_a$"], loc='center left',
               prop={'size':9})
    # plt.xlabel('Frequency (Hz)', fontsize=12)
    # plt.ylabel(' PSD ($ N \chi_+^2$/Hz)', fontsize=12)
    plt.xlabel('$\\nu$ (Hz)', fontsize=10)
    plt.ylabel(' $S(\\nu)$ (arb. units)', fontsize=10)
    plt.xlim([0,100000])
    # plt.ylim([-2,6])
    plt.savefig('imag/Noise spectrum.png', dpi=1000)


