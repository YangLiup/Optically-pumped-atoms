# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots


Rse=3*10**2 #kHz
Rsd=Rse/100
def comparison(omega_0,omega):
    delta=30 #GHz
    Delta=2 #GHz
    ga=(delta+2.56)/((delta+2.56)**2+(Delta/2)**2)
    gb=(delta-4.27)/((delta-4.27)**2+(Delta/2)**2)
    PSD1=np.zeros((0,))
    A=np.array([[-1/8*Rse-7/16*Rsd,0,-omega_0,5/8*Rse + 15/16*Rsd,0,0],
                [0,-1/8*Rse-7/16*Rsd,0,0,5/8*Rse + 15/16*Rsd,0],
                [omega_0,0,-1/8*Rse-7/16*Rsd,0,0,5/8*Rse + 15/16*Rsd],
                [1/8*Rse+3/16*Rsd,0,0,-5/8*Rse-11/16*Rsd,0,omega_0],
                [0,1/8*Rse+3/16*Rsd,0,0,-5/8*Rse-11/16*Rsd,0],
                [0,0,1/8*Rse+3/16*Rsd,-omega_0,0,-5/8*Rse-11/16*Rsd]])
    RXX0=np.diag([5/4,5/4,5/4,1/4,1/4,1/4])
    for Omega in omega:
        Sxx=-np.linalg.inv(A+1j*Omega*np.identity(6))@(A@RXX0+RXX0@A.T)@np.linalg.inv(A.T-1j*Omega*np.identity(6))
        S=ga**2*Sxx[2,2] + gb**2*Sxx[5,5]-ga*gb *(Sxx[2,5] + Sxx[5,2])
        PSD1=np.append(PSD1,S)


    gammap = 20/27*omega_0**2/Rse+Rsd/6
    gammam = 3/4*Rse

    gp=(5*ga-gb)/6
    gm=(ga+gb)/6

    Lorentzianpp = gammap / (gammap**2 + (omega - 2/3*omega_0) ** 2)
    Lorentzianpn = gammap / (gammap**2 + (omega +  2/3*omega_0) ** 2)
    Lorentzianmp = gammam / (gammam**2 + (omega -2/3*omega_0) ** 2)
    Lorentzianmn = gammam / (gammam**2 + (omega + 2/3*omega_0) ** 2)
    Lorentzianppd =(omega-2/3*omega_0) / (gammap**2 + (omega - 2/3*omega_0) ** 2)
    Lorentzianpnd =(omega+2/3*omega_0) / (gammap**2 + (omega + 2/3*omega_0) ** 2)
    c1=0.01
    c2=0.01
    revise=c1*Lorentzianppd-c2*Lorentzianpnd
    PSD2=3/2*(gp**2*(Lorentzianpp+Lorentzianpn)+5*gm**2*(Lorentzianmp+Lorentzianmn))
    return PSD1,PSD2,revise
omega1=np.arange(0,8,0.01)
omega2=np.arange(0,30,0.01)
omega3=np.arange(0,150,0.01)
PSD13,PSD23,revise3=comparison(Rse/4,omega3)
PSD12,PSD22,revise2=comparison(Rse/20,omega2)
PSD11,PSD21,revise1=comparison(Rse/100,omega1)

plt.style.use(['science'])
with plt.style.context(['science']):
    plt.rc('font',family='Times New Roman')
    fig = plt.figure(figsize=(3.46, 6.8))
    ax1= fig.add_subplot(313)
    p1,=ax1.plot(omega1, PSD11/np.max(PSD11))
    ax1.plot([], [])
    ax1.plot([], [])
    p2,=ax1.plot(omega1, PSD21/np.max(PSD11))
    ax1.plot([], [])
    ax1.plot([], [])
    ax1.plot([], [])
    ax1.plot([], [])
    p3,=ax1.plot(omega1, PSD21/np.max(PSD11)+revise1,linestyle='dashed',linewidth='0.7')
    p4,=ax1.plot(omega1, revise1,linestyle='dashed')

    # ax1.set_ylabel(' $S^{\mathrm{in}}(\\nu)$ (arb. units)', fontsize=9)
    ax1.set_xlabel('$\\nu$ (kHz)', fontsize=8)

    ax1.tick_params(axis='x', labelsize='9' )
    ax1.tick_params(axis='y', labelsize='9' )

    # ax1.set_yticks([-1,-0.5,0,0.5,1]) # 设置刻度
    # ax1.set_xlim(-25,25)


    
    ax2= fig.add_subplot(312)
    ax2.plot(omega2, PSD12/np.max(PSD12))
    ax2.plot([], [])
    ax2.plot([], [])
    ax2.plot(omega2, PSD22/np.max(PSD12))

    # ax2.set_ylabel(' $S^{\mathrm{in}}(\\nu)$ (arb. units)', fontsize=9)
    # ax2.set_xlabel('$\\nu$ (kHz)', fontsize=8)

    ax2.tick_params(axis='x', labelsize='9' )
    ax2.tick_params(axis='y', labelsize='9' )

    
    ax3= fig.add_subplot(311)
    ax3.plot(omega3, PSD13/np.max(PSD13))
    ax3.plot([], [])
    ax3.plot([], [])
    ax3.plot(omega3, PSD23/np.max(PSD13))

    # ax3.set_ylabel(' $S^{\mathrm{in}}(\\nu)$ (arb. units)', fontsize=9)
    # ax2.set_ylabel(' $S^{\mathrm{in}}(\\nu)$ (arb. units)', fontsize=9)
    # ax1.set_ylabel(' $S^{\mathrm{in}}(\\nu)$ (arb. units)', fontsize=9)

    fig.text(-0.01, 0.5, ' $S^{\mathrm{in}}(\\nu)$ (arb. units)', va='center', rotation='vertical',fontsize='9')

    # ax3.set_xlabel('$\\nu$ (kHz)', fontsize=8)
    ax2.legend([p1, p2], ["Ref.[19]","Eq.(63)"], loc='upper right',
               prop={'size': 8})
    ax1.legend([p3,p4], ["Eq.(63)$+$\n1st order \ncomponent","1st order \n component"], loc='upper right',
               prop={'size': 8})
    ax3.tick_params(axis='x', labelsize='9' )
    ax3.tick_params(axis='y', labelsize='9' )
    ax1.text(8*(0.9/2),1.08, '(c)',fontsize=8)
    ax2.text(30*(0.9/2),1.07, '(b)',fontsize=8)
    ax3.text(150*(0.9/2),1.07, '(a)',fontsize=8)






    plt.savefig('Fig10_comparison.png', dpi=1000)
plt.show()

