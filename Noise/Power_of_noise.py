# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.special import wofz

def voigt_profile(x, sigma, gamma):
    """
    Calculate the Voigt profile.

    Parameters:
        x (array-like): The x-values at which to calculate the profile.
        sigma (float): The Gaussian standard deviation.
        gamma (float): The Lorentzian full-width at half-maximum.

    Returns:
        array-like: The Voigt profile values at the specified x-values.
    """
    z = (x + 1j * gamma) / (sigma * np.sqrt(2))
    v = wofz(z).imag / (sigma * np.sqrt(2 * np.pi))
    return v

omega = np.arange(0,100000, 1)
P=0.8
q=2*(3+P**2)/(1+P**2)
Omega = 400/q*4/2/np.pi
gammap = 1000/q/2/np.pi
gamman = 30000/2/np.pi
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

Gammap=2      
Gammad=0.5       
sigma=Gammad/(2*np.sqrt(2*np.log(2)))

def chia(delta_nva2):
    chia=-voigt_profile(delta_nva2,sigma,Gammap)/4-3/4*voigt_profile(delta_nva2+0.8,sigma,Gammap)
    return chia
def chib(delta_nva2):
    chib=5*voigt_profile(delta_nva2-6.8,sigma,Gammap)/4-1/4*voigt_profile(delta_nva2-6,sigma,Gammap)
    return chib

def chip(delta_nva2):
    return (eta*chia(delta_nva2)+chib(delta_nva2))/(eta+1)

def chim(delta_nva2):
    return (chia(delta_nva2)-chib(delta_nva2))/(eta+1)


Lorentzianpp = gammap / (gammap**2 + (omega - Omega) ** 2)/2/np.pi
Lorentzianpn = gammap / (gammap**2 + (omega + Omega) ** 2)/2/np.pi
Lorentziannp = gamman / (gamman**2 + (omega - Omega) ** 2)/2/np.pi
Lorentziannn = gamman / (gamman**2 + (omega + Omega) ** 2)/2/np.pi


plt.style.use(['science'])
with plt.style.context(['science']):
    plt.rc('font',family='Times New Roman')
    fig = plt.figure(figsize=(3.51, 4.8))
    ax_1  = fig.add_subplot(211)
    ax_1.loglog(omega, (chim(-100)**2*Fxn2*(Lorentziannp + Lorentziannn)+chip(-100)**2*Fx2*(Lorentzianpp + Lorentzianpn))/np.max((chim(-100)**2*Fxn2*(Lorentziannp + Lorentziannn)+chip(-100)**2*Fx2*(Lorentzianpp + Lorentzianpn))),linewidth=1.5)
    ax_1.loglog(omega, (chip(-100)**2*Fx2*(Lorentzianpp + Lorentzianpn))/np.max((chim(-100)**2*Fxn2*(Lorentziannp + Lorentziannn)+chip(-10)**2*Fx2*(Lorentzianpp + Lorentzianpn))),linewidth=1.5)
    ax_1.loglog(omega, (chim(100)**2*Fxn2*(Lorentziannp + Lorentziannn))/np.max((chim(-100)**2*Fxn2*(Lorentziannp + Lorentziannn)+chip(-100)**2*Fx2*(Lorentzianpp + Lorentzianpn))),linewidth=1.5)
    ax_1.set_xlabel('$\\nu$ (Hz)', fontsize=9)
    ax_1.set_ylabel(' $S^{\mathrm{in}}(\\nu)$ (arb. units)', fontsize=9)
    ax_1.tick_params(axis='x', labelsize='9' )
    ax_1.tick_params(axis='y', labelsize='9' )
    ax_1.legend(["$\\tilde{\phi}$", "$\chi_+ \\tilde F_{x+}$", "$\chi_- \\tilde F_{x-}$"], loc='lower left',
               prop={'size':9})
    ax_1.set_xlim([0,100000])



    P = np.arange(0,1, 0.01)
    eta=((3*P**2+5)/(1-P**2))
    Fp2=0
    Z=0
    for m in [2,1,0,-1,-2]:
        Fp2=Fp2+(6-m**2)/2*(1+P)**m/(1-P)**m
    for m in [1,0,-1]:
        Fp2=Fp2+(2-m**2)/2*(1+P)**m/(1-P)**m
    for m in [2,1,0,-1,-2,1,0,-1]:
        Z=Z+(1+P)**m/(1-P)**m
    Fp2= Fp2/Z*(eta-1)**2/(eta+1)**2

    Fn2=0
    for m in [2,1,0,-1,-2]:
        Fn2=Fn2+(6-m**2)/2*(1+P)**m/(1-P)**m
    for m in [1,0,-1]:
        Fn2=Fn2+eta**2*(2-m**2)/2*(1+P)**m/(1-P)**m
    Fn2= Fn2/Z*4/(eta+1)**2

    ax_2  = fig.add_subplot(212)
    ax_2.plot(P, 10*Fp2+10*Fn2,linewidth=1.5)
    ax_2.plot(P, 10*Fp2,linewidth=1.5)
    ax_2.plot(P, 10*Fn2,linewidth=1.5)
    ax_2.set_xlabel('$P$', fontsize=9)
    ax_2.set_ylabel(' Power (arb.units) ', fontsize=9)
    ax_2.tick_params(axis='x', labelsize='9' )
    ax_2.tick_params(axis='y', labelsize='9' )
    ax_2.set_xlim([0, 1])
    ax_1.text(30000, 0.6, '(a)',fontsize=8)
    ax_2.text(0.9, 14.5, '(b)',fontsize=8)
    # ax_2.legend(["P", "Narrow", "Broad"], loc='lower left',
    #            prop={'size':9})

    plt.savefig('Fig7_intrinsic.png', dpi=1000)
    plt.show()
