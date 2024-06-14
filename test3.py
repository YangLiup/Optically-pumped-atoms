# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots


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


plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    fig1 = plt.figure()
    plt.plot(P, 10*Fp2)
    plt.plot(P, 10*Fn2)
    plt.plot(P, 10*Fp2+10*Fn2)
    plt.xlabel('$P$', fontsize=12)
    plt.ylabel('Power (arb.units) ',fontsize='12')
    plt.legend( ["$ \\varPhi_+$", "$ \\varPhi_-$", "$ \\varPhi_+$+$ \\varPhi_-$"],
               loc='upper left', prop={'size': 10})
    plt.ylim([0, 15.2])
    plt.xlim([0, 1])
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    # plt.xlabel('Frequency (Hz)', fontsize=12)
    # plt.ylabel(' PSD ($N \chi_a^2/$Hz)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('imag/Power.png', dpi=600)


