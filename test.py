# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年01月19日
"""
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

P = np.linspace(0, 1, 1000)
q = 2 * (3 + P ** 2) / (1 + P ** 2)
L=P**2/4/(P**2+1)
Gammaq = (q**2-16)/(2*q**3)
GammaNq = 2/3+4/3/16-4*L

qq = 2 * (19 + 26 * P ** 2 + 3 * P ** 4) / (3 + 10 * P ** 2 + 3 * P ** 4)
LL=8*P**2*(3*P**2+7)/27/(3*P**4+10*P**2+3)
Gammaqq = (qq**2-36)/(2*qq**3)
GammaNqq = 2/3+4/3/36-4*L

qqq = 2 * (11 + 35 * P ** 2 + 17 * P ** 4 + P ** 6) / (1 + 7 * P ** 2 + 7 * P ** 4 + P ** 6)
LLL=P**2*(5*P**4+30*P**2+21)/16/(P**6+7*P**4+7*P**2+1)
Gammaqqq = (qqq**2-64)/(2*qqq**3)
GammaNqqq = 2/3+4/3/64-4*L



plt.style.use(['science' ])
with plt.style.context(['science']):
    plt.figure()
    # p1, = plt.plot(P, Gammaq)
    # p2, = plt.plot(P, Gammaqq)
    # p3, = plt.plot(P, Gammaqqq)
    # # p4, = plt.plot(P, q,color='dodgerblue', linestyle='dashdot')
    # # p5, = plt.plot(P, qq,color='black', linestyle='dashdot')
    # # p6, = plt.plot(P, qqq,color='olive', linestyle='dashdot')
    # plt.legend([p1, p2, p3], ["$I={3/2}$", "$I={5/2}$", "$I={7/2}$"],
    #            loc='upper right', prop={'size': 10})
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    # plt.xlabel('P', fontsize=12)
    # plt.ylabel('', fontsize=12)



    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(P, Gammaq)
    ax1.plot(P, Gammaqq)
    ax1.plot(P, Gammaqqq)
    ax1.set_ylabel('$\Gamma^+\;(\omega_e^2/R_{se})$')
    plt.legend( ["$I={3/2}$", "$I={5/2}$", "$I={7/2}$"],
               loc='upper right', prop={'size': 10})
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(P, GammaNq,linestyle='dashed')
    ax2.plot(P, GammaNqq,linestyle='dashed')
    ax2.plot(P, GammaNqqq,linestyle='dashed')
    ax2.set_ylabel('$\Gamma^-\;(R_{se})$')
    # my_y_ticks = np.arange(0, 1, 0.2)
    # plt.yticks(my_y_ticks)
    plt.savefig('linewidth.png', dpi=600)

# plt.figure()
# plt.plot(t, C_1x2x)
# plt.figure()
# plt.plot(t, (np.array(C_1x2x)+np.array(C_1x1x)))

plt.show()
