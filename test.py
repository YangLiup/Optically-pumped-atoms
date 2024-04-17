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
Gammaq = (q**2-16)/(2*q**3)

qq = 2 * (19 + 26 * P ** 2 + 3 * P ** 4) / (3 + 10 * P ** 2 + 3 * P ** 4)
Gammaqq = (qq**2-36)/(2*qq**3)
qqq = 2 * (11 + 35 * P ** 2 + 17 * P ** 4 + P ** 6) / (1 + 7 * P ** 2 + 7 * P ** 4 + P ** 6)
Gammaqqq = (qqq**2-64)/(2*qqq**3)


plt.style.use(['science' ])
with plt.style.context(['science']):
    plt.figure()
    p1, = plt.plot(P, Gammaq)
    p2, = plt.plot(P, Gammaqq)
    p3, = plt.plot(P, Gammaqqq)
    # p4, = plt.plot(P, q,color='dodgerblue', linestyle='dashdot')
    # p5, = plt.plot(P, qq,color='black', linestyle='dashdot')
    # p6, = plt.plot(P, qqq,color='olive', linestyle='dashdot')
    plt.legend([p1, p2, p3], ["$I={3/2}$", "$I={5/2}$", "$I={7/2}$"],
               loc='upper right', prop={'size': 10})
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('P', fontsize=12)
    plt.ylabel('$\Gamma^+(\omega_e^2/R_{se})$', fontsize=12)
    # my_y_ticks = np.arange(0, 1, 0.2)
    # plt.yticks(my_y_ticks)
    plt.savefig('linewidth.png', dpi=600)

# plt.figure()
# plt.plot(t, C_1x2x)
# plt.figure()
# plt.plot(t, (np.array(C_1x2x)+np.array(C_1x1x)))

plt.show()
