# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年01月19日
"""
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from pylab import mpl

P = np.linspace(0, 1, 100)
q = 2 * (3 + P ** 2) / (1 + P ** 2)
Q = 2 * (3 + P ** 4) / (1 + P ** 2) ** 2
qq = 2 * (19 + 26 * P ** 2 + 3 * P ** 4) / (3 + 10 * P ** 2 + 3 * P ** 4)
QQ = 2 * (57 + 44 * P ** 2 + 134 * P ** 4 + 12 * P ** 6 + 9 * P ** 8) / (3 + 10 * P ** 2 + 3 * P ** 4) ** 2
qqq = 2 * (11 + 35 * P ** 2 + 17 * P ** 4 + P ** 6) / (1 + 7 * P ** 2 + 7 * P ** 4 + P ** 6)
QQQ = 2 * (11 + 28 * P ** 2 + 99 * P ** 4 + 64 * P ** 6 + 49 * P ** 8 + 4 * P ** 10 +
           P ** 12) / (1 + 7 * P ** 2 + 7 * P ** 4 + P ** 6) ** 2

plt.style.use(['science'])
with plt.style.context(['science']):
    fig=plt.figure()
    ax = fig.add_subplot()
    p1, = ax.plot(P, q, linewidth='1')
    p2, = ax.plot(P, qq, linewidth='1' )
    p3, = ax.plot(P,qqq, linewidth='1' )
    ax.plot([],[])
    ax.plot([],[])
    ax.plot([],[])
    ax.plot([],[])
    p4, = ax.plot(P, Q,linestyle='dashed', linewidth='1')
    p5, = ax.plot(P, QQ,linestyle='dashed', linewidth='1' )
    p6, = ax.plot(P, QQQ,linestyle='dashed', linewidth='1' )

    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.tick_params(axis='both', which='minor', labelsize=9)

    plt.legend([p1,p2,p3], ["$I={3/2}$",  "$I={5/2}$", "$I={7/2}$"],
               loc='upper right', prop={'size': 9})
    plt.xlabel('$P$', fontsize=9)
    plt.ylabel('$q$, $Q$', fontsize=9)
    plt.ylim([0,23])

    plt.savefig('qq.png', dpi=600)

plt.show()



