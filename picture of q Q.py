# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年01月19日
"""
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

P = np.linspace(0, 1, 100)
q = 2 * (3 + P ** 2) / (1 + P ** 2)
Q = 2 * (3 + P ** 4) / (1 + P ** 2) ** 2
qq = 2 * (19 + 26 * P ** 2 + 3 * P ** 4) / (3 + 10 * P ** 2 + 3 * P ** 4)
QQ = 2 * (57 + 44 * P ** 2 + 134 * P ** 4 + 12 * P ** 6 + 9 * P ** 8) / (3 + 10 * P ** 2 + 3 * P ** 4) ** 2
qqq = 2 * (11 + 35 * P ** 2 + 17 * P ** 4 + P ** 6) / (1 + 7 * P ** 2 + 7 * P ** 4 + P ** 6)
QQQ = 2 * (11 + 28 * P ** 2 + 99 * P ** 4 + 64 * P ** 6 + 49 * P ** 8 + 4 * P ** 10 +
           P ** 12) / (1 + 7 * P ** 2 + 7 * P ** 4 + P ** 6) ** 2

plt.style.use(['science' ,'nature'])
with plt.style.context(['science','nature']):
    plt.figure()
    p1, = plt.plot(P, q,linestyle='solid')
    p2, = plt.plot(P, qq,linestyle='solid' )
    p3, = plt.plot(P,qqq,linestyle='solid')
    plt.plot([],[])
    plt.plot([],[])
    plt.plot([],[])
    plt.plot([],[])
    p4, = plt.plot(P, Q,linestyle='dashed')
    p5, = plt.plot(P, QQ,linestyle='dashed')
    p6, = plt.plot(P, QQQ,linestyle='dashed')
    plt.legend([p1,p2,p3], ["$I={3/2}$",  "$I={5/2}$", "$I={7/2}$"],
               loc='upper right', prop={'size': 10})

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('$P$', fontsize=11)
    plt.ylabel('$q$, $Q$', fontsize=11)
    # my_y_ticks = np.arange(0, 1, 0.2)
    # plt.yticks(my_y_ticks)
    plt.ylim([0,23])
    plt.xlim([0,1])

    plt.savefig('qq.png', dpi=600)


# plt.figure()
# plt.plot(t, C_1x2x)
# plt.figure()
# plt.plot(t, (np.array(C_1x2x)+np.array(C_1x1x)))


