# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年01月19日
"""
import numpy as np
import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from scipy.linalg import *
import matplotlib.pyplot as plt
import scienceplots

P = np.linspace(0, 1, 100)
q = 2 * (3 + P ** 2) / (1 + P ** 2)
Q = 2 * (3 + P ** 4) / (1 + P ** 2) ** 2
qq = 2 * (19 + 26 * P ** 2 + 3 * P ** 4) / (3 + 10 * P ** 2 + 3 * P ** 4)
QQ = 2 * (57 + 44 * P ** 2 + 134 * P ** 4 + 12 * P ** 6 + 9 * P ** 8) / (3 + 10 * P ** 2 + 3 * P ** 4) ** 2
qqq = 2 * (11 + 35 * P ** 2 + 17 * P ** 4 + P ** 6) / (1 + 7 * P ** 2 + 7 * P ** 4 + P ** 6)
QQQ = 2 * (11 + 28 * P ** 2 + 99 * P ** 4 + 64 * P ** 6 + 49 * P ** 8 + 4 * P ** 10 +
           P ** 12) / (1 + 7 * P ** 2 + 7 * P ** 4 + P ** 6) ** 2

plt.style.use(['science', ])
with plt.style.context(['science']):
    plt.figure()
    p1, = plt.plot(P, Q, linestyle='solid')
    p2, = plt.plot(P, QQ, linestyle='dotted')
    p3, = plt.plot(P, QQQ, linestyle='dashdot')
    p4, = plt.plot(P, q, linestyle='solid')
    p5, = plt.plot(P, qq, linestyle='dotted')
    p6, = plt.plot(P, qqq, linestyle='dashdot')
    plt.legend([p1, p2, p3, p4, p5, p6], ["$Q_{3/2}$", "$Q_{5/2}$", "$Q_{7/2}$", "$q_{7/2}$", "$q_{3/2}$", "$q_{3/2}$"],
               loc='upper right', prop={'size': 10})
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('P', fontsize=12)
    plt.ylabel('Slowing down factor', fontsize=12)
    # my_y_ticks = np.arange(0, 1, 0.2)
    # plt.yticks(my_y_ticks)
    plt.savefig('3.png', dpi=600)

# plt.figure()
# plt.plot(t, C_1x2x)
# plt.figure()
# plt.plot(t, (np.array(C_1x2x)+np.array(C_1x1x)))

plt.show()