# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年12月24日
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

omega=np.arange(0,10,0.01)
cxy=omega/4/(omega**2+1)
cxx=omega**2/4/(omega**2+1)

fig=plt.figure(figsize=(7.5,6))
p1,=plt.plot(omega,cxx,linewidth='3')
p2,=plt.plot(omega,cxy,linewidth='3')
plt.xlabel('$\Omega \;(\gamma^+)$',size=15)
plt.ylabel('$c_{xx},c_{xy} \;(1/\gamma^+)$',size=15)
plt.legend([p1,p2],['$c_{xx}$','$c_{xy}$'],fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)  # 设置主刻度标签大小
plt.savefig('cxx.png', dpi=600)
plt.show()






