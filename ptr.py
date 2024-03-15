# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年12月17日
"""
# this function is used to take partial trace for two spins. N-dimension is partial-traced#
import numpy as np
def ptr(Rho, N, D):
    a = np.zeros((D, D))
    for j in np.arange(0, D, 1):
        for k in np.arange(0, D, 1):
            S = 0
            for l in np.arange(0, N, 1):
                S = S + Rho[j * N + l, k * N + l]
            a[j, k] = S
    return a
