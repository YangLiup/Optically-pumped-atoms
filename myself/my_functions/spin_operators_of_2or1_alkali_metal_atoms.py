# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年12月20日
"""
import numpy as np
from qutip import *
from scipy.linalg import *


def spin_operators_of_2or1_alkali_metal_atoms(N, I):
    a = round(I + 1 / 2)
    b = round(I - 1 / 2)

    if N == 1:
        ax = block_diag(spin_Jx(a).full(), qzero(2 * b + 1).full())
        bx = block_diag(qzero(2 * a + 1).full(), spin_Jx(b).full())
        ay = block_diag(spin_Jy(a).full(), qzero(2 * b + 1).full())
        by = block_diag(qzero(2 * a + 1).full(), spin_Jy(b).full())
        az = block_diag(spin_Jz(a).full(), qzero(2 * b + 1).full())
        bz = block_diag(qzero(2 * a + 1).full(), spin_Jz(b).full())
        return ax, ay, az, bx, by, bz
    if N == 2:
        ax = block_diag(spin_Jx(a).full(), qzero(2 * b + 1).full())
        bx = block_diag(qzero(2 * a + 1).full(), spin_Jx(b).full())
        ay = block_diag(spin_Jy(a).full(), qzero(2 * b + 1).full())
        by = block_diag(qzero(2 * a + 1).full(), spin_Jy(b).full())
        az = block_diag(spin_Jz(a).full(), qzero(2 * b + 1).full())
        bz = block_diag(qzero(2 * a + 1).full(), spin_Jz(b).full())

        a1x = np.kron(ax, np.eye(2 * (a + b + 1)))
        a2x = np.kron(np.eye(2 * (a + b + 1)), ax)
        a1y = np.kron(ay, np.eye(2 * (a + b + 1)))
        a2y = np.kron(np.eye(2 * (a + b + 1)), ay)
        a1z = np.kron(az, np.eye(2 * (a + b + 1)))
        a2z = np.kron(np.eye(2 * (a + b + 1)), az)
        b1x = np.kron(bx, np.eye(2 * (a + b + 1)))
        b2x = np.kron(np.eye(2 * (a + b + 1)), bx)
        b1y = np.kron(by, np.eye(2 * (a + b + 1)))
        b2y = np.kron(np.eye(2 * (a + b + 1)), by)
        b1z = np.kron(bz, np.eye(2 * (a + b + 1)))
        b2z = np.kron(np.eye(2 * (a + b + 1)), bz)

        Fx = a1x + a2x + b1x + b2x
        Fy = a1y + a2y + b1y + b2y
        Fz = a1z + a2z + b1z + b2z
        return a1x, a2x, a1y, a2y, a1z, a2z, b1x, b2x, b1y, b2y, b1z, b2z, Fx, Fy, Fz
    if N == 3:
        ax = block_diag(spin_Jx(a).full(), qzero(2 * b + 1).full())
        bx = block_diag(qzero(2 * a + 1).full(), spin_Jx(b).full())
        ay = block_diag(spin_Jy(a).full(), qzero(2 * b + 1).full())
        by = block_diag(qzero(2 * a + 1).full(), spin_Jy(b).full())
        az = block_diag(spin_Jz(a).full(), qzero(2 * b + 1).full())
        bz = block_diag(qzero(2 * a + 1).full(), spin_Jz(b).full())

        a1x = np.kron(np.kron(ax, np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
        a2x = np.kron(np.kron(np.eye(2 * (a + b + 1)), ax), np.eye(2 * (a + b + 1)))
        a3x = np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))), ax)
        a1y = np.kron(np.kron(ay, np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
        a2y = np.kron(np.kron(np.eye(2 * (a + b + 1)), ay), np.eye(2 * (a + b + 1)))
        a3y = np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))), ay)
        a1z = np.kron(np.kron(az, np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
        a2z = np.kron(np.kron(np.eye(2 * (a + b + 1)), az), np.eye(2 * (a + b + 1)))
        a3z = np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))), az)

        b1x = np.kron(np.kron(bx, np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
        b2x = np.kron(np.kron(np.eye(2 * (a + b + 1)), bx), np.eye(2 * (a + b + 1)))
        b3x = np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))), bx)
        b1y = np.kron(np.kron(by, np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
        b2y = np.kron(np.kron(np.eye(2 * (a + b + 1)), by), np.eye(2 * (a + b + 1)))
        b3y = np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))), by)
        b1z = np.kron(np.kron(bz, np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
        b2z = np.kron(np.kron(np.eye(2 * (a + b + 1)), bz), np.eye(2 * (a + b + 1)))
        b3z = np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))), bz)

        Fx = a1x + a2x + a3x + b1x + b2x + b3x
        Fy = a1y + a2y + a3y + b1y + b2y + b3y
        Fz = a1z + a2z + a3z + b1z + b2z + b3z
        return a1x, a2x, a3x, a1y, a2y, a3y, a1z, a2z, a3z, b1x, b2x, b3x, b1y, b2y, b3y, b1z, b2z, b3z, Fx, Fy, Fz
