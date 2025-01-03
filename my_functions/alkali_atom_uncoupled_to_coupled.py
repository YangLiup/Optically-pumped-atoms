# ,-*-,coding:utf-8,-*-
"""
作者：DELL
日期：2023年12月18日
"""
# This function provides the transformation from uncoupled  representation to coupled  representation for Rb87
import sys
sys.path.append(r"D:\python\pythonProject\Optically_pumped_atoms\my_functions")
import numpy as np
from sympy.physics.quantum.spin import JzKet, couple, JzKetCoupled, uncouple
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.represent import represent
from sympy import S


def alkali_atom_uncoupled_to_coupled(double_I):
    I = S(double_I) / 2
    s = S(1) / 2
    b = round(I - 1 / 2)
    N = round(I * 2 + 1) * round(s * 2 + 1)
    U = np.empty([N, N])
    # U = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
    #               [0, np.sqrt(1 / 4), np.sqrt(3 / 4), 0, 0, 0, 0, 0],
    #               [0, 0, 0, np.sqrt(1 / 2), np.sqrt(1 / 2), 0, 0, 0],
    #               [0, 0, 0, 0, 0, np.sqrt(3 / 4), np.sqrt(1 / 4), 0],
    #               [0, 0, 0, 0, 0, 0, 0, 1],
    #               [0, np.sqrt(3 / 4), -np.sqrt(1 / 4), 0, 0, 0, 0, 0],
    #               [0, 0, 0, np.sqrt(1 / 2), -np.sqrt(1 / 2), 0, 0, 0],
    #               [0, 0, 0, 0, 0, np.sqrt(1 / 4), -np.sqrt(3 / 4), 0]]).T.conjugate()
    i = 0
    for mI in np.arange(I, -I - 1, -1):
        for ms in np.arange(s, -s - 1, -1):
            stupid = np.array(represent(couple(TensorProduct(JzKet(I, mI), JzKet(s, ms)))))
            for j in np.arange(0, b**2, 1):
                stupid = np.delete(stupid, 0, 0)

            U[:, [i]] = stupid
            i = i + 1
    for i in np.arange(0, 2 * b + 1, 1):
        U = np.vstack((U, U[[0], :]))
        U = np.delete(U, 0, 0)

    return U.T



# alkali_atom_uncoupled_to_coupled(5)
