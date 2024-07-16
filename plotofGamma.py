import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from sympy.physics.quantum.spin import Rotation
from sympy import pi
from NewGamma import Gamma
from scipy.linalg import *
import numpy as np
from my_functions.test7_copy import gammam
from matplotlib import ticker


DD1,z,PP1=Gamma(3/2,0.01,0.005,100000)
DD2,zz,PP2=Gamma(5/2,0.01,0.005,100000)
DD3,zzz,PP3=Gamma(7/2,0.01,0.005,100000)

q1=2*(3+PP1**2)/(1+PP1**2)
eta1=(q1+4)/(q1-4)
fp1 = (q1-4)**2*(q1+4)/(2*16*q1**3) #*(q1+4)/(q1-4)
fm1 = 2*q1/(q1-4)#*(q1-4)/(q1+4)

q2 = 2 * (19 + 26 * PP2 ** 2 + 3 * PP2 ** 4) / (3 + 10 * PP2 ** 2 + 3 * PP2 ** 4)
eta2=(q2+6)/(q2-6)
fp2 =  (q2-6)**2*(q2+6)/(2*36*q2**3)#*(q2+6)/(q2-6)
fm2 =2*q2/(q2-6)#*(q2-6)/(q2+6)

q3 = 2 * (11 + 35 * PP3 ** 2 + 17 * PP3 ** 4 + PP3 ** 6) / (1 + 7 * PP3 ** 2 + 7 * PP3 ** 4 + PP3 ** 6)
eta3=(q3+8)/(q3-8)
fp3 = (q3-8)**2*(q3+8)/(2*64*q3**3)#*(q3+8)/(q3-8)
fm3 =  2*q3/(q3-8)#*(q3-8)/(q3+8)

plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    fig = plt.figure(figsize=(3.35, 6))
    plt.rc('font',family='Times New Roman')
    ax1 = fig.add_subplot(411)
    ax1.plot(PP1,(fp1/DD1-z/fm1)/(fp1/DD1+z/fm1))
    ax1.plot(PP2,(fp2/DD2-zz/fm2)/(fp2/DD2+zz/fm2))
    ax1.plot(PP3,(fp3/DD3-zzz/fm3)/(fp3/DD3+zzz/fm3))
    # ax1.plot(PP3,-fp3*fm3/DD3/zzz*0+1,linestyle='dotted')
    ax1.set_xlim([0.,0.99])
    ax1.set_ylim([0.,0.002])
    # ax1.set_ylim([-0.1,0.1])
    ax1.set_ylabel('$\\varepsilon$', fontsize=10)
    ax1.tick_params(axis='x', labelsize='10' )
    ax1.tick_params(axis='y', labelsize='10' )
    ax1.set_yticks([0,0.001,0.002]) # 设置刻度

    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ax1.text(0.45, 0.00202, '(a)',fontsize=8)
    ax1.axes.xaxis.set_ticklabels([])

    ax2 = fig.add_subplot(412)
    ax2.plot(PP1,fp1/DD1)
    ax2.plot(PP2,fp2/DD2)
    ax2.plot(PP3,fp3/DD3)
    # ax1.plot(PP3,-fp3*fm3/DD3/zzz*0+1,linestyle='dotted')
    ax2.set_xlim([0.,0.99])
    # ax1.set_ylim([-0.1,0.1])
    ax2.set_ylabel('$\kappa(P)$', fontsize=10)
    ax2.tick_params(axis='x', labelsize='10' )
    ax2.tick_params(axis='y', labelsize='10' )
    ax2.text(0.45, 0.234, '(b)',fontsize=8)
    ax2.axes.xaxis.set_ticklabels([])

    ax3 = fig.add_subplot(413)
    ax3.plot(PP1, DD1)
    ax3.plot(PP2, DD2)
    ax3.plot(PP3, DD3)
    ax3.plot([],[])
    ax3.plot([],[])
    ax3.plot([],[])
    ax3.plot([],[])
    # ax2.plot(PP1, xiao,linestyle='dotted')
    # ax2.plot(PP2, Gammap2,linestyle='dotted')
    # ax2.plot(PP3, Gammap3,linestyle='dotted')
    ax3.text(0.45, 0.049, '(c)',fontsize=8)
    ax3.set_xlim([0.003,1])
    # ax2.set_ylim([0,0.1])

    ax3.set_ylabel('$\Gamma^+_t$ $(\omega_e^2/R_{\\rm{se}})$', fontsize=10)
    ax3.tick_params(axis='x', labelsize='10' )
    ax3.tick_params(axis='y', labelsize='10' )
    ax3.axes.xaxis.set_ticklabels([])

    ax4 = fig.add_subplot(414)
    p21=ax4.plot(PP1, z)
    p22=ax4.plot(PP2, zz)
    p23=ax4.plot(PP3, zzz)
    ax4.plot([],[])
    ax4.plot([],[])
    ax4.plot([],[])
    ax4.plot([],[])
    # ax3.plot(PP1, Gammam1,linestyle='dotted')
    # ax3.plot(PP2, Gammam2,linestyle='dotted')
    # ax3.plot(PP3, Gammam3,linestyle='dotted')
    ax1.legend(["$ I=3/2$", "$ I=5/2$", "$ I=7/2$"],
               loc='lower right', prop={'size': 9})
    ax4.set_xlabel('$P$', fontsize=10)
    ax4.set_ylabel('$\Gamma_t^- \\approx \Gamma_z^-$ ($R_{\\rm{se}}$) ', fontsize=10)
    ax4.tick_params(axis='x', labelsize='10' )
    ax4.tick_params(axis='y', labelsize='10' )
    ax4.text(0.45,0.928, '(d)',fontsize=8)
    # p24=ax2.plot(PP, h*np.ones(bound),linestyle='dotted')
    # p25=ax2.plot(PP, hh*np.ones(bound),linestyle='dotted')
    # p26=ax2.plot(PP, hhh*np.ones(bound),linestyle='dotted')
    ax4.set_xlim([0,1])
    # ax3.set_ylim([0.5,1])

plt.savefig('Gamma_.png', dpi=1000)
plt.show()