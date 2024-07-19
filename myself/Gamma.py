import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from sympy.physics.quantum.spin import Rotation
from sympy import pi
from my_functions.master_relaxationrate import masterequation
from scipy.linalg import *
import numpy as np
from my_functions.test7_copy import gammam
from matplotlib import ticker

I=3/2
a = round(I + 1 / 2)
b = round(I - 1 / 2)

PP1,DD1=masterequation(3/2,round(5e5),0.05,0.999)
PP2,DD2=masterequation(5/2,round(5e5),0.05,0.999)
PP3,DD3=masterequation(7/2,round(5e5),0.05,0.999)

z=gammam(3/2,PP1)
zz=gammam(5/2,PP2)
zzz=gammam(7/2,PP3)

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

# kappa1=(fp1/DD1-z/fm1)/2
# kappa2=(fp2/DD2-zz/fm2)/2
# kappa3=(fp3/DD3-zzz/fm3)/2

kappa1=(-z/fm1)
kappa2=(-zz/fm2)
kappa3=(-zzz/fm3)

Gammap1=4*eta1/(kappa1*16*(1+eta1)**3)
Gammap2=4*eta2/(kappa2*36*(1+eta2)**3)
Gammap3=4*eta3/(kappa3*64*(1+eta3)**3)

Gammam1=(kappa1*(1+eta1))
Gammam2=(kappa2*(1+eta2))
Gammam3=(kappa3*(1+eta3))

plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    fig = plt.figure(figsize=(3.35, 6))
    plt.rc('font',family='Times New Roman')
    ax1 = fig.add_subplot(311)
    ax1.plot(PP1,-(fp1/DD1+z/fm1)/(fp1/DD1-z/fm1))
    ax1.plot(PP2,-(fp2/DD2+zz/fm2)/(fp2/DD2-zz/fm2))
    ax1.plot(PP3,-(fp3/DD3+zzz/fm3)/(fp3/DD3-zzz/fm3))
    # ax1.plot(PP3,-fp3*fm3/DD3/zzz*0+1,linestyle='dotted')
    ax1.set_xlim([0.,0.99])
    # ax1.set_ylim([-0.1,0.1])
    ax1.set_ylabel('$\\varepsilon$', fontsize=11)
    ax1.tick_params(axis='x', labelsize='10' )
    ax1.tick_params(axis='y', labelsize='10' )
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax1.text(0.45, 0.082, '(a)',fontsize=8)
    ax1.axes.xaxis.set_ticklabels([])

    ax2 = fig.add_subplot(312)
    ax2.plot(PP1, DD1)
    ax2.plot(PP2, DD2)
    ax2.plot(PP3, DD3)
    ax2.plot([],[])
    ax2.plot([],[])
    ax2.plot([],[])
    ax2.plot([],[])
    ax2.plot(PP1, Gammap1,linestyle='dotted')
    ax2.plot(PP2, Gammap2,linestyle='dotted')
    ax2.plot(PP3, Gammap3,linestyle='dotted')
    ax2.text(0.45, 0.05, '(b)',fontsize=8)
    ax2.set_xlim([0.001,1])
    # ax2.set_ylim([0,0.1])

    ax2.set_ylabel('$\Gamma^+_t$ $(\omega_e^2/R_{\\rm{se}})$', fontsize=10)
    ax2.tick_params(axis='x', labelsize='10' )
    ax2.tick_params(axis='y', labelsize='10' )
    ax2.axes.xaxis.set_ticklabels([])

    ax3 = fig.add_subplot(313)
    p21=ax3.plot(PP1, -z)
    p22=ax3.plot(PP2, -zz)
    p23=ax3.plot(PP3, -zzz)
    ax3.plot([],[])
    ax3.plot([],[])
    ax3.plot([],[])
    ax3.plot([],[])
    ax3.plot(PP1, Gammam1,linestyle='dotted')
    ax3.plot(PP2, Gammam2,linestyle='dotted')
    ax3.plot(PP3, Gammam3,linestyle='dotted')
    ax1.legend(["$ I=3/2$", "$ I=5/2$", "$ I=7/2$"],
               loc='lower right', prop={'size': 9})
    ax3.set_xlabel('$P$', fontsize=10)
    ax3.set_ylabel('$\Gamma_t^- \\approx \Gamma_z^-$ ($R_{\\rm{se}}$) ', fontsize=10)
    ax3.tick_params(axis='x', labelsize='10' )
    ax3.tick_params(axis='y', labelsize='10' )
    ax3.text(0.45, 0.987, '(c)',fontsize=8)
    # p24=ax2.plot(PP, h*np.ones(bound),linestyle='dotted')
    # p25=ax2.plot(PP, hh*np.ones(bound),linestyle='dotted')
    # p26=ax2.plot(PP, hhh*np.ones(bound),linestyle='dotted')
    ax3.set_xlim([0,1])
    # ax3.set_ylim([0.5,1])

plt.savefig('Gamma_.png', dpi=1000)
plt.show()