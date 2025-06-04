from sympy.physics.wigner import racah
def alpha_gt(Fg1,Fg2,Fe):
    Je,I=1/2,3/2
    alpha_gt=(-1)**(Fe-Fg2)*(2*Fe+1)*racah(1,1,Fg1,Fg2,1,Fe)/racah(1,1/2,Fg1,I,1/2,Fg2)*racah(Je,Fe,1/2,Fg1,I,1)*racah(Je,Fe,1/2,Fg2,I,1)
    return alpha_gt
print("alpha_gt(11)=",alpha_gt(1,1,2)+alpha_gt(1,1,1))
print("alpha_gt(22)=",alpha_gt(2,2,2)+alpha_gt(2,2,1))
print("alpha_gt(12)=",alpha_gt(1,2,2)+alpha_gt(1,2,1))
print("alpha_gt(21)=",alpha_gt(2,1,2)+alpha_gt(2,1,1))
