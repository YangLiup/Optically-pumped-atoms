from sympy import *
init_printing()
ax, bx, ay, by = symbols('a_x b_x a_y b_y', real=True)
eta = symbols('eta', real=True)
U=1/(eta+1)*Matrix([[eta,1,eta,1],[1,-1,1,-1],[-I*eta,-I,I*eta,I],[-I,I,I,-I]])
X=Matrix([ax,bx,ay,by])
Y=U*X 
pprint(Y)
