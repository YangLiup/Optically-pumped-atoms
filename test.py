import matplotlib.pyplot as plt
import quitp

# needs Axes3D object to activate the '3d' projection
from mpl_toolkits.mplot3d import Axes3D

fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='3d'))

ax.axis('square') # to get a nice circular plot

b1 = qutip.Bloch(fig=fig, axes=ax)
b1.add_states(qutip.sigmax()/2)
b1.zlabel = ['z', '']
b1.render(fig=fig, axes=ax) # render to the correct subplot 

# set title for the axis
ax.set_title('TITLE goes here', y=1.1, fontsize=20)

# You can anything else you want to the axis as well!
ax.annotate('TEXT', xy=(0.1, 0.9), xytext=(0.1, 0.7), xycoords='axes fraction',
            fontsize=15, color='r', ha='center',)

plt.show()