from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
eta = [0.1,
       0.1,
       0.1,
       0.01,
       0.001,
       0.01,
       0.001,
       0.01,
       0.001]
lambda_ = [0.1,
           0.01,
           0.001,
           0.1,
           0.1,
           0.01,
           0.01,
           0.001,
           0.001]
accuracy = [0.7783333333,
            0.76,
            0.7354166667,
            0.7783333333,
            0.78125,
            0.7691666667,
            0.7804166667,
            0.76625,
            0.78125]

fig = plt.figure()
ax = fig.add_subplot( projection='3d')
ax.scatter(eta, lambda_, zs=accuracy, s=200, label='True Position')
for x, y, z in zip(eta, lambda_, accuracy):
    text = '  (' + str(x) + ', ' + str(y) + ', ' + str(z) + ')'
    ax.text(x, y, z, text, zdir=(1, 1, 6), fontsize=12)



ax.set_xlabel('Eta', fontsize=16, labelpad=10)
ax.set_ylabel('Lambda', fontsize=16, labelpad=10)
ax.set_zlabel('Accuracy', fontsize=16, labelpad=10)
ax.set_xticks([0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
ax.set_yticks([0.001, 0.01, 0.02, 0.03, 0.04, 0.05,
              0.06, 0.07, 0.08, 0.09, 0.1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
for t in ax.zaxis.get_major_ticks():
    t.label.set_fontsize(14)


plt.show()
