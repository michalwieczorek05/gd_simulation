glob_vars_file = '../gd_simulation/global_variables.py'
import os
import sys

sys.path.append(os.path.dirname(os.path.expanduser(glob_vars_file)))

from global_variables import Comment, N, output_dir_path

import numpy as np
import matplotlib.pylab as plt
from matplotlib import animation

EfoldsListData = open(f'{output_dir_path}EfoldsBoxPlotList{Comment}.dat','r')
EfoldsTable = [v for v in EfoldsListData.read().split()]
ChiPlotTable = []
Legend = []
z_constrain = 10
for x in EfoldsTable:
    PeriodicChiFieldData = np.load(f'{output_dir_path}PeriodicChiFieldData_Efolds={x.replace(".",",")}{Comment}.npy')
    ChiPlotTable.append(PeriodicChiFieldData[:,z_constrain])
    Legend.append(x)

for i in range(len(ChiPlotTable)):
    ChiPlotTable[i] = np.reshape(ChiPlotTable[i], (N, N))

fig = plt.figure()
plt.pcolormesh(ChiPlotTable[0])

plt.hold(True)
plt.pcolormesh(ChiPlotTable[0])

def animate(i):
    plt.pcolormesh(ChiPlotTable[i])
    plt.title('Efolds='+EfoldsTable[i])
    

anim = animation.FuncAnimation(fig, animate, frames = len(EfoldsTable), interval=1000, repeat = False)

anim.save(f'{output_dir_path}AnimationDensityPlot{Comment}z_constrain={z_constrain}.mp4',writer='ffmpeg')
plt.hold(False)
plt.show()