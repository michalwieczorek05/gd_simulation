import os
import sys
import numpy as np
import matplotlib.pylab as plt
from matplotlib import animation

glob_vars_file = '../gd_simulation/global_variables.py'
sys.path.append(os.path.dirname(os.path.expanduser(glob_vars_file)))
from global_variables import Comment, N, output_dir_path

def read_data(z_constrain=N // 2):
    EfoldsListData = open(os.path.join(output_dir_path, f'EfoldsBoxPlotList{Comment}.dat'),'r')
    EfoldsTable = [v for v in EfoldsListData.read().split()]
    ChiPlotTable = []
    for x in EfoldsTable:
        PeriodicChiFieldData = np.load(os.path.join(output_dir_path, f'PeriodicChiFieldData_Efolds={x.replace(".",",")}{Comment}.npy'))
        ChiPlotTable.append(PeriodicChiFieldData[:,z_constrain])
    for i in range(len(ChiPlotTable)):
        ChiPlotTable[i] = np.reshape(ChiPlotTable[i], (N, N))
    return ChiPlotTable, EfoldsTable

def make_animkation(ChiPlotTable, EfoldsTable, z_constrain=N // 2):
    fig = plt.figure()
    plt.pcolormesh(ChiPlotTable[0])
    plt.hold(True)
    plt.pcolormesh(ChiPlotTable[0])
    def animate(i):
        plt.pcolormesh(ChiPlotTable[i])
        plt.title('Efolds='+EfoldsTable[i])
    anim = animation.FuncAnimation(fig, animate, frames = len(EfoldsTable), interval=1000, repeat = False)
    anim.save(os.path.join(output_dir_path, f'AnimationDensityPlot{Comment}z_constrain={z_constrain}.mp4'),writer='ffmpeg')
    plt.hold(False)

def main():
    ChiPlotTable, EfoldsTable = read_data()
    make_animkation(ChiPlotTable, EfoldsTable)

if __name__ == "__main__":
   main()
