import os
import sys
import math
import numpy as np

glob_vars_file = '../gd_simulation/global_variables.py'
sys.path.append(os.path.dirname(os.path.expanduser(glob_vars_file)))
from global_variables import Comment, N, output_dir_path, MPl

def pw2(a):
	return a*a

def power_spectrum(PhiK,r1,r2):
	counter = 0
	avarageK = 0
	avarageFieldK2 = 0
	for i in range(N):
		for j in range(N):
			for k in range(N//2+1):
				if pw2(min(i,N-i)) + pw2(min(j,N-j)) + pw2(min(k,N-k)) >= pw2(r1) and pw2(min(i,N-i)) + pw2(min(j,N-j)) + pw2(min(k,N-k)) < pw2(r2):
					avarageK += math.sqrt(pw2(min(i,N-i)) + pw2(min(j,N-j)) + pw2(min(k,N-k)))
					avarageFieldK2 += abs(PhiK[i][j][k]*PhiK[i][j][k].conjugate())
					counter += 1
	avarageFieldK2 /= counter
	avarageK /= counter
	avarageK *= (2*math.pi)/(N*MPl)
	return [avarageK, avarageFieldK2]

def read_data(efold):
	PeriodicPhiFieldData = np.load(
		os.path.join(output_dir_path, f'PeriodicPhiFieldData_Efolds={efold.replace(".", ",")}{Comment}.npy'))
	PeriodicChiFieldData = np.load(
		os.path.join(output_dir_path, f'PeriodicChiFieldData_Efolds={efold.replace(".", ",")}{Comment}.npy'))
	Phi = np.reshape(PeriodicPhiFieldData, (N, N, N))
	Chi = np.reshape(PeriodicChiFieldData, (N, N, N))
	return Phi, Chi

def compute_spectrum(Phi, Chi):
	PhiK = np.fft.rfftn(Phi)
	ChiK = np.fft.rfftn(Chi)
	PhiResultTable = []
	ChiResultTable = []	
	for i in range(int(math.sqrt(3)*N/2 + 1)):
		PhiResultTable.append(power_spectrum(PhiK,i,i+1))
		ChiResultTable.append(power_spectrum(ChiK,i,i+1))
	return PhiResultTable, ChiResultTable

def main():
	EfoldsListData = open(os.path.join(output_dir_path, f'EfoldsBoxPlotList{Comment}.dat'), 'r')
	EfoldsTable = [v for v in EfoldsListData.read().split()]
	for efold in EfoldsTable:
		Phi, Chi = read_data(efold)
		PhiResultTable, ChiResultTable = compute_spectrum(Phi, Chi)
		PhiModesData = open(
			os.path.join(output_dir_path, f'ChiPowerSpectrumData_Efolds={efold.replace(".", ",")}{Comment}.dat'), 'w')
		ChiModesData = open(
			os.path.join(output_dir_path, f'ChiPowerSpectrumData_Efolds={efold.replace(".", ",")}{Comment}.dat'), 'w')
		for i in range(int(math.sqrt(3) * N / 2 + 1)):
			PhiModesData.write(repr(PhiResultTable[i][0]))
			PhiModesData.write('	')
			PhiModesData.write(repr(PhiResultTable[i][1]))
			PhiModesData.write('\n')
			ChiModesData.write(repr(ChiResultTable[i][0]))
			ChiModesData.write('	')
			ChiModesData.write(repr(ChiResultTable[i][1]))
			ChiModesData.write('\n')

if __name__ == '__main__':
	main()

	
	
	
	
