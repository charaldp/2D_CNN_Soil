import pandas as pnd
import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt

print('Reading CSV...')
data = pnd.read_csv('./reflectances.csv', low_memory=False)
# data2 = pnd.read_csv('./Absorbances_SG1_reduced.csv', low_memory=False)
# data = pnd.read_csv('./reflectances.csv', nrows=1000)
# data2 = pnd.read_csv('./Absorbances_SG1_reduced.csv', nrows=10)
# data_abs = pnd.read_csv('./Absorbances_reduced.csv', nrows=10)
# data3 = pnd.read_csv('./Reflectances_reduced.csv', nrows=10)
print('Converting to np array...')
data_array = data.values
# data2_array = data2.values.tolist()
# data3_array = data3.values.tolist()
# data_abs_array = data_abs.values.tolist()
print('Converting to np list...')
data_array = data_array.tolist()
print('Extracting savgol filter values...')
data = None
# for test_index in range(1,1000, 3):
# 	data_array[test_index].pop(0)
	# data2_array[test_index].pop(0)
	# data3_array[test_index].pop(0)
	# data_abs_array[test_index].pop(0)

	# plt.plot(data_array[test_index])
	# plt.savefig('./SpectraTest/'+str(test_index)+'Initial.png')
	# plt.close()

	# plt.plot(data3_array[test_index])
	# plt.savefig('./SpectraTest/'+str(test_index)+'Initial_Reduced.png')
	# plt.close()

	# # [ x*10 for x in data_array[0]]
	# absorb = -np.log10(data_array[test_index])
	# plt.plot(absorb)
	# plt.savefig('./SpectraTest/'+str(test_index)+'Abs.png')
	# plt.close()

	# plt.plot(data_abs_array[test_index])
	# plt.savefig('./SpectraTest/'+str(test_index)+'Abs_Reduced.png')
	# plt.close()

	# data_array[test_index] = signal.savgol_filter(absorb, window_length=101, polyorder=3, deriv=1, mode='nearest').tolist()
	# for x in range(100):
	# 	data_array[test_index][x] *= pow(x / 100.0, 1.6)
	# print(data_array[0])
	# print(data2_array[0])

	# plt.plot(data_array[test_index])
	# plt.savefig('./SpectraTest/'+str(test_index)+'Abs_SG1.png')
	# plt.close()

	# plt.plot(data2_array[test_index])
	# plt.savefig('./SpectraTest/'+str(test_index)+'Abs_SG1_Reduced.png')
	# plt.close()
# exit()

ndigits = 5
strform = "{:."+str(ndigits)+"f}"
for i in range(len(data_array)):
	temp = data_array[i].pop(0)
	data_array[i] = signal.savgol_filter(-np.log10(data_array[i]), 101, 3, 1).tolist()
	for x in range(len(data_array[i])):
		if x < 100:
			data_array[i][x] = 333 * pow(x / 100.0, 1.6) * data_array[i][x] + 0.5
		else:
			data_array[i][x] = 333 * data_array[i][x] + 0.5
		data_array[i][x] = strform.format(data_array[i][x])
	data_array[i].insert(0, temp)
	print(i)
print('Creating DataFrame for extraction...')
columns = range(4200)
for i in range(len(columns)):
	columns[i] -= 0.5*i
	columns[i] = str(columns[i] + 400)
columns.insert(0, 'ID')

dtfr = pnd.DataFrame(data_array, columns=columns)
data_array = None
print('Extracting CSV...')
# for i in range(len(index_list)):
# 	index_list[i] = str(index_list[i])

dtfr.to_csv('./absorbances_sg1.csv',index=False)
