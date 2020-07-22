import pandas as pnd
import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import argparse as argp



def parse_args():
	parser = argp.ArgumentParser(description='2D Convolutional Neural Network')
	parser.add_argument('-is','--inputSpectra', type=str, help='Input spectra to use', default="../dataset/sources/reflectances.csv")
	parser.add_argument('-li','--readLines', type=int, help='Read lines for the input spectra file', nargs='+', default=[-1])
	parser.add_argument('-us','--undersampling', type=float, help='Undersampling to apply at input spcetra',default=1.0)
	parser.add_argument('-md','--mode', type=str, help='Undersampling to apply at input spcetra',default='extract_abs')
	args = parser.parse_args()
	return args

args = parse_args()
print(args)
print('Reading CSV...')
if args.readLines != -1:
	data = pnd.read_csv(args.inputSpectra, low_memory=False, skiprows=lambda x: x not in args.readLines)
else:
	data = pnd.read_csv(args.inputSpectra, low_memory=False)
print(data)

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
if args.mode == 'extract_plot':
	for i in range(len(data_array)):
		temp = data_array[i].pop(0)
	# if args.undersampling != 1:
	indices = []
	columns = []
	i = 0
	length = len(data_array[0])
	print(length)
	indices_orig = range(length)
	while i < length:
		indices.append(i)
		columns.append(0.5*i+400)
		i += args.undersampling
	print(len(indices))

	x_1 = []
	for i in range(len(data_array)):
		x_1.append(np.interp(indices, indices_orig, data_array[i]))
	data_array = x_1
	print(args.undersampling)
	# columns = range(400, 1500, 0.5 * args.undersampling)
	# for i in range(len(columns)):
	# 	columns[i] -= 0.5*i
		# columns[i] = str(columns[i])
	# columns.insert(0, 'ID')
	# index = pnd.date_range("1 1 2000")

	# columns = {'Wavelength': columns}
	dtfr = pnd.DataFrame(data_array, columns=columns).transpose()
	# dtfr.insert(loc=0, column='Wavelength', value=columns)
	
	# dtfr
	# print(dtfr)
	# dtfr.rename({'ID': 'Wavelegth'})
	'''x='Wavelength (nm)', y='Reflectance', '''
	ax = sns.lineplot(data=dtfr, legend=False)
	# for ind, label in enumerate(plot_.get_xticklabels()):
	# 	if ind % 10 == 0:  # every 10th label is kept
	# 		label.set_visible(True)
	# 	else:
	# 		label.set_visible(False)
	plt.xlabel('Wavelength (nm)', fontsize=18)
	plt.ylabel('Reflectance', fontsize=18)
	plt.savefig('Reflactances.svg', bbox_inches='tight')
	plt.show()
	exit()
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
