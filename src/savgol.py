import pandas as pnd
import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import argparse as argp
import json
import math
import os
from datetime import datetime

def get_JSON_data(file_in):
		with open(file_in) as data_file:
			data = json.load(data_file)
		return data

def parse_args():
	parser = argp.ArgumentParser(description='2D Convolutional Neural Network')
	parser.add_argument('-is','--inputSpectra', type=str, help='Input spectra to use', default="../dataset/sources/reflectances.csv")
	parser.add_argument('-li','--readLines', type=int, help='Read lines for the input spectra file', nargs='+', default=[-1])
	parser.add_argument('-us','--undersampling', type=float, help='Undersampling to apply at input spcetra',default=1.0)
	parser.add_argument('-md','--mode', type=str, help='Undersampling to apply at input spcetra',default='extract_abs')
	parser.add_argument('-org','--readOrganic', help='Read organic soil samples',action='store_true')
	parser.add_argument('-nm','--name', type=str, help='Extra name for experiment description',default="")
	args = parser.parse_args()
	return args

args = parse_args()
OUTPUT_PATH = '../output/diagrams'
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
datetime_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S_")+args.name+'/'
path_datetime = os.path.join(OUTPUT_PATH, datetime_str)
if not os.path.exists(path_datetime):
    os.mkdir(path_datetime)
print(args)
print('Reading CSV...')
if args.readOrganic:
	j_data = get_JSON_data('C:\diploma_thesis\py\\2D_CNN_Soilll\dataset\Mineral_Absorbances.json')
	# TODO: Use ID of dataset
	# data = pnd.read_csv(args.inputSpectra, low_memory=False, skiprows=j_data["read_lines"])
	# print([chunk['ID'] for chunk in iter_csv])
	# iter_csv = pnd.read_csv(args.inputSpectra, low_memory=False, iterator=True, usecols=range(0,4201), chunksize=1000)
	# data = pnd.concat([chunk[chunk['ID'][1] not in j_data["read_lines"]] for chunk in iter_csv])
	# print(j_data["read_lines"], len(j_data["read_lines"]))
	notReadLines = j_data["read_lines"]
	notReadLines.pop(0)
	print(notReadLines)
	for k in range(len(notReadLines)):
		notReadLines[k] += 1
	print(notReadLines)
	'''Indices for a json of Organic Absorbances'''
	readLines = []
	for k in range(0, 19039):
		if k not in notReadLines:
			readLines.append(k)
	print(len(readLines))
	data = pnd.read_csv(args.inputSpectra, low_memory=False, skiprows=notReadLines)
	data = data[2:7]
	# data_y = pnd.read_csv('..\dataset\properties.csv', low_memory=False, skiprows=notReadLines)
	iter_csv = pnd.read_csv('..\dataset\properties.csv', low_memory=False, iterator=True, chunksize=1000)
	# indices = [chunk['mineral'] == 'organic' for chunk in iter_csv]
	# for prop in indices:
	# 	print(prop['mineral'])
	# readLines = range(19038)[indices]
	# print(readLines)
	data_y = pnd.concat([chunk[chunk['mineral'] == 'organic'] for chunk in iter_csv])
	print(data_y)
elif args.readLines != -1:
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
print('Extracting plot values...')
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

	# columns  & columns}
	dtfr = pnd.DataFrame(data_array, columns=columns).transpose()
	# dtfr.insert(loc=0, column='Wavelength', value=columns)
	
	# dtfr
	# print(dtfr)
	# dtfr. & 'Wavelegth'})
	'''x='Wavelength (nm)', y='Reflectance', '''
	ax = sns.lineplot(data=dtfr, legend=False)
	# for ind, label in enumerate(plot_.get_xticklabels()):
	# 	if ind % 10 == 0:  # every 10th label is kept
	# 		label.set_visible(True)
	# 	else:
	# 		label.set_visible(False)
	plt.xlabel('Wavelength (nm)', fontsize=18)
	plt.ylabel('Reflectance', fontsize=18)
	ax.tick_params(axis='x', labelsize=18)
	ax.tick_params(axis='y', labelsize=18)
	plt.savefig('Reflactances.svg', bbox_inches='tight')
	plt.show()
	exit()
# for test_index in range(1,1000, 3):
# 	data_array[test_index].pop(0)
	# data2_array[test_index].pop(0)
	# data3_array[test_index].pop(0)
	# data_abs_array[test_index].pop(0)
elif args.mode=='extract_abs_sg1_implement':
	print(len(data_array))
	for i in range(len(data_array)):
		data_array[i].pop(0)
		ax = plt.plot(data_array[i])
		plt.xlabel('Wavelength index', fontsize=18)
		plt.ylabel('Reflectance', fontsize=18)
		plt.savefig(path_datetime+'/'+str(i)+'_Initial.svg', dpi=1000,bbox_inches='tight')
		plt.close()
		# plt.plot(data3_array[test_index])
		# plt.savefig('./SpectraTest/'+str(test_index)+'Initial_Reduced.png'),
		# plt.close()

		# # [ x*10 for x in data_array[0]]
		absorb = -np.log10(data_array[i])
		ax = plt.plot(absorb)
		plt.xlabel('Wavelength index', fontsize=18)
		plt.ylabel('Absorbance', fontsize=18)
		plt.savefig(path_datetime+'/'+str(i)+'_Abs.svg', dpi=1000,bbox_inches='tight')
		plt.close()

		# plt.plot(data_abs_array[test_index])
		# plt.savefig(path_datetime+'/'+str(test_index)+'Abs_Reduced.png')
		# plt.close()
		data_array[i] = signal.savgol_filter(-np.log10(data_array[i]), 101, 3).tolist()
		ax = plt.plot(data_array[i])
		plt.xlabel('Wavelength index', fontsize=18)
		plt.ylabel('Absorbance SG1', fontsize=18)
		plt.savefig(path_datetime+'/'+str(i)+'_Abs_SG1.svg', dpi=1000,bbox_inches='tight')
		plt.close()
		for x in range(len(data_array[i])):
			if x < 100:
				data_array[i][x] = 333 * pow(x / 100.0, 1.6) * data_array[i][x] + 0.5
			else:
				data_array[i][x] = 333 * data_array[i][x] + 0.5
			# data_array[i][x] = strform.format(data_array[i][x])
		ax = plt.plot(data_array[i])
		plt.xlabel('Wavelength index', fontsize=18)
		plt.ylabel('Absorbance SG1', fontsize=18)
		plt.savefig(path_datetime+'/'+str(i)+'_Abs_SG1_Fixed.svg', dpi=1000,bbox_inches='tight')
		plt.close()
	exit()
		# data_array[i].insert(0, temp)
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
