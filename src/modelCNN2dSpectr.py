import tensorflow
tensorflow_ver = tensorflow.__version__
if tensorflow_ver[0] == '2':
	# Tensorflow 2.X.X
	from tensorflow import keras
	from tensorflow.keras.layers import *
	from tensorflow.keras.activations import *
	from tensorflow.keras.initializers import *
	from tensorflow.keras.models import Sequential,Model,load_model
	from tensorflow.keras.optimizers import Adam,Adamax
	from tensorflow.keras.callbacks import Callback, LambdaCallback, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
elif tensorflow_ver[0] == '1':
	# Tensorflow 1.X.X
	BACKEND=tensorflow
	import keras
	from keras.layers import *
	from keras.activations import *
	from keras.models import Sequential,Model,Input,load_model
	from keras.optimizers import Adam,Adamax
	from keras.callbacks import Callback, LambdaCallback, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
	from keras.initializers import *
	from keras import backend as K

import sys,math,random,os,argparse, scipy
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pnd
from time import time



class TrainingResetCallback(Callback):
	def __init__(self):
		Callback.__init__(self)
		# self.logs = []
		self.val_loss = []

	# def on_epoch_begin(self, epoch, logs={}):
	 # 		self.starttime=time()

	def on_epoch_end(self, epoch, logs={}):
		# self.logs.append(time()-self.starttime)
		self.val_loss.append(logs.get('val_loss'))
		stdev = np.std(self.val_loss)
		mean = np.mean(self.val_loss)
		print(self.val_loss)
		print('Stdev = %f' % stdev)
		if len(self.val_loss) > 2 and stdev < 0.01 and mean > 0.1:
			self.model.stop_training = True


class PrintModelCallback(Callback):
	def __init__(self):
		Callback.__init__(self)
		# self.logs = []
		self.checkedLayers = [0,4,7,11,14,19]
		# for layer in self.checkedLayers:
		# 	print(self.model.layers[layer])

	# def on_epoch_begin(self, epoch, logs={}):
 	# 		self.starttime=time()

	def on_epoch_end(self, batch, epoch, logs={}):
		for layer in self.checkedLayers:
			print(self.model.layers[layer].get_weights())


def coeff_determination(y_true, y_pred):
	from keras import backend as K
	SS_res =  K.sum(K.square( y_true-y_pred ))
	SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
	return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def lucas_soil_properties():
	return {
	 	'min' : {
	 		'OC':  0,
	 		'CEC': 0,
	 		'Clay': 0,
	 		'Sand': 1,
	 		'pH': 3.21,
	 		'N': 0
	 	},
	 	'max' : {
	 		'OC':  586.8,
	 		'CEC': 234,
	 		'Clay': 79,
	 		'Sand': 99,
	 		'pH': 10.08,
	 		'N': 38.6
	 	},
	 	'mean' : {
	 		'OC': 50.00,
			'CEC': 15.76,
			'Clay': 18.88,
			'Sand': 42.88,
			'pH': 6.20,
			'N': 2.92
	 	},
	 	'median': {
		 	'OC': 20.80,
			'CEC': 12.40,
			'Clay': 17.00,
			'Sand': 42.00,
			'pH': 6.21,
			'N': 1.70
	 	},
	 	'st.dev.': {
	 		'OC': 91.31,
			'CEC': 14.48,
			'Clay': 13.00,
			'Sand': 26.11,
			'pH': 1.35,
			'N': 3.76
	 	},
	 	'skewness': {
		 	'OC': 3.67,
			'CEC': 4.24,
			'Clay': 0.91,
			'Sand': 0.19,
			'pH': 0.08,# Minus !!!!
			'N': 3.76
	 	}
 	}

def computeErrors(valy, pred, stand=False):
    # Compute RMSE
    sumSquaresRes = 0
    for i in range(len(pred)):
        #piirint("X=%s, Predicted=%s" % (valy[i], pred[i]))
        sumSquaresRes = sumSquaresRes + pow(valy[i] - pred[i], 2)

    rmse = math.sqrt(sumSquaresRes / float(len(pred)))

    if stand:
        return rmse

    # Compute R^2
    meanY = sum(valy) / float(len(valy))
    totSumSquares = 0
    for i in range(len(pred)):
        totSumSquares = totSumSquares + pow(valy[i] - meanY, 2)

    determ = 1 - (sumSquaresRes / float(totSumSquares))

    # Compute RPIQ
    iqr = np.percentile(valy, 75) - np.percentile(valy, 25)
    rpiq = iqr / float(rmse)

    return rmse, determ, rpiq


def getInputShape(spectra, v_to_h_ratio):
	mi = int(v_to_h_ratio * 100)
	nover = int(v_to_h_ratio * 50)
	window = signal.hann(M = mi)
	[x, t, spec] = signal.spectrogram(x = np.array(spectra), fs = 1,window = window, nperseg = mi, noverlap = nover)
	return spec.shape

def spectraToSpectrogram(x_in_spectra, mode, v_to_h_ratio, input_shape):
	if mode=='minus_1_1':
		num = 25
	elif mode=='one_zero':
		num = 50
	mi = int(v_to_h_ratio * 100)
	nover = int(v_to_h_ratio * 50)
	window = signal.hann(M = mi)
	x_in_spectra = np.array(x_in_spectra)
	x_spectrogram = np.empty(shape=(x_in_spectra.shape[0],input_shape[0],input_shape[1],1))
	for i in range(x_in_spectra.shape[0]):
		[x, t, spec] = signal.spectrogram(x = x_in_spectra[i], fs = 1,window = window, nperseg = mi, noverlap = nover)
		# print(spec)
		# a = pnd.DataFrame(spec)
		# a.to_csv('../output/test.csv')
		x_spectrogram[i] = (np.log(np.abs(spec.reshape(input_shape[0],input_shape[1], 1))) + num) / num
		# exit()
	return x_spectrogram

def spectraToSpectrogramMulti(x_in_spectra, mode, v_to_h_ratio, input_shape, props):
	if mode=='minus_1_1':
		num = 25
	elif mode=='one_zero':
		num = 50
	mi = int(v_to_h_ratio * 100)
	nover = int(v_to_h_ratio * 50)
	window = signal.hann(M = mi)
	x_in_spectra = np.array(x_in_spectra)
	x_spectrogram = np.empty(shape=(x_in_spectra.shape[0],input_shape[0],input_shape[1],1))
	for i in range(x_in_spectra.shape[0]):
		[x, t, spec] = signal.spectrogram(x = x_in_spectra[i], fs = 1,window = window, nperseg = mi, noverlap = nover)
		# print(spec)
		# a = pnd.DataFrame(spec)
		# a.to_csv('../output/test.csv')
		x_spectrogram[i] = (np.log(np.abs(spec.reshape(input_shape[0],input_shape[1], 1))) + num) / num
		# exit()
	return x_spectrogram

def extractSpectrogram(x_in_spectra, path, mode, v_to_h_ratio, input_shape ):
	if mode=='minus_1_1':
		num = 25
	elif mode=='one_zero':
		num = 50
	mi = int(v_to_h_ratio * 100)
	nover = int(v_to_h_ratio * 50)
	window = signal.hann(M = mi)
	x_in_spectra = np.array(x_in_spectra)
	x_spectrogram = np.empty(shape=(x_in_spectra.shape[0],input_shape[0],input_shape[1],1))
	i = 0
	while i < x_in_spectra.shape[0]:
		[x, t, spec] = signal.spectrogram(x = x_in_spectra[i], fs = 1,window = window, nperseg = mi, noverlap = nover)
		plt.pcolormesh(t, x, (np.log(np.abs(spec)) + num) / num)
		plt.title('Spectrogtam Magnitude')
		plt.ylabel('Frequency')
		plt.xlabel('Wavelength')
		plt.savefig(path+'/Spectrogram'+str(i))
		plt.close()
		i += 15

def outputAtNormalRange(y, prop, mode):
	props = lucas_soil_properties()
	if mode == 'linear_minus_1_1':
		y_norm = 2 * (y - props['min'][prop]) / (props['max'][prop] - props['min'][prop]) - 1
	elif mode == 'statistic_minus_1_1':
		y_norm = (y - props['mean'][prop]) / (props['st.dev.'][prop])
	elif mode == 'linear_zero_one':
		y_norm = outputAtNormalRange(y, prop, 'linear_minus_1_1') / 2 + 0.5
	elif mode == 'statistic_zero_one':
		y_norm = outputAtNormalRange(y, prop, 'statistic_minus_1_1') / 2 + 0.5
	return y_norm

def outputFromNormalRange(y_norm, prop, mode):
	props = lucas_soil_properties()
	if mode == 'linear_minus_1_1':
		y = (y_norm + 1) * (props['max'][prop] - props['min'][prop]) / 2 + props['min'][prop]
	elif mode == 'statistic_minus_1_1':
		y = y_norm * props['st.dev.'][prop] + props['mean'][prop]
	elif mode == 'linear_zero_one':
		y = outputFromNormalRange((y_norm - 0.5) * 2, prop, 'linear_minus_1_1')
	elif mode == 'statistic_zero_one':
		y = outputFromNormalRange((y_norm - 0.5) * 2, prop, 'statistic_minus_1_1')
	return y

def applySpectraSavgol( x_in_spectra, window_length, polyorder, deriv ):
	spectra_out = signal.savgol_filter(x, window_length, polyorder, deriv)
	return spectra_out


def convUnit(net,filNumber,kernel_size,pool=False):
	kernel_initializer = 'random_uniform'
	bias_initializer = 'zeros'
	out = Conv2D(filters=filNumber, kernel_size=kernel_size, padding="same", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(net)
	out = BatchNormalization()(out)
	out = ReLU()(out)

	if pool:
		out = MaxPooling2D(pool_size=(2, 2))(out)
	return out

def createModelSingle(input_shape, printDetails):
	# To try activations: (relu, elu, tanh, )
	input_shape_1 = tuple([input_shape[0], input_shape[1], 1])
	input_shape_2 = tuple([input_shape_1[0] / 2, input_shape_1[1] / 2, 1])
	input_shape_3 = tuple([input_shape_2[0] / 2, input_shape_2[1] / 2, 1])
	input_shape_4 = tuple([input_shape_3[0] / 2, input_shape_3[1] / 2, 1])
	activation_fun = 'relu'
	kernel_initializer = 'random_uniform'
	bias_initializer = 'zeros'
	constant_initializer = Constant(value=-0.01)
	# kernel_initializer = constant_initializer
	optimizer = Adam(lr=0.0005)
	#create model
	model = Sequential()
	# Layer 1
	model.add(Conv2D(64, kernel_size=3, padding='same', input_shape=input_shape_1, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
	model.add(BatchNormalization())
	model.add(ReLU())
	# Layer 2
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# Layer 3
	model.add(Conv2D(128, kernel_size=3, padding='same', input_shape=input_shape_2, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
	model.add(BatchNormalization())
	model.add(ReLU())
	# Layer 4
	model.add(Conv2D(256, kernel_size=3, padding='same', input_shape=input_shape_2, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
	model.add(BatchNormalization())
	model.add(ReLU())
	# Layer 5	
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# Layer 6
	model.add(Conv2D(512, kernel_size=3, padding='same', input_shape=input_shape_3, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
	model.add(BatchNormalization())
	model.add(ReLU())
	# Layer 7
	model.add(Conv2D(64, kernel_size=3, padding='same', input_shape=input_shape_3, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
	model.add(BatchNormalization())
	model.add(ReLU())
	# Layer 8
	model.add(Flatten(input_shape=model.output_shape[1:]))
	model.add(Dropout(0.5))
	model.add(Dense(100, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
	model.add(BatchNormalization())
	# model.add(Dense(10))
	model.add(ReLU())
	# Layer 9
	model.add(Dense(1, activation='linear', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
	model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
	if printDetails:
		print(model.summary())
	return model

def createModelMulti(input_shape, prop_count, printDetails):
	# global prop_count
	input_shape_1 = tuple([input_shape[0], input_shape[1], 1])
	input_shape_2 = tuple([input_shape_1[0] / 2, input_shape_1[1] / 2, 1])
	input_shape_3 = tuple([input_shape_2[0] / 2, input_shape_2[1] / 2, 1])
	input_shape_4 = tuple([input_shape_3[0] / 2, input_shape_3[1] / 2, 1])
	kernel_initializer = 'random_uniform'
	bias_initializer = 'zeros'
	optimizer = Adam(lr=0.0005)
	# TODO: Multi input?
	input_layer = Input(shape=input_shape_1)
	# Layer 1
	# Layer 2
	cnn_common = convUnit(input_layer, 64, 3, True)
	# Layer 3
	cnn_common = convUnit(cnn_common, 128, 3, False)
	# Layer 4
	# Layer 5	
	cnn_common = convUnit(cnn_common, 256, 3, True)
	# Layer 6
	cnn_common = convUnit(cnn_common, 512, 3, False)
	# Split to multi
	outputs = []
	for i in range(prop_count):
		# Layer 7
		mult_layer = convUnit(cnn_common, 64, 1, False)
		# Layer 8
		mult_layer = Flatten()(mult_layer)
		# mult_layer = Flatten(input_shape=cnn_common.output_shape[1:])(mult_layer)
		# model.add(Dropout(0.5))
		mult_layer = Dense(100, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(mult_layer)
		mult_layer = BatchNormalization()(mult_layer)
		# model.add(Dense(10))
		mult_layer = ReLU()(mult_layer)
		# Layer 9
		mult_layer = Dense(1, activation='linear', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(mult_layer)
		outputs.append(mult_layer)

	model = Model(inputs=input_layer, outputs=outputs)
	model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
	if printDetails:
		print(model.summary())
	return model 

def customModelSingle( fold_path, prop, x_in_train, y_train, x_in_val, y_val, x_in_test, y_test, v_to_h_ratio, epochs, batch_size ):
	output_mode = 'statistic_minus_1_1'
	input_mode = 'minus_1_1'

	input_shape = getInputShape(x_in_train[0], v_to_h_ratio)
	x_train_spec = spectraToSpectrogram(x_in_train, input_mode, v_to_h_ratio, input_shape)
	x_val_spec = spectraToSpectrogram(x_in_val, input_mode, v_to_h_ratio, input_shape)
	x_test_spec = spectraToSpectrogram(x_in_test, input_mode, v_to_h_ratio, input_shape)
	# extractSpectrogram(x_in_train, fold_path, input_mode, v_to_h_ratio, input_shape)

	# Normalize output properties at range [-1, 1]
	y_train = outputAtNormalRange(np.array(y_train), prop, output_mode)
	y_test = outputAtNormalRange(np.array(y_test), prop, output_mode)
	y_val = outputAtNormalRange(np.array(y_val), prop, output_mode)

	print(x_train_spec.shape)
	print(np.amin(x_train_spec))
	print(np.amax(x_train_spec))
	
	patience = 0
	while patience < 5:
		model = createModelSingle(input_shape, printDetails = True)
		#train the model
		clbcks = []
		clbcks.append(TrainingResetCallback())
		# clbcks.append(ReduceLROnPlateau(min_lr=0.0001))
		clbcks.append(ModelCheckpoint(fold_path+'/'+prop+'_weights.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min', period=1))
		# {epoch:02d}-{val_loss:.2f}.
		# clbcks.append(EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min', baseline=None, restore_best_weights=True))
		# clbcks.append(PrintModelCallback())
		history = model.fit(x_train_spec, y_train, validation_data=(x_val_spec, y_val), epochs=epochs, batch_size=batch_size, callbacks=clbcks)
		stand_d = np.std(history.history['val_loss'])
		mean = np.mean(history.history['val_loss'])
		print("Learning Curve Standard Deviation: "+str(stand_d))
		train = stand_d < 0.01 and mean > 0.1
		if not train:
			break
		patience += 1
		print("Reinitializing Model...")

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Global Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='upper right')
	plt.grid(linestyle=':')
	plt.savefig(fold_path+'/'+prop+'model-loss.eps',format='eps',dpi=1000,bbox_inches='tight')
	plt.close()
	# y_test_pred = []
	# for i in range(x_test_spec.shape[0]):
	# 	y_test_pred.append(model.predict(x_test_spec[i].reshape(1,51,83,1)).reshape(1).tolist())
	# y_test_pred = np.array(y_test_pred)
	model_weights = fold_path+'/'+prop+'_weights.hdf5'
	model = createModelSingle(input_shape, printDetails = False)
	model.load_weights(model_weights)
	# TODO: restore model...
	y_train_pred = model.predict(x_train_spec)
	y_val_pred = model.predict(x_val_spec)
	y_test_pred = model.predict(x_test_spec)
	rmse_val, determ_val, rpiq_val = computeErrors(y_val, y_val_pred)
	print('rmse_val = '+str(rmse_val))
	print('determ_val = '+str(determ_val))
	print('rpiq_val = '+str(rpiq_val))
	# model.save(fold_path+'/'+prop+'model')
	y_test = np.array(y_test)

	# Return to initial prop range
	y_train = outputFromNormalRange(y_train, prop, output_mode)
	y_test = outputFromNormalRange(y_test, prop, output_mode)
	y_val = outputFromNormalRange(y_val, prop, output_mode)
	y_train_pred = outputFromNormalRange(y_train_pred, prop, output_mode)
	y_val_pred = outputFromNormalRange(y_val_pred, prop, output_mode)
	y_test_pred = outputFromNormalRange(y_test_pred, prop, output_mode)

	y_train_pred = np.array(y_train_pred.reshape(y_train_pred.shape[0]).tolist())
	y_val_pred = np.array(y_val_pred.reshape(y_val_pred.shape[0]).tolist())
	y_test_pred = np.array(y_test_pred.reshape(y_test_pred.shape[0]).tolist())
	print('y_test')
	print(y_test)
	print('y_test_pred')
	print(y_test_pred)
	mse_test = ((y_test_pred - y_test)**2).mean()
	mse_train = ((y_train_pred - y_train)**2).mean()
	mse_val = ((y_val_pred - y_val)**2).mean()
	rmse_test, determ_test, rpiq_test = computeErrors(y_test, y_test_pred)
	print('rmse = '+str(rmse_test))
	print('determ = '+str(determ_test))
	print('rpiq = '+str(rpiq_test))
	rmse_train, determ_train, rpiq_train = computeErrors(y_train, y_train_pred)
	rmse_val, determ_val, rpiq_val = computeErrors(y_val, y_val_pred)
	metrics = pnd.DataFrame({'Set': ['Train', 'Val', 'Test'],
							 'MSE': [mse_train, mse_val, mse_test],
							 'RMSE': [rmse_train, rmse_val, rmse_test],
							 'determ': [determ_train, determ_val, determ_test],
							 'rpiq': [rpiq_train, rpiq_val, rpiq_test]})
	metrics.to_csv(fold_path+'/'+prop+'_metrics.csv')
	# batch_size
	# import pandas
	# pandas.dataframe
	a = pnd.DataFrame({'y_test': y_test, 'y_test_pred': y_test_pred})
	a.to_csv(fold_path+'/'+prop+'.csv')

	return rmse_train, rmse_val, rmse_test, determ_train, determ_val, determ_test, rpiq_train, rpiq_val, rpiq_test
	# oc, Clay
	# savgSl_filter feflectances

def customModelMulti( fold_path, output_properties, x_in_train, y_train, x_in_val, y_val, x_in_test, y_test, v_to_h_ratio, epochs, batch_size ):
	# multi
	print(len(y_train))
	print(output_properties.items())
	# exit()
	prop_count = len(y_train)
	instances_train = len(y_train[0])
	instances_test = len(y_test[0])
	instances_val = len(y_val[0])
	output_mode = 'statistic_minus_1_1'
	input_mode = 'minus_1_1'
	input_shape = getInputShape(x_in_train[0], v_to_h_ratio)
	x_train_spec = spectraToSpectrogram(x_in_train, input_mode, v_to_h_ratio, input_shape)
	x_val_spec = spectraToSpectrogram(x_in_val, input_mode, v_to_h_ratio, input_shape)
	x_test_spec = spectraToSpectrogram(x_in_test, input_mode, v_to_h_ratio, input_shape)

	model = createModelMulti(input_shape, prop_count, True)
	# x_train_spec = spectraToSpectrogramMulti(x_in_train, input_mode)
	# x_val_spec = spectraToSpectrogramMulti(x_in_val, input_mode)
	# x_test_spec = spectraToSpectrogramMulti(x_in_test, input_mode)
	# extractSpectrogram(x_in_train, fold_path, input_mode, v_to_h_ratio, input_shape)

	# Normalize output properties at range [-1, 1]
	if tensorflow.__version__[0] == '1':
		y_train_model = []
		y_test_model = []
		y_val_model = []
		i = 0
		for prop, col in output_properties.items():
			y_train_model.append([x for x in outputAtNormalRange(np.array(y_train[i]), prop, output_mode)])
			y_test_model.append([x for x in outputAtNormalRange(np.array(y_test[i]), prop, output_mode)])
			y_val_model.append([x for x in outputAtNormalRange(np.array(y_val[i]), prop, output_mode)])
			i += 1
			# print(y_train)
	else:
		y_train_model = np.zeros(shape=(prop_count, instances_train))
		y_test_model = np.zeros(shape=(prop_count, instances_test))
		y_val_model = np.zeros(shape=(prop_count, instances_val))
		i = 0
		for prop, col in output_properties.items():
			y_train_model[i, :] = [x for x in outputAtNormalRange(np.array(y_train[i]), prop, output_mode)]
			y_test_model[i, :] = [x for x in outputAtNormalRange(np.array(y_test[i]), prop, output_mode)]
			y_val_model[i, :] = [x for x in outputAtNormalRange(np.array(y_val[i]), prop, output_mode)]
			i += 1
		# y_train_model = y_train_model.transpose()
		# y_test_model = y_test_model.transpose()
		# y_val_model = y_val_model.transpose()
	# print(y_train_model)
	# print(y_val_model)
	clbcks = []
	clbcks.append(TrainingResetCallback())
	# clbcks.append(ReduceLROnPlateau(min_lr=0.0001))
	clbcks.append(ModelCheckpoint(fold_path+'/multi_weights.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min', period=1))
	history = model.fit(x_train_spec, y_train_model, epochs=epochs, batch_size=batch_size,
                               validation_data=(x_val_spec, y_val_model),callbacks=clbcks)
	print(x_train_spec.shape)
	print(np.amin(x_train_spec))
	print(np.amax(x_train_spec))
	return insd

def createOuputData(self, ind):
	y = np.zeros(shape=(len(self.outputs), len(ind)))
	for i, key in enumerate(self.outputs):
		y[i, :] = [self.outputs[key][j] for j in ind]
	return y.transpose()

def getMultTrainData(self, f):
	print('Calculating Training Fold...')
	fold = self.sp.k_fold(f)

	trainX = np.zeros(
		(len(fold[0]), len(self.spectra["Absorbances"][0]), 6))
	for i, key in enumerate(self.spectra.keys()):
		print(key)
		print(len(self.spectra[key][0]))
		trainX[:, :, i] = np.array([self.spectra[key][j] for j in fold[0]])

	if self.multipleOutput:
	    trainY = self.createOuputData(fold[0])
	else:
	    trainY = np.array([self.out[i] for i in fold[0]])
	return trainX, trainY