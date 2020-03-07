#!/usr/bin/env python
# coding: utf-8
import sys
if (len(sys.argv) < 6):
    print('Error! Incorrect arguments')
    print('Usage: python parse_data.py <model_type> <epochs> <batch_size> <undersampling> <v/h> <property1> <property2> <etc>')
    exit(1)
import spectra_parser as sp
import os
import numpy as np
import pandas as pnd
import modelCNN2dSpectr
from datetime import datetime
arg_it = 1
# global model_type
model_type = str(sys.argv[arg_it])
arg_it+=1
# global epochs
epochs = int(sys.argv[arg_it])
arg_it+=1
# global batch_size
batch_size = int(sys.argv[arg_it])
arg_it+=1
# global undersampling_factor
undersampling_factor = float(sys.argv[arg_it])
arg_it+=1
# global v_to_h_ratio
v_to_h_ratio = float(sys.argv[arg_it])
arg_it+=1
print("Epochs: "+str(epochs))
print("Batch Size: "+str(batch_size))
print("Undersampling Factor: "+str(undersampling_factor))
print("V/H Image: "+str(v_to_h_ratio))
# Path definitions here
folder_with_spectra = "../dataset/sources"
# pre_processing_techniques = [
#     "Absorbances_reduced.csv", "Absorbances_SG0_SNV_reduced.csv", 
#     "Absorbances_SG1_reduced.csv", "Absorbances_SG1_SNV_reduced.csv",
#     "Absorbances_SG2_reduced.csv", "CR_reduced.csv"]
pre_processing_techniques = ["absorbances_sg1.csv", "reflectances.csv"]
path_to_properties  = "../dataset/properties.csv"
output_properties_cols = {
    "Clay" :5,
    "Silt": 6,
    "Sand": 7,
    # "pH.in.H20": 9,
    "pH": 9,
    "OC": 10,
    "CaCO3": 11,
    "N": 12,
    "P": 13,
    "K": 14,
    "CEC": 15
}
output_properties = { sys.argv[x] : output_properties_cols[sys.argv[x]] for x in range(arg_it, len(sys.argv)) }
print(output_properties)
# global prop_count
prop_count = len(output_properties.items())

print("Loading Data")
data_parser = sp.SpectraParser("../dataset/Mineral_Absorbances.json")
data_parser.output_file = path_to_properties
print("Done")
# Output Paths
output_path = '../output'
if not os.path.exists(output_path):
    os.mkdir(output_path)
datetime_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")+'/'
path_datetime = os.path.join(output_path, datetime_str)
if not os.path.exists(path_datetime):
    os.mkdir(path_datetime)

if model_type == "single" or model_type == "single_multi":
    for out_name, out_col in output_properties.items(): # iteritems for python2
        print("Prop "+out_name)
        for pre_process in pre_processing_techniques:
            preproc_datetime = os.path.join(path_datetime, pre_process[0:len(pre_process) - 4])
            if not os.path.exists(preproc_datetime):
                os.mkdir(preproc_datetime)
            print("Preproc "+pre_process)
            data_parser.input_spectra = os.path.join(folder_with_spectra, pre_process)
            x = data_parser.x()
            y = data_parser.y(out_col)
            print(len(x))
            # exit()
            if undersampling_factor != 1:
                indices = []
                i = 0
                indices_orig = range(len(x[0]))
                while i < len(x[0]):
                    indices.append(i)
                    i += undersampling_factor
                       
                x_1 = []
                for i in range(len(x)):
                    x_1.append(np.interp(indices, indices_orig, x[i]))
                x = x_1
            rmse_train_ar = []
            rmse_val_ar = []
            rmse_test_ar = []
            determ_train_ar = []
            determ_val_ar = []
            determ_test_ar = []
            rpiq_train_ar = []
            rpiq_val_ar = []
            rpiq_test_ar = []
            for fold in range(len(data_parser.cal_folds)): # Number of internal folds
                print("==========| FOLD "+str(fold+1)+", on "+out_name+" |==========")
                [trn, val] = data_parser.k_fold(fold)
                x_trn = [x[i] for i in trn]
                y_trn = [y[i] for i in trn]
                x_val = [x[i] for i in val]
                y_val = [y[i] for i in val]
                x_tst = [x[i] for i in data_parser.tst_indices]
                y_tst = [y[i] for i in data_parser.tst_indices]
                path_fold = os.path.join(preproc_datetime, str(fold))
                if not os.path.exists(path_fold):
                    os.mkdir(path_fold)
                rmse_train, rmse_val, rmse_test, determ_train, determ_val, determ_test, rpiq_train, rpiq_val, rpiq_test, y_pred = modelCNN2dSpectr.customModelSingle(path_fold, out_name, x_trn, y_trn, x_val, y_val, x_tst, y_tst, v_to_h_ratio, epochs, batch_size)
                rmse_train_ar.append(rmse_train)
                rmse_val_ar.append(rmse_val)
                rmse_test_ar.append(rmse_test)
                determ_train_ar.append(determ_train)
                determ_val_ar.append(determ_val)
                determ_test_ar.append(determ_test)
                rpiq_train_ar.append(rpiq_train)
                rpiq_val_ar.append(rpiq_val)
                rpiq_test_ar.append(rpiq_test)
            metrics = pnd.DataFrame({'rmse_train': np.array(rmse_train_ar),
                                    'rmse_val': np.array(rmse_val_ar),
                                    'rmse_test': np.array(rmse_test_ar),
                                    'determ_train': np.array(determ_train_ar),
                                    'determ_val': np.array(determ_val_ar),
                                    'determ_test': np.array(determ_test_ar),
                                    'rpiq_train': np.array(rpiq_train_ar),
                                    'rpiq_val': np.array(rpiq_val_ar),
                                    'rpiq_test': np.array(rpiq_test_ar)})
            metrics.to_csv(preproc_datetime+'/'+out_name+'_metrics.csv')

if model_type == "multi" or model_type == "single_multi":
    for pre_process in pre_processing_techniques:
        preproc_datetime = os.path.join(path_datetime, pre_process[0:len(pre_process) - 4])
        if not os.path.exists(preproc_datetime):
            os.mkdir(preproc_datetime)
        print("Preproc "+pre_process)
        data_parser.input_spectra = os.path.join(folder_with_spectra, pre_process)
        x = data_parser.x()
        y = []
        for out_name, out_col in output_properties.items():
            y.append(data_parser.y(out_col))
        print(len(y))
        print(len(x))
        if undersampling_factor != 1:
            indices = []
            i = 0
            indices_orig = range(len(x[0]))
            while i < len(x[0]):
                indices.append(i)
                i += undersampling_factor
             
            x_1 = []
            for i in range(len(x)):
                x_1.append(np.interp(indices, indices_orig, x[i]))
            x = x_1
        for fold in range(len(data_parser.cal_folds)):
            [trn, val] = data_parser.k_fold(fold)
            x_trn = [x[i] for i in trn]
            x_val = [x[i] for i in val]
            x_tst = [x[i] for i in data_parser.tst_indices]
            y_trn = []
            y_val = []
            y_tst = []
            for j in range(len(output_properties.items())):
                y_trn.append([y[j][i] for i in trn])
                y_val.append([y[j][i] for i in val])
                y_tst.append([y[j][i] for i in data_parser.tst_indices])
            path_fold = os.path.join(preproc_datetime, str(fold))
            if not os.path.exists(path_fold):
                os.mkdir(path_fold)
            modelCNN2dSpectr.customModelMulti( path_fold, output_properties, x_trn, y_trn, x_val, y_val, x_tst, y_tst, v_to_h_ratio, epochs, batch_size )
