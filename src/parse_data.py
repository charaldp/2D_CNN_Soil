#!/usr/bin/env python
# coding: utf-8
import spectra_parser as sp
import os
import numpy as np
import pandas as pnd
import sys
import modelCNN2dSpectr
from datetime import datetime
# Usage: python parse_data.py <epochs> <undertsamplig> <v/h> [<properties>]
epochs = int(sys.argv[1])
undersampling_factor = float(sys.argv[2])
v_to_h_ratio = float(sys.argv[3])
print("Epochs: "+str(epochs))
print("Undersampling Factor: "+str(undersampling_factor))
print("V/H Image: "+str(v_to_h_ratio))
# Path definitions here
folder_with_spectra = "../dataset/sources"
# pre_processing_techniques = [
#     "Absorbances_reduced.csv", "Absorbances_SG0_SNV_reduced.csv", 
#     "Absorbances_SG1_reduced.csv", "Absorbances_SG1_SNV_reduced.csv",
#     "Absorbances_SG2_reduced.csv", "CR_reduced.csv"]
pre_processing_techniques = ["absorbances_sg1.csv"]
path_to_properties  = "../dataset/properties.csv"
output_properties_cols = {
    "clay" :5,
    "silt": 6,
    "sand": 7,
    # "pH.in.H20": 9,
    "pH": 9,
    "OC": 10,
    "CaCO3": 11,
    "N": 12,
    "P": 13,
    "K": 14,
    "CEC": 15
}
output_properties = { sys.argv[x] : output_properties_cols[sys.argv[x]] for x in range(4, len(sys.argv)) }
print(output_properties)

print("Loading Data")
data_parser = sp.SpectraParser("../dataset/Woodland_Absorbances.json")
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
for out_name, out_col in output_properties.items(): # iteritems for python2
    print("Prop "+out_name)
    for pre_process in pre_processing_techniques:
        print("Preproc "+pre_process)
        data_parser.input_spectra = os.path.join(folder_with_spectra, pre_process)
        x = data_parser.x()
        y = data_parser.y(out_col)
        indices = []
        indices_orig = range(len(x[0]))
        i = 0
        print(len(x))
        # exit()
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
            path_fold = os.path.join(path_datetime, str(fold))
            if not os.path.exists(path_fold):
                os.mkdir(path_fold)
            rmse_train, rmse_val, rmse_test, determ_train, determ_val, determ_test, rpiq_train, rpiq_val, rpiq_test = modelCNN2dSpectr.customModel(path_fold, out_name, x_trn, y_trn, x_val, y_val, x_tst, y_tst, v_to_h_ratio, epochs)
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
        metrics.to_csv(path_datetime+out_name+'_metrics.csv')
