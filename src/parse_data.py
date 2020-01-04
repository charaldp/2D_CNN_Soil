#!/usr/bin/env python
# coding: utf-8

import spectra_parser as sp
import os
import numpy as np
import pandas as pnd
import modelCNN2dSpectr
from datetime import datetime

# Path definitions here
folder_with_spectra = "../dataset/sources"
# pre_processing_techniques = [
#     "Absorbances_reduced.csv", "Absorbances_SG0_SNV_reduced.csv", 
#     "Absorbances_SG1_reduced.csv", "Absorbances_SG1_SNV_reduced.csv",
#     "Absorbances_SG2_reduced.csv", "CR_reduced.csv"]
pre_processing_techniques = ["reflectances.csv"]
path_to_properties  = "../dataset/properties.csv"
output_properties = {
    # "clay" :5
    # "silt": 6,
    # "sand": 7,
    # "pH.in.H20": 9,
    "OC": 10
    # "CaCO3": 11,
    # "N": 12,
    # "P": 13,
    # "K": 14,
    # "CEC": 15
}

print("Loading Data")
data_parser = sp.SpectraParser("../dataset/Mineral_Absorbances.json")
data_parser.output_file = path_to_properties
print("Done")
# Output Paths
output_path = './Output'
if not os.path.exists(output_path):
    os.mkdir(output_path)
datetime = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")+'/'
path_datetime = os.path.join(output_path, datetime)
if not os.path.exists(path_datetime):
    os.mkdir(path_datetime)
for out_name, out_col in output_properties.items(): # iteritems for python2
    print("Prop "+out_name)
    for pre_process in pre_processing_techniques:
        print("Preproc "+pre_process)
        data_parser.input_spectra = os.path.join(folder_with_spectra, pre_process)
        x = data_parser.x()
        y = data_parser.y(out_col)
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
            rmse_train, rmse_val, rmse_test, determ_train, determ_val, determ_test, rpiq_train, rpiq_val, rpiq_test = modelCNN2dSpectr.customModel(path_fold, out_name, x_trn, y_trn, x_val, y_val, x_tst, y_tst)
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
            # y_train_pred_array = np.concatenate(y_train_pred_array, y_train_pred)
            # y_train_array = np.concatenate(y_train_array, np.array(y_trn))
            # y_val_array = np.concatenate(y_val_array, np.array(y_val))
            # y_val_pred_array = np.concatenate(y_val_pred_array, y_val_pred)
            # y_test_array = np.concatenate(y_test_array, np.array(y_tst))
            # y_test_pred_array = np.concatenate(y_test_pred_array, y_test_pred)
        # Compute overall errors
        # rmse_train, determ_train, rpiq_train = modelCNN2dSpectr.computeErrors(y_train_array, y_train_pred_array)
        # rmse_val, determ_val, rpiq_val = modelCNN2dSpectr.computeErrors(y_train_array, y_train_pred_array)
        # rmse_test, determ_test, rpiq_test = modelCNN2dSpectr.computeErrors(y_train_array, y_train_pred_array)
        # metrics = pnd.DataFrame({'Set': ['Train', 'Val', 'Test'],
        #                      'RMSE': [rmse_train, rmse_val, rmse_test],
        #                      'determ': [determ_train, determ_val, determ_test],
        #                      'rpiq': [rpiq_train, rpiq_val, rpiq_test]})
        # metrics.to_csv(path_datetime+property+'_metrics.csv')





