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
from modelCNN2dSpectr import OutputStandarizer
from datetime import datetime
arg_it = 1
model_type = str(sys.argv[arg_it])
arg_it+=1
epochs = int(sys.argv[arg_it])
arg_it+=1
batch_size = int(sys.argv[arg_it])
arg_it+=1
undersampling_factor = float(sys.argv[arg_it])
arg_it+=1
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
    single_path = os.path.join(path_datetime, 'single_output')
    if not os.path.exists(single_path):
        os.mkdir(single_path)
    for pre_process in pre_processing_techniques:
        preproc_datetime = os.path.join(single_path, pre_process[0:len(pre_process) - 4])
        if not os.path.exists(preproc_datetime):
            os.mkdir(preproc_datetime)
        print("Preproc "+pre_process)
        metrics = {}
        train_dataframe = {}
        test_dataframe = {}
        val_dataframe = {}
        for out_name, out_col in output_properties.items():
            print("Prop "+out_name)
        
            data_parser.input_spectra = os.path.join(folder_with_spectra, pre_process)
            x = data_parser.x()
            y = data_parser.y(out_col)
            standarizer = modelCNN2dSpectr.OutputStandarizer({out_name: out_col}, [y])
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
            y_train_actuals = []
            y_train_preds = []
            y_test_actuals = []
            y_test_preds = []
            y_val_actuals = []
            y_val_preds = []
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
                y_train_model, y_train_pred, y_test_model, y_test_pred, y_val_model, y_val_pred = modelCNN2dSpectr.customModelSingle(path_fold, out_name, x_trn, y_trn, x_val, y_val, x_tst, y_tst, v_to_h_ratio, epochs, batch_size, standarizer)
                y_train_actuals.extend(y_train_model.flatten().tolist())
                y_train_preds.extend(y_train_pred.flatten().tolist())
                y_test_actuals.extend(y_test_model.flatten().tolist())
                y_test_preds.extend(y_test_pred.flatten().tolist())
                y_val_actuals.extend(y_val_model.flatten().tolist())
                y_val_preds.extend(y_val_pred.flatten().tolist())
            metrics[out_name] = {
                'train_rmse': 0, 'train_determ': 0, 'train_rpiq': 0,
                'test_rmse': 0, 'test_determ': 0, 'test_rpiq': 0,
                'val_rmse': 0, 'val_determ': 0, 'val_rpiq': 0
            }
            metrics[out_name]['train_rmse'], metrics[out_name]['train_determ'], metrics[out_name]['train_rpiq'] = modelCNN2dSpectr.computeErrors(y_train_actuals, y_train_preds)
            metrics[out_name]['test_rmse'], metrics[out_name]['test_determ'], metrics[out_name]['test_rpiq'] = modelCNN2dSpectr.computeErrors(y_test_actuals, y_test_preds)
            metrics[out_name]['val_rmse'], metrics[out_name]['val_determ'], metrics[out_name]['val_rpiq'] = modelCNN2dSpectr.computeErrors(y_val_actuals, y_val_preds)

            train_dataframe[out_name+'_train_actuals'] = y_train_actuals
            train_dataframe[out_name+'_train_preds'] = y_train_preds
            test_dataframe[out_name+'_test_actuals'] = y_test_actuals
            test_dataframe[out_name+'_test_preds'] = y_test_preds
            val_dataframe[out_name+'_val_actuals'] = y_val_actuals
            val_dataframe[out_name+'_val_preds'] = y_val_preds

        train_dataframe = pnd.DataFrame(train_dataframe)
        test_dataframe = pnd.DataFrame(test_dataframe)
        val_dataframe = pnd.DataFrame(val_dataframe)
        train_dataframe.to_csv(preproc_datetime+'/train_predictions.csv')
        test_dataframe.to_csv(preproc_datetime+'/test_predictions.csv')
        val_dataframe.to_csv(preproc_datetime+'/val_predictions.csv')
        metrics = pnd.DataFrame(metrics)
        metrics.to_csv(preproc_datetime+'/metrics.csv')

        

if model_type == "multi" or model_type == "single_multi":
    multi_path = os.path.join(path_datetime, 'multi_output')
    if not os.path.exists(multi_path):
        os.mkdir(multi_path)
    for pre_process in pre_processing_techniques:
        preproc_datetime = os.path.join(multi_path, pre_process[0:len(pre_process) - 4])
        if not os.path.exists(preproc_datetime):
            os.mkdir(preproc_datetime)
        print("Preproc "+pre_process)
        data_parser.input_spectra = os.path.join(folder_with_spectra, pre_process)
        x = data_parser.x()
        y = []
        for out_name, out_col in output_properties.items():
            print(out_name)
            y.append(data_parser.y(out_col))
        standarizer = modelCNN2dSpectr.OutputStandarizer(output_properties, y)
        print(standarizer.statistics)
        # print(len(y))
        # print(len(x))
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
        y_train_actuals = []
        y_train_preds = []
        y_test_actuals = []
        y_test_preds = []
        y_val_actuals = []
        y_val_preds = []
        for i in range(len(y)):
            y_train_actuals.append([])
            y_train_preds.append([])
            y_test_actuals.append([])
            y_test_preds.append([])
            y_val_actuals.append([])
            y_val_preds.append([])
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
            y_train_model, y_train_pred, y_test_model, y_test_pred, y_val_model, y_val_pred = modelCNN2dSpectr.customModelMulti( path_fold, output_properties, x_trn, y_trn, x_val, y_val, x_tst, y_tst, v_to_h_ratio, epochs, batch_size, standarizer )
            for i in range(len(y)):
                y_train_actuals[i].extend(y_train_model[i].flatten().tolist())
                y_train_preds[i].extend(y_train_pred[i].flatten().tolist())
                y_test_actuals[i].extend(y_test_model[i].flatten().tolist())
                y_test_preds[i].extend(y_test_pred[i].flatten().tolist())
                y_val_actuals[i].extend(y_val_model[i].flatten().tolist())
                y_val_preds[i].extend(y_val_pred[i].flatten().tolist())
        i = 0
        metrics = {}
        train_dataframe = {}
        test_dataframe = {}
        val_dataframe = {}
        for out_name, out_col in output_properties.items():
            metrics[out_name] = {
                'train_rmse': 0, 'train_determ': 0, 'train_rpiq': 0,
                'test_rmse': 0, 'test_determ': 0, 'test_rpiq': 0,
                'val_rmse': 0, 'val_determ': 0, 'val_rpiq': 0
            }
            metrics[out_name]['train_rmse'], metrics[out_name]['train_determ'], metrics[out_name]['train_rpiq'] = modelCNN2dSpectr.computeErrors(y_train_actuals[i], y_train_preds[i])
            metrics[out_name]['test_rmse'], metrics[out_name]['test_determ'], metrics[out_name]['test_rpiq'] = modelCNN2dSpectr.computeErrors(y_test_actuals[i], y_test_preds[i])
            metrics[out_name]['val_rmse'], metrics[out_name]['val_determ'], metrics[out_name]['val_rpiq'] = modelCNN2dSpectr.computeErrors(y_val_actuals[i], y_val_preds[i])
            train_dataframe[out_name+'_train_actuals'] = y_train_actuals[i]
            train_dataframe[out_name+'_train_preds'] = y_train_preds[i]
            test_dataframe[out_name+'_test_actuals'] = y_test_actuals[i]
            test_dataframe[out_name+'_test_preds'] = y_test_preds[i]
            val_dataframe[out_name+'_val_actuals'] = y_val_actuals[i]
            val_dataframe[out_name+'_val_preds'] = y_val_preds[i]
            i += 1
        print(metrics)
        train_dataframe = pnd.DataFrame(train_dataframe)
        test_dataframe = pnd.DataFrame(test_dataframe)
        val_dataframe = pnd.DataFrame(val_dataframe)
        train_dataframe.to_csv(preproc_datetime+'/train_predictions.csv')
        test_dataframe.to_csv(preproc_datetime+'/test_predictions.csv')
        val_dataframe.to_csv(preproc_datetime+'/val_predictions.csv')
        metrics = pnd.DataFrame(metrics)
        metrics.to_csv(preproc_datetime+'/metrics.csv')
