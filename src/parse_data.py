#!/usr/bin/env python
# coding: utf-8
import sys
# if (len(sys.argv) < 6):
#     print('Error! Incorrect arguments')
#     exit('Usage: python parse_data.py <model_type> <epochs> <batch_size> <undersampling> <v/h> <property1> <property2> <etc>')
import spectra_parser as sp
import argparse as argp
import os
import numpy as np
import pandas as pnd
import modelCNN2dSpectr
from modelCNN2dSpectr import OutputStandarizer
from datetime import datetime

def parse_args():
    parser = argp.ArgumentParser(description='2D Convolutional Neural Network')
    # Required
    parser.add_argument('-n','--name',type=str, help='A specific name for the test of the batch',default='')
    parser.add_argument('-si','--singleInput', help='Single Input',action='store_true')
    parser.add_argument('-so','--singleOutput', help='Single Output',action='store_true')
    parser.add_argument('-pr','--properties',type=str, help='Properties on which models will be trained',default=['OC'], nargs='+')
    parser.add_argument('-mp','--maxPooling', type=int, help='Positions of Max Pooling layers', nargs='*', default=[0, 2])
    parser.add_argument('-lf','--layersFilters', type=int, help='Number of filter at each layer', nargs='+', default=[64, 128, 256, 512, 64])
    parser.add_argument('-mdn','--middleDenseLayersSizes', type=int, help='Size of the middle dense layers', nargs='+', default=[])
    parser.add_argument('-dn','--denseLayersSizes', type=int, help='Size of the last dense layers', nargs='+', default=[0])
    parser.add_argument('-prt','--preprecessingTec', type=str, help='Preprocessing techniques for input spectra', nargs='+', default=["reflectances.csv", "absorbances_sg1.csv"])
    parser.add_argument('-is','--inputSpectra', type=str, help='Input Spectra instances json to use', default='Mineral_Absorbances')
    # OptionalMineral_Absorbances
    parser.add_argument('-k','--kernelSize',type=int, help='Select kernel size',default=3)
    parser.add_argument('-b','--batchSize',type=int, help='Select batch size',default=24)
    parser.add_argument('-e','--epochs',type=int, help='Number of epochs',default=100)
    parser.add_argument('-fl','--folds',type=int, help='Number of folds',default=5)
    parser.add_argument('-us','--undersampling',type=float, help='Undersampling factor (greater or equal to 1)',default=1)
    parser.add_argument('-vh','--vhRatio',type=float, help='Ratio of vertical to horizontal image aspect (diversion from [51, 83])',default=1)
    parser.add_argument('-opt','--optimizer',type=str, help='Optimizer used during training',default='Adam')
    parser.add_argument('-mod','--saveModel', help='Decide whether model will be saved at output directory',action='store_true')
    parser.add_argument('-pl','--plotLearnCurves', help='Extract Learning Curve Plots', action='store_true')

    args = parser.parse_args()
    return args

args = parse_args()
print(args)
print("Epochs: ", args.epochs)
print("Batch Size: ", args.batchSize)
print("Undersampling Factor: ", args.undersampling)
print("V/H Image: ", args.vhRatio)
# Path definitions here
FOLDER_WITH_SPECTRA = "../dataset/sources"
# PREPROCESSING_TECHNIQUES = ["reflectances.csv", "absorbances_sg1.csv"]
PATH_TO_PROPERTIES  = "../dataset/properties.csv"
OUTPUT_PROPERTIES_COLUMNS = {
    "Clay" :5,
    "Silt": 6,
    "Sand": 7,
    "pH": 9,
    "OC": 10,
    "CaCO3": 11,
    "N": 12,
    "P": 13,
    "K": 14,
    "CEC": 15
}
output_properties = { prop : OUTPUT_PROPERTIES_COLUMNS[prop] for prop in args.properties }
print(output_properties)
prop_count = len(output_properties.items())
for out_name, out_col in output_properties.items():
    print("Prop "+out_name)
for index, out_col in enumerate(args.properties):
    print(index, out_col)

print("Loading Data")
INPUT_SPECTRA = "../dataset/"+args.inputSpectra+".json"
data_parser = sp.SpectraParser(INPUT_SPECTRA)
data_parser.output_file = PATH_TO_PROPERTIES
print("Done")
# Output Paths
OUTPUT_PATH = '../output'
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
datetime_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S_")+args.name+'/'
path_datetime = os.path.join(OUTPUT_PATH, datetime_str)
if not os.path.exists(path_datetime):
    os.mkdir(path_datetime)
df = pnd.DataFrame.from_dict(args.__dict__, orient='index')
df.transpose()
# arguments = pnd.DataFrame(args.__dict__)
df.to_csv(path_datetime+'/arguments.csv')
if args.singleOutput:# == "single" or model_type == "single_multi":
    single_path = os.path.join(path_datetime, 'single_output')
    if not os.path.exists(single_path):
        os.mkdir(single_path)
    for pre_process in args.preprecessingTec:
        preproc_datetime = os.path.join(single_path, pre_process[0:len(pre_process) - 4]) if args.singleInput else os.path.join(single_path, 'multi_input')
        if not os.path.exists(preproc_datetime):
            os.mkdir(preproc_datetime)
        
        print("Preproc "+pre_process)
        metrics = {}
        train_dataframe = {}
        test_dataframe = {}
        val_dataframe = {}
        for out_name, out_col in output_properties.items():
            print("Prop "+out_name)
        
            data_parser.input_spectra = os.path.join(FOLDER_WITH_SPECTRA, pre_process)
            if args.singleInput:
                print("Preproc "+pre_process)
                data_parser.input_spectra = os.path.join(FOLDER_WITH_SPECTRA, pre_process)
                x = data_parser.x()
            else:
                x = []
                for pre_proc in args.preprecessingTec:
                    print("Preproc "+pre_proc)
                    data_parser.input_spectra = os.path.join(FOLDER_WITH_SPECTRA, pre_proc)
                    x.append(data_parser.x())
            y = data_parser.y(out_col)
            standarizer = modelCNN2dSpectr.OutputStandarizer({out_name: out_col}, [y])
            print(len(x))
            # exit()
            if args.undersampling != 1:
                indices = []
                i = 0
                length = len(x[0] if args.singleInput else x[0][0])
                indices_orig = range(length)
                while i < length:
                    indices.append(i)
                    i += args.undersampling

                if args.singleInput:
                    x_1 = []
                    for i in range(len(x)):
                        x_1.append(np.interp(indices, indices_orig, x[i]))
                    x = x_1
                else:
                    for j in range(len(args.preprecessingTec)):
                        x_1 = []
                        for i in range(len(x[j])):
                            x_1.append(np.interp(indices, indices_orig, x[j][i]))
                        x[j] = x_1
            y_train_actuals = []
            y_train_preds = []
            y_test_actuals = []
            y_test_preds = []
            y_val_actuals = []
            y_val_preds = []
            for fold in range(len(data_parser.cal_folds)): # Number of internal folds
                print("==========| FOLD "+str(fold+1)+", on "+out_name+" |==========")
                [trn, val] = data_parser.k_fold(fold)
                if args.singleInput: 
                    x_trn = [x[i] for i in trn]
                    x_val = [x[i] for i in val]
                    x_tst = [x[i] for i in data_parser.tst_indices]
                else:
                    x_trn = []
                    x_val = []
                    x_tst = []
                    for j in range(len(args.preprecessingTec)):
                        x_trn.append([x[j][i] for i in trn])
                        x_val.append([x[j][i] for i in val])
                        x_tst.append([x[j][i] for i in data_parser.tst_indices])

                y_trn = [y[i] for i in trn]
                y_val = [y[i] for i in val]
                y_tst = [y[i] for i in data_parser.tst_indices]
                path_fold = os.path.join(preproc_datetime, str(fold))
                if not os.path.exists(path_fold):
                    os.mkdir(path_fold)
                
                soil_predictor = modelCNN2dSpectr.SoilModel(path_fold, x_trn[0], args, standarizer, out_name)
                y_train_model, y_train_pred, y_test_model, y_test_pred, y_val_model, y_val_pred = soil_predictor.customModelSingle(x_trn, y_trn, x_val, y_val, x_tst, y_tst)
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

        

if not args.singleOutput:#model_type == "multi" or model_type == "single_multi":
    multi_path = os.path.join(path_datetime, 'multi_output')
    if not os.path.exists(multi_path):
        os.mkdir(multi_path)
    for pre_process in args.preprecessingTec:
        preproc_datetime = os.path.join(multi_path, pre_process[0:len(pre_process) - 4]) if args.singleInput else os.path.join(multi_path, 'multi_input')
        if not os.path.exists(preproc_datetime):
            os.mkdir(preproc_datetime)
        if args.singleInput:
            print("Preproc "+pre_process)
            data_parser.input_spectra = os.path.join(FOLDER_WITH_SPECTRA, pre_process)
            x = data_parser.x()
        else:
            x = []
            for pre_proc in args.preprecessingTec:
                print("Preproc "+pre_proc)
                data_parser.input_spectra = os.path.join(FOLDER_WITH_SPECTRA, pre_proc)
                x.append(data_parser.x())
        y = []
        for out_name, out_col in output_properties.items():
            print(out_name)
            y.append(data_parser.y(out_col))
        standarizer = modelCNN2dSpectr.OutputStandarizer(output_properties, y)
        print(standarizer.statistics)
        # print(len(y))
        # print(len(x))
        if args.undersampling != 1:
            indices = []
            i = 0
            length = len(x[0] if args.singleInput else x[0][0])
            indices_orig = range(length)
            while i < length:
                indices.append(i)
                i += args.undersampling
            
            if args.singleInput:
                x_1 = []
                for i in range(len(x)):
                    x_1.append(np.interp(indices, indices_orig, x[i]))
                x = x_1
            else:
                for j in range(len(args.preprecessingTec)):
                    x_1 = []
                    for i in range(len(x[j])):
                        x_1.append(np.interp(indices, indices_orig, x[j][i]))
                    x[j] = x_1

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
            if args.singleInput: 
                x_trn = [x[i] for i in trn]
                x_val = [x[i] for i in val]
                x_tst = [x[i] for i in data_parser.tst_indices]
            else:
                x_trn = []
                x_val = []
                x_tst = []
                for j in range(len(args.preprecessingTec)):
                    x_trn.append([x[j][i] for i in trn])
                    x_val.append([x[j][i] for i in val])
                    x_tst.append([x[j][i] for i in data_parser.tst_indices])
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
            soil_predictor = modelCNN2dSpectr.SoilModel(path_fold, x_trn[0], args, standarizer, '', output_properties)
            y_train_model, y_train_pred, y_test_model, y_test_pred, y_val_model, y_val_pred = soil_predictor.customModelMulti( x_trn, y_trn, x_val, y_val, x_tst, y_tst )
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
        if not args.singleInput:
            break
