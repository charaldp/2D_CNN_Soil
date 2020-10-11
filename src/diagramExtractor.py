import sys,math,random,os,argparse, scipy
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pnd
import argparse as argp
import seaborn as sns
from datetime import datetime

mpl.rcParams.update({'font.size': 6})

def parse_args():
    parser = argp.ArgumentParser(description='Diagram extractor')
    # Required
    parser.add_argument('-n','--name',type=str, help='A specific name for the plot group',default='diag')
    parser.add_argument('-fld','--folders',type=str, help='Model folders to process',default=[''], nargs='+')
    parser.add_argument('-md','--mode',type=str, help='Mode to use script 1) metrics 2) boxplots 3) parameterDiagram ',default='metrics')
    parser.add_argument('-mtrc','--metrics',type=str, help='Metrics to use',default=['rmse'], nargs='+')
    parser.add_argument('-mn','--modelNames',type=str, help='Model names to identify each folder on plots',default=[], nargs='+')
    parser.add_argument('-pv','--parameterValues',type=float, help='Values of parameter used for horizontal axis',default=[], nargs='+')
    parser.add_argument('-pn','--parameterName',type=str, help='Name of parameter used',default='V/H Ratio')
    parser.add_argument('-ps','--parameterSymbol',type=str, help='Name of parameter used',default='v_h')
    parser.add_argument('-set','--useSet',type=str, help='Set of dataset to use',default=['Test'], nargs='+')
    parser.add_argument('-prt','--properties',type=str, help='Properties to examine',default=['OC'], nargs='+')
    
    args = parser.parse_args()
    return args

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
def extractDiagram(x, y, x_name, y_name, path):
    # dtfr = pnd.DataFrame(data_array, columns=columns).transpose()
    # ax = sns.lineplot(data=dataframe, legend=False)
    print(len(x), len(y))
    ax = plt.plot(x, y)
    plt.xlabel(x_name, fontsize=18)
    plt.ylabel(y_name, fontsize=18)
    # ax.tick_params(axis='x', labelsize=18)
    # ax.tick_params(axis='y', labelsize=18)
    plt.savefig(path,format='pdf', dpi=1000, bbox_inches='tight')
    plt.close()

def fixbarplots(ax, data_set, prop, width, metric, metrics_symbols):
    j = 0
    for index, row in data_set[metric].iterrows():
        ax.text(j,row[prop]*1.01, round(row[prop],3), color='black', ha="center", fontsize=10)
        j+=1
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - width
        patch.set_width(width)
        patch.set_x(patch.get_x() + diff * .5)
        plt.ylabel(prop+' '+metrics_symbols[metric], fontsize=12)

metrics_symbols = {'determ': "Coefficient of Determination", 'rpiq': "RPIQ", 'rmse': "Root Mean Squared Error"}
OUTPUT_PATH = '../output/diagrams'
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
datetime_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")+'/'
path_datetime = os.path.join(OUTPUT_PATH, datetime_str)
if not os.path.exists(path_datetime):
    os.mkdir(path_datetime)
args = parse_args()
print(args)
print(args.modelNames)
# exit()
if args.mode == 'metrics':
    if 'Test' in args.useSet:
        data_test = {}
        for metric in args.metrics:
            data_test[metric] = {}
        for metric in args.metrics:
            for prop in args.properties:
                data_test[metric][prop] = {}
    if 'Val' in args.useSet:
        data_val = {}
        for metric in args.metrics:
            data_val[metric] = {}
        for metric in args.metrics:
            for prop in args.properties:
                data_val[metric][prop] = {}
    if 'Train' in args.useSet:
        data_train = {}
        for metric in args.metrics:
            data_train[metric] = {}
        for metric in args.metrics:
            for prop in args.properties:
                data_train[metric][prop] = {}
    props = []
    for j, folder in enumerate(args.folders):
        data = pnd.read_csv(folder+'/metrics.csv', index_col=0)
        properties = data.columns
        # datrans = data.T
        print('properties', properties)
        print('data')
        print(data)
        for metric in args.metrics:
            if 'Test' in args.useSet:
                for prop in args.properties:
                    data_test[metric][prop][args.modelNames[j]] = data[prop]['test_'+metric]
            if 'Val' in args.useSet:
                for prop in args.properties:
                    data_val[metric][prop][args.modelNames[j]] = data[prop]['val_'+metric]
            if 'Train' in args.useSet:
                for prop in args.properties:
                    data_train[metric][prop][args.modelNames[j]] = data[prop]['train_'+metric]
    for metric in args.metrics:
        if 'Test' in args.useSet:
            data_test[metric] = pnd.DataFrame(data_test[metric])
        if 'Val' in args.useSet:
            data_val[metric] = pnd.DataFrame(data_val[metric])
        if 'Train' in args.useSet:
            data_train[metric] = pnd.DataFrame(data_train[metric])

    tips = sns.load_dataset("tips")
    for prop in args.properties:
        for metric in args.metrics:
            print(prop, metric)
            if 'Test' in args.useSet:
                ax = sns.barplot(x=data_test[metric].index, y=prop, data=data_test[metric])
                fixbarplots(ax, data_test, prop, 0.5, metric, metrics_symbols)
                plt.savefig(path_datetime+'/test_'+prop+'_'+metric+'.pdf',format='pdf', dpi=1000, bbox_inches='tight')
                plt.close()
            if 'Val' in args.useSet:
                ax = sns.barplot(x=data_val[metric].index, y=prop, data=data_val[metric])
                fixbarplots(ax, data_val, prop, 0.5, metric, metrics_symbols)
                plt.savefig(path_datetime+'/val_'+prop+'_'+metric+'.pdf',format='pdf', dpi=1000, bbox_inches='tight')
                plt.close()
            if 'Train' in args.useSet:
                ax = sns.barplot(x=data_train[metric].index, y=prop, data=data_train[metric])
                fixbarplots(ax, data_train, prop, 0.5, metric, metrics_symbols)
                plt.savefig(path_datetime+'/train_'+prop+'_'+metric+'.pdf',format='pdf', dpi=1000, bbox_inches='tight')
                plt.close()
            
    print(data_test)
    exit()

if args.mode == 'boxplots':
    if 'Test' in args.useSet:
        data_test = []
    if 'Val' in args.useSet:
        data_val = []
    if 'Train' in args.useSet:
        data_train = []
    props = []
    center_offset = 0.24 + len(args.useSet)
    for j, folder in enumerate(args.folders):
        print(folder+'/test_predictions.csv')
        if 'Test' in args.useSet:
            data_test.append(pnd.read_csv(folder+'/test_predictions.csv', index_col=False))
        if 'Val' in args.useSet:
            data_val.append(pnd.read_csv(folder+'/val_predictions.csv', index_col=False))
        if 'Train' in args.useSet:
            data_train.append(pnd.read_csv(folder+'/train_predictions.csv', index_col=False))
        # Search for properties at each new folder
        cols = data_test[0].columns
        for col in cols:
            if col != cols[0]:
                prop = col[0: col.find('_')]
                if not prop in props:
                    props.append(prop)
    print(props)
    for prop in props:
        data = []
        labels = []
        bplots = []
        fig, ax = plt.subplots()
        sets_char = ''
        for set_char in args.useSet:
            sets_char = sets_char+set_char
        ax.set_title(prop+' '+sets_char+' Absolute Error')
        for j, folder in enumerate(args.folders):
            # fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(6, 6), sharey=True)
            # axs[1, 2].bxp(stats, showfliers=False)
            # axs[1, 2].set_title('showfliers=False', fontsize=fs)
            if prop+'_test_preds' in data_test[j]:
                if 'Test' in args.useSet:
                    errors_test = np.absolute(data_test[j][prop+'_test_preds'] - data_test[j][prop+'_test_actuals'])
                if 'Val' in args.useSet:
                    errors_val = np.absolute(data_val[j][prop+'_val_preds'] - data_val[j][prop+'_val_actuals'])
                if 'Train' in args.useSet:
                    errors_train = np.absolute(data_train[j][prop+'_train_preds'] - data_train[j][prop+'_train_actuals'])
                # labels.append('')
                labels.append(args.modelNames[j])# +' '+prop
                # labels.append('')
                # labels.append(args.modelNames[j]+' '+prop+'_val')
                # labels.append(args.modelNames[j]+' '+prop+'_train')
            # print('N_test_preds', data_test['N_test_preds'])
            # print('N_test_actuals', data_test['N_test_actuals'],)
            # print('errors', errors)
            data = []
            if 'Test' in args.useSet:
                data.append(errors_test)
            if 'Val' in args.useSet:
                data.append(errors_val)
            if 'Train' in args.useSet:
                data.append(errors_train)
            pos = center_offset*j
            # ax = sns.boxplot(x="day", y="total_bill", hue="smoker", data=tips, palette="Set3")
            positions_1 = [pos+i for i in range(1, 1+len(args.useSet))]
            bplots.append(ax.boxplot(data, positions=positions_1, showfliers=False, patch_artist=True, widths=0.6))
        

        colors = ['lightblue', 'orange', 'lightgreen']
        for bplot in bplots:
            for j, patch in enumerate(bplot['boxes']):
                patch.set_facecolor('lightblue')
                # patch.set_facecolor(colors[j%3])

        # ax.legend([bplots[0]["boxes"][0], bplots[0]["boxes"][1], bplots[0]["boxes"][2]],  ['Test', 'Validation', 'Training'], loc='upper right')
        # plt.show()
        positions = np.arange(1, center_offset * (len(labels)) + 1, center_offset)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        plt.savefig(path_datetime+'/'+prop+'.pdf',format='pdf', dpi=1000, bbox_inches='tight')
        plt.close()
         

        # plt.plot(history.history[name])
        # plt.plot(history.history['val_'+name])
        # plt.title('Global Model Loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Validation'], loc='upper right')
        # plt.grid(linestyle=':')
        # plt.savefig(self.__fold_path+'/model_'+name+'.eps',format='eps',dpi=1000,bbox_inches='tight')
        # plt.close()
        # print(datrans)

        # for col in data_test.columns:
        #     print(data_test[col])
        # for col in data_test.columns:
        #     print(data_test[col])

if args.mode=='parameterDiagram':
    values_x = args.parameterValues
    mean_val_rmse = []
    mean_val_rpiq = []
    mean_val_determ = []
    for j, folder in enumerate(args.folders):
        data = pnd.read_csv(folder+'/multi_output/multi_input/metrics.csv', index_col=0)
        datrans = data.T
        # datrans[0]
        mean_val_rmse.append(np.mean(datrans['val_rmse']))
        mean_val_rpiq.append(np.mean(datrans['val_rpiq']))
        mean_val_determ.append(np.mean(datrans['val_determ']))
    print(mean_val_rmse)
    print(mean_val_rpiq)
    print(mean_val_determ)
    # dataframe_val_rmse = pnd.DataFrame(data=mean_val_rmse, columns=columns).transpose()
    # dataframe_val_rpiq = pnd.DataFrame(data=mean_val_rpiq, columns=columns).transpose()
    # dataframe_val_determ = pnd.DataFrame(data=mean_val_determ, columns=columns).transpose()
    extractDiagram(values_x, mean_val_rmse, args.parameterName, 'Validation RMSE', path_datetime+'/RMSE_'+args.parameterSymbol+'.pdf')
    extractDiagram(values_x, mean_val_rpiq, args.parameterName, 'Validation RPIQ', path_datetime+'/RPIQ_'+args.parameterSymbol+'.pdf')
    extractDiagram(values_x, mean_val_determ, args.parameterName, 'Validation R^2', path_datetime+'/Determ_'+args.parameterSymbol+'.pdf')