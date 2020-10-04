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
    plt.savefig(path,format='svg', dpi=1000, bbox_inches='tight')
    plt.close()

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
        for prop in args.properties:
            data_test[prop] = {}
        for prop in data_test:
            for metric in args.metrics:
                data_test[prop][metric] = []
    if 'Val' in args.useSet:
        data_val = {}
        for prop in args.properties:
            data_val[prop] = {}
        for prop in data_val:
            for metric in args.metrics:
                data_val[prop][metric] = []
    if 'Train' in args.useSet:
        data_train = {}
        for prop in args.properties:
            data_train[prop] = {}
        for prop in data_train:
            for metric in args.metrics:
                data_train[prop][metric] = []
    props = []
    for folder in args.folders:
        data = pnd.read_csv(folder+'/metrics.csv', index_col=0)
        properties = data.columns
        # datrans = data.T
        print('properties', properties)
        print('data')
        print(data)
        for prop in args.properties:
            if 'Test' in args.useSet:
                for metric in args.metrics:
                    data_test[prop][metric].append(data[prop]['test_'+metric])
            if 'Val' in args.useSet:
                for metric in args.metrics:
                    data_val[prop][metric].append(data[prop]['val_'+metric])
            if 'Train' in args.useSet:
                for metric in args.metrics:
                    data_train[prop][metric].append(data[prop]['train_'+metric])
    
    print(data_test)
    exit()
    fig, ax = plt.subplots()

    labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    men_means = [20, 34, 30, 35, 27]
    women_means = [25, 32, 34, 20, 25]

    x = np.arange(len(properties))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects = []
    pos = 4
    rects.append(ax.bar(x - width/2, men_means, width, label='Men'))
    rects1 = ax.bar(x - width/2, men_means, width, label='Men')
    rects2 = ax.bar(x + width/2, women_means, width, label='Women')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()

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
            sets_char = sets_char+set_char+'-'
        ax.set_title(prop+' '+sets_char+'RMSE')
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
        plt.savefig(path_datetime+'/'+prop+'.svg',format='svg', dpi=1000, bbox_inches='tight')
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
    extractDiagram(values_x, mean_val_rmse, args.parameterName, 'Validation RMSE', path_datetime+'/RMSE_'+args.parameterSymbol+'.svg')
    extractDiagram(values_x, mean_val_rpiq, args.parameterName, 'Validation RPIQ', path_datetime+'/RPIQ_'+args.parameterSymbol+'.svg')
    extractDiagram(values_x, mean_val_determ, args.parameterName, 'Validation R^2', path_datetime+'/Determ_'+args.parameterSymbol+'.svg')