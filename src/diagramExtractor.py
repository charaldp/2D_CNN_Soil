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
    parser.add_argument('-md','--mode',type=str, help='Mode to use script 1) \'metrics\' ',default='metrics')
    parser.add_argument('-mn','--modelNames',type=str, help='Model names to identify each folder on plots',default=[], nargs='+')
    parser.add_argument('-pv','--parameterValues',type=float, help='Values of parameter used for horizontal axis',default=[], nargs='+')
    parser.add_argument('-pn','--parameterName',type=str, help='Name of parameter used',default='V/H Ratio')
    parser.add_argument('-ps','--parameterSymbol',type=str, help='Name of parameter used',default='v_h')
    
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
    for folder in args.folders:
        data = pnd.read_csv(folder+'/metrics.csv')
        properties = data.columns
        datrans = data.T
        print('properties', properties)
        print('datrans', datrans)
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
    data_test = []
    data_val = []
    data_train = []
    props = []
    center_offset = 3.24
    for j, folder in enumerate(args.folders):
        print(folder+'/test_predictions.csv')
        data_test.append(pnd.read_csv(folder+'/test_predictions.csv', index_col=False))
        data_val.append(pnd.read_csv(folder+'/val_predictions.csv', index_col=False))
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
        ax.set_title(prop+' Test-Val-Train RMSE')
        for j, folder in enumerate(args.folders):
            # fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(6, 6), sharey=True)
            # axs[1, 2].bxp(stats, showfliers=False)
            # axs[1, 2].set_title('showfliers=False', fontsize=fs)
            if prop+'_test_preds' in data_test[j]:
                errors_test = np.absolute(data_test[j][prop+'_test_preds'] - data_test[j][prop+'_test_actuals'])
                errors_val = np.absolute(data_val[j][prop+'_val_preds'] - data_val[j][prop+'_val_actuals'])
                errors_train = np.absolute(data_train[j][prop+'_train_preds'] - data_train[j][prop+'_train_actuals'])
                # labels.append('')
                labels.append(args.modelNames[j]+' '+prop)
                # labels.append('')
                # labels.append(args.modelNames[j]+' '+prop+'_val')
                # labels.append(args.modelNames[j]+' '+prop+'_train')
            # print('N_test_preds', data_test['N_test_preds'])
            # print('N_test_actuals', data_test['N_test_actuals'],)
            # print('errors', errors)
            data = []
            data.append(errors_test)
            data.append(errors_val)
            data.append(errors_train)
            pos = center_offset*j
            # ax = sns.boxplot(x="day", y="total_bill", hue="smoker", data=tips, palette="Set3")
            bplots.append(ax.boxplot(data, positions=[pos+1, pos+2, pos+3], showfliers=False, patch_artist=True, widths=0.6))
        

        colors = ['lightblue', 'orange', 'lightgreen']
        for bplot in bplots:
            for j, patch in enumerate(bplot['boxes']):
                patch.set_facecolor(colors[j%3])

        ax.legend([bplots[0]["boxes"][0], bplots[0]["boxes"][1], bplots[0]["boxes"][2]],  ['Test', 'Validation', 'Training'], loc='upper right')
        # plt.show()
        positions = np.arange(2, center_offset * len(labels) + 1, center_offset)
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
    mean_test_rmse = []
    mean_test_rpiq = []
    mean_test_determ = []
    for j, folder in enumerate(args.folders):
        data = pnd.read_csv(folder+'/multi_output/multi_input/metrics.csv', index_col=0)
        datrans = data.T
        # datrans[0]
        mean_test_rmse.append(np.mean(datrans['test_rmse']))
        mean_test_rpiq.append(np.mean(datrans['test_rpiq']))
        mean_test_determ.append(np.mean(datrans['test_determ']))
    print(mean_test_rmse)
    print(mean_test_rpiq)
    print(mean_test_determ)
    # dataframe_test_rmse = pnd.DataFrame(data=mean_test_rmse, columns=columns).transpose()
    # dataframe_test_rpiq = pnd.DataFrame(data=mean_test_rpiq, columns=columns).transpose()
    # dataframe_test_determ = pnd.DataFrame(data=mean_test_determ, columns=columns).transpose()
    extractDiagram(values_x, mean_test_rmse, args.parameterName, 'Test RMSE', path_datetime+'/RMSE_'+args.parameterSymbol+'.svg')
    extractDiagram(values_x, mean_test_rpiq, args.parameterName, 'Test RPIQ', path_datetime+'/RPIQ_'+args.parameterSymbol+'.svg')
    extractDiagram(values_x, mean_test_determ, args.parameterName, 'Test R^2', path_datetime+'/Determ_'+args.parameterSymbol+'.svg')