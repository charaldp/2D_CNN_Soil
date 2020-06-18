import sys,math,random,os,argparse, scipy
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pnd
import argparse as argp
from datetime import datetime

def parse_args():
    parser = argp.ArgumentParser(description='Diagram extractor')
    # Required
    parser.add_argument('-n','--name',type=str, help='A specific name for the plot group',default='diag')
    parser.add_argument('-fld','--folders',type=str, help='Model folders to process',default=[''], nargs='+')
    parser.add_argument('-md','--mode',type=str, help='Mode to use script 1) \'metrics\' ',default='metrics')
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

OUTPUT_PATH = '../output'
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
datetime_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")+'/'
path_datetime = os.path.join(OUTPUT_PATH, datetime_str)
# if not os.path.exists(path_datetime):
#     os.mkdir(path_datetime)
args = parse_args()
print(args)

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
    for folder in args.folders:
        data_test = pnd.read_csv(folder+'/test_predictions.csv', index_col=False)
        data_val = pnd.read_csv(folder+'/val_predictions.csv', index_col=False)
        data_train = pnd.read_csv(folder+'/train_predictions.csv', index_col=False)
        cols = data_test.columns
        props = []
        for col in cols:
            if col != cols[0]:
                prop = col[0: col.find('_')]
                if not prop in props:
                    props.append(prop)
        print(props)
        for prop in props:
            # fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(6, 6), sharey=True)
            # axs[1, 2].bxp(stats, showfliers=False)
            # axs[1, 2].set_title('showfliers=False', fontsize=fs)
            errors_test = data_test[prop+'_test_preds'] - data_test[prop+'_test_actuals']
            errors_val = data_val[prop+'_val_preds'] - data_val[prop+'_val_actuals']
            errors_train = data_train[prop+'_train_preds'] - data_train[prop+'_train_actuals']
            # print('N_test_preds', data_test['N_test_preds'])
            # print('N_test_actuals', data_test['N_test_actuals'],)
            # print('errors', errors)
            data = [np.absolute(errors_test), np.absolute(errors_val), np.absolute(errors_train)]
            fig7, ax7 = plt.subplots()
            ax7.set_title(prop+' Test-Val-Train RMSE')
            ax7.boxplot(data)
            plt.show()

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
