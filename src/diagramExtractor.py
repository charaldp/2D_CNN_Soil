import sys,math,random,os,argparse, scipy
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pnd
import argparse as argp

def parse_args():
    parser = argp.ArgumentParser(description='Diagram extractor')
    # Required
    parser.add_argument('-n','--name',type=str, help='A specific name for the plot group',default='diag')
    parser.add_argument('-fl','--files',type=str, help='CSV meterics files to process',default=[''], nargs='+')
    parser.add_argument('-md','--mode',type=str, help='Mode to use script 1) \'metrics\' ',default='metrics')
    args = parser.parse_args()
    return args

args = parse_args()
print(args)
if args.mode == 'metrics':
    for filename in args.files:
        data = pnd.read_csv(filename)
        columns = data.columns
        datrans = data.T
        print(datrans)

        for col in datrans.columns:
            print(datrans[col])
