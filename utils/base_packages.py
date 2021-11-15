# The main packages used all along the project are loaded here and
# divided by categories
# This script also adds to PATH the different important
# path containing the basic repo directories (utils and parsing)

try:
    import utils.consts as cts
except ModuleNotFoundError:
    import consts as cts
import os
import shutil
import sys
import warnings

from getpass import getuser

# Data Processing
import numpy as np
import pandas as pd
import scipy.signal as signal
import multiprocessing
import subprocess
import itertools
import math

# Graphics
#import matplotlib.pyplot as plt

# I/O
import csv
import xlrd
import xlsxwriter
import scipy.io as sio
import wfdb
# import pyedflib
from sklearn.externals import joblib
from termcolor import colored
import zipfile
import io
import copy
import random
# import pyedflib
import pickle

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.utils.class_weight as skl_cw
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

#os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

if getuser() == 'itsno':
    sys.path.append(r'C:\Users\itsno\PycharmProjects\ECG_compression\parsing')
    sys.path.append(r'C:\Users\itsno\PycharmProjects\ECG_compression\utils')
    sys.path.append('/home/noacohen/git/ECG_compression')
    sys.path.append('/home/noacohen/git/ECG_compression/utils')
else:
    sys.path.append('C:\\Users\\noamb\\projectB\\ECG_compression\\parsing\\')
    sys.path.append('C:\\Users\\noamb\\projectB\\ECG_compression\\utils\\')
    sys.path.append('/home/bnoam/ECG_compression/parsing')
    sys.path.append('/home/bnoam/ECG_compression/utils')

np.random.seed(cts.SEED)
random.seed(cts.SEED)
warnings.filterwarnings('ignore')

