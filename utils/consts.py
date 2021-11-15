# This script defines all the constant values used further in the different scripts

import pathlib
import os
import numpy as np

# Labels definition
PATIENT_LABEL_UNKNOWN = 5
PATIENT_LABEL_NON_AF = 0
PATIENT_LABEL_AF_MILD = 1
PATIENT_LABEL_AF_MODERATE = 2
PATIENT_LABEL_AF_SEVERE = 3
PATIENT_LABEL_OTHER_CVD = 4

PATIENT_LABELS = np.array([PATIENT_LABEL_NON_AF, PATIENT_LABEL_AF_MILD, PATIENT_LABEL_AF_MODERATE, PATIENT_LABEL_AF_SEVERE, PATIENT_LABEL_OTHER_CVD, PATIENT_LABEL_UNKNOWN])

WINDOW_LABEL_UNKNOWN = -1
WINDOW_LABEL_NON_AF = 0
WINDOW_LABEL_AF = 1
WINDOW_LABEL_OTHER = 2

WINDOW_LABELS = np.array([WINDOW_LABEL_NON_AF, WINDOW_LABEL_AF, WINDOW_LABEL_OTHER, WINDOW_LABEL_UNKNOWN])

# Feature names
BASELINE_FEATURES = np.array(['cosEn', 'AFEv', 'OriginCount', 'IrrEv', 'PACEv', 'AVNN', 'minRR', 'medFreq'])
HRV_FEATURES = np.array(['AVNN', 'medRR', 'minRR', 'maxRR', 'VarRR', 'VardRR', 'VarddRR', 'CoeffVarRR', 'RMSSD'])
POINCARE_FEATURES = np.array(['poinc_sd1', "poinc_sd2", "poinc_ratio"])
IMPLEMENTED_FEATURES = np.array(['cosEn', 'AFEv', 'OriginCount', 'IrrEv', 'PACEv', 'AVNN', 'minRR', 'medHR',
                                 'SDNN', 'SEM', 'PNN20', 'PNN50', 'RMSSD', 'CV', 'SD1', 'SD2', 'sq_map_intercept',
                                 'sq_map_linear', 'sq_map_quadratic', 'PIP', 'IALS', 'PSS', 'PAS'])
SELECTED_FEATURES = np.array(['cosEn', 'AFEv', 'OriginCount', 'IrrEv', 'PACEv', 'AVNN', 'minRR', 'medHR',
                            'SDNN', 'SEM', 'PNN20', 'PNN50', 'RMSSD', 'CV', 'SD1', 'SD2', 'PIP', 'IALS', 'PSS', 'PAS'])
ANNOTATION_TYPES = np.array(['epltd0', 'xqrs', 'gqrs', 'rqrs', 'jqrs', 'wqrs', 'wavedet', 'wrqrs'])

# Label names used in graphs
BASE_WINDOWS = np.arange(60, 121, 10)
COLORS = np.array(['dodgerblue', 'orange', 'red', 'purple', 'brown', 'yellow'])
GLOBAL_LAB_TITLES = np.array(['$Non-AF$', '$AF_{Mild}$', '$AF_{Mod}$', '$AF_{Se}$', '$O$', '$Unknown$'])
BINARY_TITLES = np.array(['$Non-Prominent-AF$', '$Prominent-AF$'])
WINDOW_LAB_TITLES = np.array(['$N$', '$AF$', '$O$', '$Unknown$'])

# Constants definitions (time related)
N_S_IN_HOUR = 3600
N_MS_IN_S = 1000
N_HOURS_IN_DAY = 24
N_MIN_IN_HOUR = 60
N_SEC_IN_MIN = 60

# Parameters used for data filtering in the different scripts
RR_OUTLIER_THRESHOLD_SUP = 10           # Sup Threshold to exclude a RR interval window (excluded if one RR exceeds this value, in seconds)
RR_OUTLIER_THRESHOLD_INF = 0            # Inf Threshold to exclude a RR interval window (excluded if one RR is below this value, in seconds)
RR_FILE_THRESHOLD = 3 * N_S_IN_HOUR     # Criterion for exclusion of a whole recording (Above 3 Hours of corrupted data, file is excluded)
AF_MILD_THRESHOLD = 30                  # Threshold on AF Burden to define a patient as Mild AF patient. CAREFUL ! TIME IN SECONDS HERE
AF_MODERATE_THRESHOLD = 0.04            # Threshold on AF Burden to define a patient as Moderate AF patient
AF_SEVERE_THRESHOLD = 0.8               # Threshold on AF Burden to define a patient as Severe AF patient
SQI_FILE_THRESHOLD = 0.75               # Threshold on the number of corrupted windows (based on bsqi criterion) to exclude a file
SQI_WINDOW_THRESHOLD = 0.8              # Threshold on the bsqi criterion to exclude a window
MISSING_ANN_THRESHOLD = 0.75            # Threshold on the percentage of missing annotations to exclude a file.
OSA_AHI_THRESHOLD = 15                  # Threshold on the Apnea Hypopnea Index (AHI) to define a patient as OSA
FRAGMENTATION_LIM_SMALL_SEG = 3         # The limit to set a segment as short for fragmentation features

# Paths definition. 'nt' refers to Windows, 'posix' as Linux (triton01 GPU cluster)
if os.name == 'nt':
    BASE_DIR = pathlib.PurePath('V:\\AIMLab')
    REPO_DIR = pathlib.PurePath('C:\\Users\\armand.chocr\\Documents\\armand_repo')
    REPO_DIR_POSIX = pathlib.PurePath(str(REPO_DIR).replace('\\', '/').replace('C:','/mnt/c'))
    N_PROCESSES = 8
else:
    BASE_DIR = pathlib.PurePath('/MLdata/AIMLab')
    REPO_DIR = pathlib.PurePath('/home/armand/armand_repo')
    REPO_DIR_POSIX = REPO_DIR
    N_PROCESSES = 10

# Paths definition
DATA_DIR = BASE_DIR / "databases"
PREPROCESSED_DATA_DIR = BASE_DIR / "Armand" / "PreprocessedDatabases"
CLASSIFIERS_DIR = BASE_DIR / "Armand" / "Classifiers"
SNAPSHOTS_DIR = BASE_DIR / "Armand" / "Snapshots"
MATLAB_TEST_VECTORS_DIR = BASE_DIR / "Armand" / "Matlab_test_vectors"  # changed from DATA_DIR to BASE_DIR / "Armand"
RESULTS_DIR = BASE_DIR / "Armand" / "Results"
EPLTD_PROG_DIR = REPO_DIR_POSIX / "epltd" / "epltd_all"
WQRS_PROG_DIR = pathlib.PurePath('/usr/local/bin/wqrs')
PARSING_PROJECT_DIR = REPO_DIR_POSIX / "parsing"
ERROR_ANALYSIS_DIR = BASE_DIR / "Armand" / "ErrorAnalysis"

START_NIGHT = 22                    # Considering night starts at 10 P.M.
END_NIGHT = 5                       # Considering night starts at 5 A.M.

EPLTD_FS = 200                      # Sampling frequency of the signals which can be analyzed by EPLTD detector for peak detection

# Sign definitions for weak classifiers
INF = -1
SUP = 1

# Seed for pseudo-random number generation
SEED = 42

# Metrics computed all over the scripts
METRICS = np.array(['Accuracy', 'Fb-Score', 'Se', 'Sp', 'PPV', 'NPV', 'AUROC'])