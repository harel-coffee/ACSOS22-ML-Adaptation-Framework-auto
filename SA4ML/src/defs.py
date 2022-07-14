#!/usr/bin/env python

import platform


# """
###########
## PATHS ##
###########
# """
BASE_PATH = "/home/vagrant/SA4ML/"

BASE_DATASETS_PATH = f"{BASE_PATH}datasets/ieee-fraud-detection/"
PRISM_PATH = f"{BASE_PATH}PRISM/"
PRISM_EXEC_FILE = "test_prism_ieee_cis.sh"

"""
######################
## COMMON CONSTANTS ##
######################
"""
TIME_INTERVALS = [10]  # in hours

"""
########################################
## CONSTANTS FOR BASELINE EXPERIMENTS ##
########################################
"""
SEEDS = 1
DATASET_NAME_TEST = "ieee-cis"

ASK_PRISM = True  # whether to actually call prism to decide whether to retrain

SAT_VALUES = [0.9]
USE_NOP_MODELS = ["random_forest"]



"""
#####################################################################
## CONSTANTS FOR EXPERIMENTS TO GENERATE RETRAIN FILES AND DATASET ##
#####################################################################
"""
DATASET_NAME = "ieee-cis"
MAX_TIME = 3160

# The following definitions are only used to generate the retrain files
NUM_RUNS = 1
RETRAIN_MODES = ["single"]
