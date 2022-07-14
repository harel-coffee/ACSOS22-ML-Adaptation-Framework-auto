#!/usr/bin/env python

"""
#####################################
## EXPERIMENTAL SETTINGS/VARIABLES ##
#####################################
"""

RETRAIN_COSTS = [8]
RETRAIN_LATENCIES = [0] # in hours
FPR_T = [1]  # in %
RECALL_T = [70]  # in %
SLA_COSTS = [10]

BASELINES = [
    "optimum",
    "no_retrain",
    "periodic",
    "reactive",
    "random",
    "delta_retrain", # this is AIP
]
