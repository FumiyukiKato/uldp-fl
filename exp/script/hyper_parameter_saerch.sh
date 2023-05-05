#! /bin/bash

# This script is used to search the hyper parameters of the model.

# usage: 
# script/hyper_parameter_saerch.sh

# AVG
python ../src/run_simulation.py --agg_strategy=ULDP-AVG --sigma=5.0 --hp_tune=1 --n_total_round=4 --times=5

# SGD
python ../src/run_simulation.py --agg_strategy=ULDP-SGD --sigma=5.0 --hp_tune=1 --n_total_round=4 --times=5