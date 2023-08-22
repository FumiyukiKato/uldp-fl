#! /bin/bash
set -eux

# # abs path of this script
SCRIPT_PATH="$(cd "$(dirname "$0")"; pwd -P)"
RUN_SIMULATION_PATH="$SCRIPT_PATH/../../src/run_simulation.py"

VERSION=0
TIMES=5

DRY_RUN=""
for arg in "$@"
do
    if [ "$arg" == "--dry_run" ]; then
        # --dry_run
        DRY_RUN="--dry_run"
    fi
done

# GPU="--gpu=0"
GPU=""


## creditcard
n_silos_list=(5 20 50)
dist_list=("uniform" "zipf")
method_list=("ULDP-AVG" "ULDP-AVG-w")

for n_silos in "${n_silos_list[@]}"
do
    for dist in "${dist_list[@]}"
    do
        for method in "${method_list[@]}"
        do
            python $RUN_SIMULATION_PATH --dataset_name=creditcard --verbose=1 --agg_strategy=$method --n_users=1000 --global_learning_rate=10.0 --clipping_bound=1.0 --n_total_round=100 --local_learning_rate=0.01 --local_epochs=30 --sigma=5.0 --user_dist=$dist --user_alpha=0.5 --silo_dist=$dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN --n_silos=$n_silos --n_silo_per_round=$n_silos
        done
    done
done



## tcga_brca
dist_list=("uniform" "zipf")
method_list=("ULDP-AVG" "ULDP-AVG-w")

for dist in "${dist_list[@]}"
do
    for method in "${method_list[@]}"
    do
        python $RUN_SIMULATION_PATH --dataset_name=tcga_brca --verbose=1 --agg_strategy=$method --n_users=200 --global_learning_rate=10.0 --clipping_bound=0.1 --n_total_round=50 --local_learning_rate=0.001 --local_epochs=50 --sigma=5.0 --user_dist=$dist --user_alpha=0.5 --silo_dist=$dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
    done
done
