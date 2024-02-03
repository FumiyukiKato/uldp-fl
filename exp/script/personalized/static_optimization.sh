#! /bin/bash
set -eux

# # abs path of this script
SCRIPT_PATH="$(cd "$(dirname "$0")"; pwd -P)"
RUN_SIMULATION_PATH="$SCRIPT_PATH/../../../src/run_static_optimization.py"

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

# to restart from a specific counter
counter=0
RESTART=0
# Error Handler
trap 'echo "Error occurred at counter=$counter"; exit 1' ERR


## heart_disease
n_users_list=(50 400)
n_users_length=${#n_users_list[@]}
silo_dist_list=("uniform" "zipf")
user_dist_list=("uniform" "zipf-iid")
dist_length=${#silo_dist_list[@]}

for ((j=0; j<$n_users_length; j++)); do
    for ((i=0; i<$dist_length; i++)); do
        n_users=${n_users_list[$j]}
        user_dist=${user_dist_list[$i]}
        silo_dist=${silo_dist_list[$i]}

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=heart_disease --verbose=1 --n_users=$n_users --n_total_round=20  --global_learning_rate=10.0 --local_learning_rate=0.001 --local_epochs=30 --sigma=1.0 --epsilon_list=[0.5] --group_thresholds=[0.5] --ratio_list=[1.0] --q_step_size=0.8 --validation_ratio=0.0 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))
    done
done

