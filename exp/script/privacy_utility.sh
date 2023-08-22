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

# to restart from a specific counter
counter=0
RESTART=0
# Error Handler
trap 'echo "Error occurred at counter=$counter"; exit 1' ERR

## creditcard
n_users_list=(100 1000)
silo_dist_list=("uniform" "zipf")
user_dist_list=("uniform" "zipf-iid")
dist_length=${#silo_dist_list[@]}

for n_users in "${n_users_list[@]}"
do
    for ((i=0; i<$dist_length; i++)); do
        user_dist=${user_dist_list[$i]}
        silo_dist=${silo_dist_list[$i]}

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=creditcard --verbose=1 --agg_strategy="DEFAULT" --n_users=$n_users --global_learning_rate=1.0 --n_total_round=100 --local_learning_rate=0.01 --local_epochs=50 --user_dist=$user_dist  --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=creditcard --verbose=1 --agg_strategy="ULDP-NAIVE" --n_users=$n_users --global_learning_rate=1.0 --clipping_bound=0.1 --n_total_round=100 --local_learning_rate=0.01 --local_epochs=50 --user_dist=$user_dist  --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=creditcard --verbose=1 --agg_strategy="ULDP-GROUP-max" --n_users=$n_users --global_learning_rate=1.0 --clipping_bound=0.1 --n_total_round=100 --local_learning_rate=0.01 --local_epochs=50 --user_dist=$user_dist  --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=creditcard --verbose=1 --agg_strategy="ULDP-GROUP-median" --n_users=$n_users --global_learning_rate=1.0 --clipping_bound=0.1 --n_total_round=100 --local_learning_rate=0.01 --local_epochs=50 --user_dist=$user_dist  --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=creditcard --verbose=1 --agg_strategy="ULDP-GROUP" --group_k=2 --n_users=$n_users --global_learning_rate=1.0 --clipping_bound=0.1 --n_total_round=100 --local_learning_rate=0.01 --local_epochs=50 --user_dist=$user_dist  --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=creditcard --verbose=1 --agg_strategy="ULDP-GROUP" --group_k=8 --n_users=$n_users --global_learning_rate=1.0 --clipping_bound=0.1 --n_total_round=100 --local_learning_rate=0.01 --local_epochs=50 --user_dist=$user_dist  --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=creditcard --verbose=1 --agg_strategy="ULDP-SGD" --n_users=$n_users --global_learning_rate=10.0 --clipping_bound=0.1 --n_total_round=100 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=creditcard --verbose=1 --agg_strategy="ULDP-AVG" --n_users=$n_users --global_learning_rate=10.0 --clipping_bound=0.1 --n_total_round=100 --local_learning_rate=0.01 --local_epochs=30 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=creditcard --verbose=1 --agg_strategy="ULDP-AVG-w" --n_users=$n_users --global_learning_rate=10.0 --clipping_bound=0.1 --n_total_round=100 --local_learning_rate=0.01 --local_epochs=30 --user_dist=$user_dist  --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))
    done
done

## heart_disease
n_users_list=(50 200)
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
            python $RUN_SIMULATION_PATH --dataset_name=heart_disease --verbose=1 --agg_strategy="DEFAULT" --n_users=$n_users --global_learning_rate=1.0  --n_total_round=50 --local_learning_rate=0.001 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=heart_disease --verbose=1 --agg_strategy="ULDP-NAIVE" --n_users=$n_users --global_learning_rate=1.0 --clipping_bound=0.1 --n_total_round=50 --local_learning_rate=0.001 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=heart_disease --verbose=1 --agg_strategy="ULDP-GROUP-max" --n_users=$n_users --global_learning_rate=1.0 --clipping_bound=0.1 --n_total_round=50 --local_learning_rate=0.001 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=heart_disease --verbose=1 --agg_strategy="ULDP-GROUP-median" --n_users=$n_users --global_learning_rate=1.0 --clipping_bound=0.1 --n_total_round=50 --local_learning_rate=0.001 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=heart_disease --verbose=1 --agg_strategy="ULDP-GROUP" --group_k=2 --n_users=$n_users --global_learning_rate=1.0 --clipping_bound=0.1 --n_total_round=50 --local_learning_rate=0.001 --local_epochs=50 --user_dist=$user_dist  --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=heart_disease --verbose=1 --agg_strategy="ULDP-GROUP" --group_k=8 --n_users=$n_users --global_learning_rate=1.0 --clipping_bound=0.1 --n_total_round=50 --local_learning_rate=0.001 --local_epochs=50 --user_dist=$user_dist  --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=heart_disease --verbose=1 --agg_strategy="ULDP-SGD" --n_users=$n_users --global_learning_rate=10.0 --clipping_bound=0.1 --n_total_round=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=heart_disease --verbose=1 --agg_strategy="ULDP-AVG" --n_users=$n_users --global_learning_rate=10.0 --clipping_bound=0.1 --n_total_round=50 --local_learning_rate=0.001 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=heart_disease --verbose=1 --agg_strategy="ULDP-AVG-w" --n_users=$n_users --global_learning_rate=10.0 --clipping_bound=0.1 --n_total_round=50 --local_learning_rate=0.001 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))
    done
done


## tcga_brca
n_users_list=(50 200)
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
            python $RUN_SIMULATION_PATH --dataset_name=tcga_brca --verbose=1 --agg_strategy="DEFAULT" --n_users=$n_users --global_learning_rate=1.0  --n_total_round=50 --local_learning_rate=0.001 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=tcga_brca --verbose=1 --agg_strategy="ULDP-NAIVE" --n_users=$n_users --global_learning_rate=1.0 --clipping_bound=0.1 --n_total_round=50 --local_learning_rate=0.001 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=tcga_brca --verbose=1 --agg_strategy="ULDP-GROUP-max" --n_users=$n_users --global_learning_rate=0.1 --clipping_bound=100.0 --n_total_round=50 --local_learning_rate=0.001 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=tcga_brca --verbose=1 --agg_strategy="ULDP-GROUP-median" --n_users=$n_users --global_learning_rate=0.1 --clipping_bound=100.0 --n_total_round=50 --local_learning_rate=0.001 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=tcga_brca --verbose=1 --agg_strategy="ULDP-GROUP" --group_k=2 --n_users=$n_users --global_learning_rate=0.1 --clipping_bound=100.0 --n_total_round=50 --local_learning_rate=0.001 --local_epochs=50 --user_dist=$user_dist  --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=tcga_brca --verbose=1 --agg_strategy="ULDP-GROUP" --group_k=8 --n_users=$n_users --global_learning_rate=0.1 --clipping_bound=100.0 --n_total_round=50 --local_learning_rate=0.001 --local_epochs=50 --user_dist=$user_dist  --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=tcga_brca --verbose=1 --agg_strategy="ULDP-SGD" --n_users=$n_users --global_learning_rate=10.0 --clipping_bound=0.1 --n_total_round=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=tcga_brca --verbose=1 --agg_strategy="ULDP-AVG" --n_users=$n_users --global_learning_rate=10.0 --clipping_bound=0.1 --n_total_round=50 --local_learning_rate=0.001 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=tcga_brca --verbose=1 --agg_strategy="ULDP-AVG-w" --n_users=$n_users --global_learning_rate=10.0 --clipping_bound=0.1 --n_total_round=50 --local_learning_rate=0.001 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
        fi
        ((counter+=1))
    done
done



## mnist
TIMES=5
N_TOTAL_ROUND=200
n_users_list=(100 10000)
global_learning_rate_list=(10.0 100.0)
n_users_length=${#n_users_list[@]}
silo_dist_list=("uniform" "zipf" "zipf")
user_dist_list=("uniform" "zipf-iid" "zipf-noniid")
dist_length=${#silo_dist_list[@]}

for ((j=0; j<$n_users_length; j++)); do
    for ((i=0; i<$dist_length; i++)); do
        n_users=${n_users_list[$j]}
        global_learning_rate=${global_learning_rate_list[$j]}
        user_dist=${user_dist_list[$i]}
        silo_dist=${silo_dist_list[$i]}

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=mnist --verbose=1 --agg_strategy="DEFAULT" --n_users=$n_users --global_learning_rate=1.0 --n_total_round=$N_TOTAL_ROUND --local_learning_rate=0.01 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION --n_labels=2 $DRY_RUN $GPU
        fi
        ((counter+=1))

        ## Too Bad!
        # if ((counter >= RESTART)); then
        #   python $RUN_SIMULATION_PATH --dataset_name=mnist --verbose=1 --agg_strategy="ULDP-NAIVE" --n_users=$n_users --global_learning_rate=1.0 --clipping_bound=1.0 --n_total_round=$N_TOTAL_ROUND --local_learning_rate=0.01 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION --n_labels=2 $DRY_RUN $GPU
        # fi
        # ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=mnist --verbose=1 --agg_strategy="ULDP-GROUP-max" --n_users=$n_users --global_learning_rate=1.0 --clipping_bound=1.0 --n_total_round=$N_TOTAL_ROUND --local_learning_rate=0.01 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION --n_labels=2 $DRY_RUN $GPU
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=mnist --verbose=1 --agg_strategy="ULDP-GROUP-median" --n_users=$n_users --global_learning_rate=1.0 --clipping_bound=1.0 --n_total_round=$N_TOTAL_ROUND --local_learning_rate=0.01 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION --n_labels=2 $DRY_RUN $GPU
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=mnist --verbose=1 --agg_strategy="ULDP-GROUP" --group_k=2 --n_users=$n_users --global_learning_rate=1.0 --clipping_bound=1.0 --n_total_round=$N_TOTAL_ROUND --local_learning_rate=0.01 --local_epochs=50 --user_dist=$user_dist  --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION --n_labels=2 $DRY_RUN $GPU
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=mnist --verbose=1 --agg_strategy="ULDP-GROUP" --group_k=8 --n_users=$n_users --global_learning_rate=1.0 --clipping_bound=1.0 --n_total_round=$N_TOTAL_ROUND --local_learning_rate=0.01 --local_epochs=50 --user_dist=$user_dist  --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION --n_labels=2 $DRY_RUN $GPU
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=mnist --verbose=1 --agg_strategy="ULDP-SGD" --n_users=$n_users --global_learning_rate=10.0 --clipping_bound=0.1 --n_total_round=$N_TOTAL_ROUND --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION --n_labels=2 $DRY_RUN $GPU
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=mnist --verbose=1 --agg_strategy="ULDP-AVG" --n_users=$n_users --global_learning_rate=$global_learning_rate --clipping_bound=0.1 --n_total_round=$N_TOTAL_ROUND --local_learning_rate=0.01 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION --n_labels=2 $DRY_RUN $GPU
        fi
        ((counter+=1))

        if ((counter >= RESTART)); then
            python $RUN_SIMULATION_PATH --dataset_name=mnist --verbose=1 --agg_strategy="ULDP-AVG-w" --n_users=$n_users --global_learning_rate=$global_learning_rate --clipping_bound=0.1 --n_total_round=$N_TOTAL_ROUND --local_learning_rate=0.01 --local_epochs=50 --user_dist=$user_dist --user_alpha=0.5 --silo_dist=$silo_dist --silo_alpha=2.0 --times=$TIMES --version=$VERSION --n_labels=2 $DRY_RUN $GPU
        fi
        ((counter+=1))
    done
done
