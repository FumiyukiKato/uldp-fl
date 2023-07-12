#! /bin/bash
set -eux

TIMES=5
dataset_list=("creditcatd")
dist_list=("uniform" "zipf")
n_users_list=(1000)
method_list=("ULDP-AVG" "ULDP-AVG-s" "ULDP-AVG-w" "ULDP-AVG-ws")



for dataset_name in "${dataset_list[@]}"
do
    for dist in "${dist_list[@]}"
    do
        for n_users in "${n_users_list[@]}"
        do
            for method in "${method_list[@]}"
            do
                python run_simulation.py --dataset_name=$dataset_name --verbose=1 --agg_strategy=$method --n_users=$n_users --global_learning_rate=10.0 --clipping_bound=0.1 --n_total_round=50 --local_learning_rate=0.001 --local_epoch=50 --sigma=5.0 --user_dist=$dist --user_alpha=1.0 --silo_dist=$dist --silo_alpha=1.5 --times=$TIMES
            done
        done
    done
done

for dataset_name in "${dataset_list[@]}"
do
    for dist in "${dist_list[@]}"
    do
        for n_users in "${n_users_list[@]}"
        do
            for method in "${method_list[@]}"
            do
                python run_simulation.py --dataset_name=$dataset_name --verbose=1 --agg_strategy=$method --n_users=$n_users --global_learning_rate=10.0 --clipping_bound=1.0 --n_total_round=50 --local_learning_rate=0.01 --local_epoch=20 --sigma=5.0 --user_dist=$dist --user_alpha=1.0 --silo_dist=$dist --silo_alpha=1.5 --times=$TIMES --dry_run
            done
        done
    done
done