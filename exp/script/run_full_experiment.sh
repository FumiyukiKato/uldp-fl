#! /bin/bash
set -eux

TIMES=1

methods=("ULDP-SGD" "ULDP-AVG" "DEFAULT") # "ULDP-NAIVE" is too bad.
dists=(0 1)
n_users=(100 10000)

for method in "${methods[@]}"
do
    for dist in "${dists[@]}"
    do
        for n_user in "${n_users[@]}"
        do
            python script/utility_experiment.py --dataset_name=mnist --exp_dist=$dist --agg_strategy=$method --n_users=$n_user --times=$TIMES
        done
    done
done

group_k_list=(2 4 8)

for group_k in "${group_k_list[@]}"
do
    for n_users in "${n_users_list[@]}"
    do
        python script/utility_experiment.py --dataset_name=mnist --agg_strategy=ULDP-GROUP --n_users=$n_users --group_k=$group_k --times=$TIMES
    done
done


methods=("DEFAULT" "ULDP-NAIVE" "ULDP-SGD" "ULDP-AVG")
dataset_names=("heart_disease" "tcga_brca")
n_users_list=(10 100 1100)

for method in "${methods[@]}"
do
    for dataset_name in "${dataset_names[@]}"
    do
        for n_users in "${n_users_list[@]}"
        do
            python script/utility_experiment.py --dataset_name=$dataset_name --agg_strategy=$method --n_users=$n_users --times=$TIMES
        done
    done
done


group_k_list=(2 4 8)

for group_k in "${group_k_list[@]}"
do
    for dataset_name in "${dataset_names[@]}"
    do
        for n_users in "${n_users_list[@]}"
        do
            python script/utility_experiment.py --dataset_name=$dataset_name --agg_strategy=ULDP-GROUP --n_users=$n_users --group_k=$group_k --times=$TIMES
        done
    done
done