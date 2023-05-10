#! /bin/bash
set -eux


methods=("DEFAULT" "ULDP-NAIVE" "ULDP-GROUP" "ULDP-SGD" "ULDP-AVG")
dists=(0 1)
n_users=(100 10000)

for method in "${methods[@]}"
do
    for dist in "${dists[@]}"
    do
        for n_user in "${n_users[@]}"
        do
            python script/utility_experiment.py --hyper_parameter_tuning=1 --dataset_name=mnist --exp_dist=$dist --agg_strategy=$method --n_users=$n_user
        done
    done
done

for method in "${methods[@]}"
do
    for dist in "${dists[@]}"
    do
        for n_user in "${n_users[@]}"
        do
            python script/utility_experiment.py --dataset_name=mnist --exp_dist=$dist --agg_strategy=$method --n_users=$n_user
        done
    done
done


methods=("DEFAULT" "ULDP-NAIVE" "ULDP-GROUP" "ULDP-SGD" "ULDP-AVG")
dataset_names=("heart_disease" "tcga_brca")
n_users_list=(10 100 1100)

for method in "${methods[@]}"
do
    for dataset_name in "${dataset_names[@]}"
    do
        for n_users in "${n_users_list[@]}"
        do
            python script/utility_experiment.py --hyper_parameter_tuning=1 --dataset_name=$dataset_name --agg_strategy=$method --n_users=$n_users
        done
    done
done

for method in "${methods[@]}"
do
    for dataset_name in "${dataset_names[@]}"
    do
        for n_users in "${n_users_list[@]}"
        do
            python script/utility_experiment.py --dataset_name=$dataset_name --agg_strategy=$method --n_users=$n_users
        done
    done
done