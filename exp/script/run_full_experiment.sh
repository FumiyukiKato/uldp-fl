#! /bin/bash


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


methods=("DEFAULT" "ULDP-NAIVE" "ULDP-GROUP" "ULDP-SGD" "ULDP-AVG")
dists=(0 1)
n_users=(100 10000)

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
