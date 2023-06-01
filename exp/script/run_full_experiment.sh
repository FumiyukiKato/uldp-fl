#! /bin/bash
set -eux

TIMES=5

# methods=("ULDP-AVG-w" "DEFAULT" "ULDP-NAIVE" "ULDP-AVG")
methods=("ULDP-AVG-w")
dists=(1)
n_users_list=(100 10000)
n_total_round_list=(10 50)
group_k_list=(2 8)
med_dataset_names=("heart_disease" "tcga_brca")
med_n_users_list=(10 100 1000)


# Mnist dataset
for method in "${methods[@]}"
do
    for dist in "${dists[@]}"
    do
        for n_users in "${n_users_list[@]}"
        do
            for n_total_round in "${n_total_round_list[@]}"
            do
                python script/utility_experiment.py --dataset_name=mnist --exp_dist=$dist --agg_strategy=$method --n_users=$n_users --n_total_round=$n_total_round --times=$TIMES
            done
        done
    done
done

# for group_k in "${group_k_list[@]}"
# do
#     for dist in "${dists[@]}"
#     do
#         for n_users in "${n_users_list[@]}"
#         do
#             for n_total_round in "${n_total_round_list[@]}"
#             do
#                 python script/utility_experiment.py --dataset_name=mnist --exp_dist=$dist --agg_strategy=ULDP-GROUP --n_users=$n_users --n_total_round=$n_total_round --times=$TIMES
#             done
#         done
#     done
# done


# # Mdeical dataset
# for method in "${methods[@]}"
# do
#     for dataset_name in "${med_dataset_names[@]}"
#     do
#         for n_users in "${med_n_users_list[@]}"
#         do
#             for n_total_round in "${n_total_round_list[@]}"
#             do
#                 python script/utility_experiment.py --dataset_name=$dataset_name --agg_strategy=$method --n_users=$n_users --n_total_round=$n_total_round --times=$TIMES
#             done
#         done
#     done
# done

# for group_k in "${group_k_list[@]}"
# do
#     for dataset_name in "${med_dataset_names[@]}"
#     do
#         for n_users in "${med_n_users_list[@]}"
#         do
#             for n_total_round in "${n_total_round_list[@]}"
#             do
#                 python script/utility_experiment.py --dataset_name=$dataset_name --agg_strategy=ULDP-GROUP --n_users=$n_users --group_k=$group_k --n_total_round=$n_total_round --times=$TIMES
#             done
#         done
#     done
# done
