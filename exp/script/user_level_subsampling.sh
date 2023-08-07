#! /bin/bash
set -eux

# # abs path of this scripto
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
python $RUN_SIMULATION_PATH --dataset_name=creditcard --verbose=1 --agg_strategy=ULDP-AVG-w --n_users=1000 --global_learning_rate=10.0 --clipping_bound=1.0 --n_total_round=100 --local_learning_rate=0.01 --local_epochs=30 --sigma=5.0 --user_dist=zipf --user_alpha=0.5 --silo_dist=zipf --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN

sampling_rate_q_list=(0.1 0.3 0.5 0.7)

for sampling_rate_q in "${sampling_rate_q_list[@]}"
do
    python $RUN_SIMULATION_PATH --dataset_name=creditcard --verbose=1 --agg_strategy=ULDP-AVG-ws --n_users=1000 --global_learning_rate=10.0 --clipping_bound=1.0 --n_total_round=100 --local_learning_rate=0.01 --local_epochs=30 --sigma=5.0 --sampling_rate_q=$sampling_rate_q --user_dist=zipf --user_alpha=0.5 --silo_dist=zipf --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN
done


## mnist
python $RUN_SIMULATION_PATH --dataset_name=mnist --verbose=1 --agg_strategy=ULDP-AVG-w --n_users=10000 --global_learning_rate=100.0 --clipping_bound=0.1 --n_total_round=200 --local_learning_rate=0.01 --local_epochs=50 --sigma=5.0 --user_dist=zipf --user_alpha=0.5 --silo_dist=zipf --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN $GPU

sampling_rate_q_list=(0.1 0.3 0.5)

for sampling_rate_q in "${sampling_rate_q_list[@]}"
do
    python $RUN_SIMULATION_PATH --dataset_name=mnist --verbose=1 --agg_strategy=ULDP-AVG-ws --n_users=10000 --global_learning_rate=100.0 --clipping_bound=0.1 --n_total_round=200 --local_learning_rate=0.01 --local_epochs=50 --sigma=5.0 --sampling_rate_q=$sampling_rate_q --user_dist=zipf --user_alpha=0.5 --silo_dist=zipf --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN $GPU
done
