#! /bin/bash
set -eux

# abs path of this script
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

## heart_disease
n_users_list=(10 100)

for n_users in "${n_users_list[@]}"
do
    python $RUN_SIMULATION_PATH --dataset_name=heart_disease --verbose=1 --agg_strategy=ULDP-AVG-w --n_users=$n_users --global_learning_rate=10.0 --clipping_bound=0.1 --n_total_round=3 --local_learning_rate=0.001 --local_epochs=50 --sigma=5.0 --user_dist=zipf --user_alpha=0.5 --silo_dist=zipf --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN --secure_w
done

## tcga_brca
n_users_list=(10 100)

for n_users in "${n_users_list[@]}"
do
    python $RUN_SIMULATION_PATH --dataset_name=tcga_brca --verbose=1 --agg_strategy=ULDP-AVG-w  --n_users=$n_users --global_learning_rate=10.0 --clipping_bound=0.1 --n_total_round=3 --local_learning_rate=0.001 --local_epochs=50 --sigma=5.0 --user_dist=zipf --user_alpha=0.5 --silo_dist=zipf --silo_alpha=2.0 --times=$TIMES --version=$VERSION $DRY_RUN --secure_w
done
