# ULDP-FL: Federated Learning with Across Silo User-Level Differentially Privacy

Tested at Python 3.9.2, OSX/Ubuntu18.04

## Setup
1. Run pip install 

    ```bash
    $ pip install -r requirements.txt
    ```


### Dataset 
1. Creditcard dataset

    Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud .
    Put `dataset/creditcard/creditcard.csv` .

2. Medical dataset

    This repository uses the medical dataset for cross-silo FL reseaches through [FLamby](https://github.com/owkin/FLamby), but does not contain the data itself. **If users want to use the data, please carefully read yourself with the license stated in FLamby.**

    We tested at release version 0.0.1 in [FLamby](https://github.com/owkin/FLamby). If you use the medical dataset. You try, for example,
    ```bash
    $  SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True `pip install 'git+https://github.com/owkin/FLamby.git@0.0.1[all_extra]'`
    ```

    **Git-clone and pip-install locally might be a better way, because some dataset needs downloading manually and running preprocess script.**

    (Tips: 1. You should download some dataset in FLamby such as [Heart Disease](https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_heart_disease/README.md) before pip-install. (This is because some parts of their implementation rely on relative paths from module files to specify data directories.) 2. When you install FLamby with `pip` in some virtual environments and get error, changing [`sys.executable`](https://github.com/owkin/FLamby/blob/4dfc53479ec4141849d67a6adace1137819317a2/setup.py#L11) path in setup.py may work. )


## Run
Run simulator mode.

In simulator mode, we don't use real communication path in the FL rounds.


Example.

```bash
$ cd src
$ python run_simulation.py --dataset_name=creditcard --verbose=1 --agg_strategy=ULDP-AVG-w --n_users=1000 --global_learning_rate=10.0 --clipping_bound=1.0 --n_total_round=100 --local_learning_rate=0.01 --local_epoch=30 --sigma=5.0 --sampling_rate_q=0.5 --user_dist=zipf --user_alpha=0.5 --silo_dist=zipf --silo_alpha=2.0
```


## Exeprimental script

Experimental scripts used in the paper are located in `exp/script`.

```bash
$ exp/script/privacy_utility.sh
$ exp/script/optimal_weighting.sh
$ exp/script/secure_weighting.sh
$ exp/script/user_level_subsampling.sh
```

## Run real server and client

Wake up server

```bash
$ cd src
$ python run_server.py --dataset_name=creditcard --verbose=1 --agg_strategy=ULDP-AVG-w --n_users=1000 --global_learning_rate=10.0 --clipping_bound=1.0 --n_total_round=100 --local_learning_rate=0.01 --local_epoch=30 --sigma=5.0 --sampling_rate_q=0.5 --user_dist=zipf --user_alpha=0.5 --silo_dist=zipf --silo_alpha=2.0 --n_silos=3 --n_silo_per_round=3
```

Wake up 3 silos

```bash
$ cd src
$ python run_silo.py --silo_id=0 --dataset_name=creditcard --verbose=1--agg_strategy=ULDP-AVG-ws --n_users=1000 --global_learning_rate=10.0 --clipping_bound=1.0 --n_total_round=100 --local_learning_rate=0.01 --local_epoch=30 --sigma=5.0 --sampling_rate_q=0.5 --user_dist=zipf --user_alpha=0.5 --silo_dist=zipf --silo_alpha=2.0
$ python run_silo.py --silo_id=1 --dataset_name=creditcard --verbose=1 --agg_strategy=ULDP-AVG-ws --n_users=1000 --global_learning_rate=10.0 --clipping_bound=1.0 --n_total_round=100 --local_learning_rate=0.01 --local_epoch=30 --sigma=5.0 --sampling_rate_q=0.5 --user_dist=zipf --user_alpha=0.5 --silo_dist=zipf --silo_alpha=2.0
$ python run_silo.py --silo_id=2 --dataset_name=creditcard --verbose=1 --agg_strategy=ULDP-AVG-ws --n_users=1000 --global_learning_rate=10.0 --clipping_bound=1.0 --n_total_round=100 --local_learning_rate=0.01 --local_epoch=30 --sigma=5.0 --sampling_rate_q=0.5 --user_dist=zipf --user_alpha=0.5 --silo_dist=zipf --silo_alpha=2.0
```


--- 

### regression tests
```bash

$ python -m unittest src.tests.test_coordinator
$ python -m unittest src.tests.test_dataset
$ python -m unittest src.tests.test_secure_aggregation
$ python -m unittest src.tests.test_simulation
```