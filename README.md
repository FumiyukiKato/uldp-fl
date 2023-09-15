# ULDP-FL: Federated Learning with Across Silo User-Level Differentially Privacy

The paper has been uploaded in arxiv https://arxiv.org/abs/2308.12210.

Tested at all of  
- `Ubuntu18.04`
- `macOS Monterey v12.1, Apple M1 Max Chip`
- docker image of `python:3.9.2-buster`

with Python 3.9.2



## Setup
1. Run pip install 

    ```bash
    $ pip install -r requirements.txt
    ```

2. Install datasets

    1. **Creditcard dataset**

        Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.
        Put `dataset/creditcard/creditcard.csv`.

    2. **Medical dataset ([HeartDisease](https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_heart_disease) and [TcgaBrca](https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_tcga_brca))**

        This repository uses the medical dataset for cross-silo FL reseaches through [FLamby](https://github.com/owkin/FLamby). **If users want to use the data, please carefully read yourself with the license stated in FLamby.**

        We tested at release version 0.0.1 in [FLamby](https://github.com/owkin/FLamby).

        - TcgaBrca
            - Preprocessed data is stored in the package in the file `flamby/datasets/fed_heart_disease/brca.csv`.
        - HeartDisease
            - We need to download the data.
            - Execute this script `./download_heart_disease_dataset.sh`.

    3. **MNIST dataset**

        We use the dataset from torchvision. https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html.

   
## Run
Run simulator mode.

In simulator mode, we don't use real communication path in the FL rounds.


Example.

```bash
$ cd src
$ python run_simulation.py --dataset_name=creditcard --verbose=1 --agg_strategy=ULDP-AVG-w --n_users=1000 --global_learning_rate=10.0 --clipping_bound=1.0 --n_total_round=100 --local_learning_rate=0.01 --local_epoch=30 --sigma=5.0 --sampling_rate_q=0.5 --user_dist=zipf --user_alpha=0.5 --silo_dist=zipf --silo_alpha=2.0
```

List of options: `src/options.py`

## Exeprimental script

Experimental scripts including the all of hyperparameters used in the paper are located in `exp/script`.

```bash
$ exp/script/privacy_utility.sh
$ exp/script/optimal_weighting.sh
$ exp/script/secure_weighting.sh
$ exp/script/user_level_subsampling.sh
```

## Run real server and client

GRPC-based implementation (reference implementation: [FedML](https://github.com/FedML-AI/FedML)).

Wake up server

```bash
$ cd src
$ python run_server.py --dataset_name=creditcard --verbose=1 --agg_strategy=ULDP-AVG-w --n_users=1000 --global_learning_rate=10.0 --clipping_bound=1.0 --n_total_round=100 --local_learning_rate=0.01 --local_epoch=30 --sigma=5.0 --user_dist=zipf --user_alpha=0.5 --silo_dist=zipf --silo_alpha=2.0 --n_silos=3 --n_silo_per_round=3
```

Wake up 3 silos

```bash
$ cd src
$ python run_silo.py --silo_id=0 --dataset_name=creditcard --verbose=1 --agg_strategy=ULDP-AVG-w --n_users=1000 --global_learning_rate=10.0 --clipping_bound=1.0 --n_total_round=100 --local_learning_rate=0.01 --local_epoch=30 --sigma=5.0 --user_dist=zipf --user_alpha=0.5 --silo_dist=zipf --silo_alpha=2.0 --n_silos=3 --n_silo_per_round=3
$ python run_silo.py --silo_id=1 --dataset_name=creditcard --verbose=1 --agg_strategy=ULDP-AVG-w --n_users=1000 --global_learning_rate=10.0 --clipping_bound=1.0 --n_total_round=100 --local_learning_rate=0.01 --local_epoch=30 --sigma=5.0 --user_dist=zipf --user_alpha=0.5 --silo_dist=zipf --silo_alpha=2.0 --n_silos=3 --n_silo_per_round=3
$ python run_silo.py --silo_id=2 --dataset_name=creditcard --verbose=1 --agg_strategy=ULDP-AVG-w --n_users=1000 --global_learning_rate=10.0 --clipping_bound=1.0 --n_total_round=100 --local_learning_rate=0.01 --local_epoch=30 --sigma=5.0 --user_dist=zipf --user_alpha=0.5 --silo_dist=zipf --silo_alpha=2.0 --n_silos=3 --n_silo_per_round=3
```


--- 

### regression tests
```bash

$ python -m unittest src.tests.test_coordinator
$ python -m unittest src.tests.test_dataset
$ python -m unittest src.tests.test_secure_aggregation
$ python -m unittest src.tests.test_simulation
```
