# Acsilo

tested at Python 3.9.2, OSX, Ubuntu18.04

## Setup FL
0. Git clone


1. Run pip install 

    ```bash
    $ pip install -r requirements.txt
    ```


2. Run simulator mode

    In simulator mode, we don't use real communication path in the FL rounds.

    ```bash
    $ cd src
    $ python run_simulation.py --n_silos=3 --n_silo_per_round=2 --n_total_round=5 --local_batch_size=64 --learning_rate=0.05
    ```


3. Run real mode (server and silos based on grpc)

    Wake up server
    ```bash
    $ cd src
    $ python run_server.py --n_silos=3 --n_silo_per_round=2 --n_total_round=3
    ```

    Wake up 3 silos
    ```bash
    $ cd src
    $ python run_silo.py --n_silos=3 --n_silo_per_round=2 --silo_id=0 --n_total_round=3
    $ python run_silo.py --n_silos=3 --n_silo_per_round=2 --silo_id=1 --n_total_round=3
    $ python run_silo.py --n_silos=3 --n_silo_per_round=2 --silo_id=2 --n_total_round=3
    ```


4. Medical dataset

    This repository accesses the medical dataset for cross-silo FL reseaches through [FLamby](https://github.com/owkin/FLamby), but does not contain the data itself. **If users want to use the data, please carefully read yourself with the license stated in FLamby.**

    We tested at release version 0.0.1 in [FLamby](https://github.com/owkin/FLamby). If you use the medical dataset. You try, for example,
    ```bash
    $  SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True `pip install 'git+https://github.com/owkin/FLamby.git@0.0.1[all_extra]'`
    ```

    **Git-clone and pip-install locally might be a better way, because some dataset needs downloading manually and running preprocess script.**

    (Tips: 1. You should download some dataset in FLamby such as [Heart Disease](https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_heart_disease/README.md) before pip-install. (This is because some parts of their implementation rely on relative paths from module files to specify data directories.) 2. When you install FLamby with `pip` in some virtual environments and get error, changing [`sys.executable`](https://github.com/owkin/FLamby/blob/4dfc53479ec4141849d67a6adace1137819317a2/setup.py#L11) path in setup.py may work. )

### Others
- Regenerate GPRC snipet

    ```bash
    $ cd src
    $ python acsilo_grpc/code_gen.py
    ```

- FL Parameters
    - default parameters
        - see `src/default_params.yaml`
    - args
        - see `src/options.py`

- IP setting
    - see `src/ip_utils.py`

- Methods
    - `DEFAULT`
        - FedAVG without differential privacy
    - `ULDP-NAIVE`
        - Naive user-level DP based on DP-FedAVG
    - `ULDP-GROUP`
        - Group-privacy based method
    - `ULDP-SGD`
        - User-level DP Fed-SGD
    - `ULDP-AVG`
        - User-level DP Fed-AVG
    - (not our main focus) `SILO-LEVEL-DP`
        - The privacy model is the same as in client-level DP in cross-device FL such as in https://arxiv.org/abs/1710.06963
    - (not our main focus) `RECORD-LEVEL-DP`
        - The algorithm is simple DP-SGD (https://arxiv.org/abs/1607.00133) in each silo
        - The same privacy model is seen in e.g., https://arxiv.org/abs/2206.07902


- Test
    - python -m unittest -v src.tests.test_secure_aggregation


--- 

The source code is based on a simplified version of FedML (https://github.com/FedML-AI/FedML).
