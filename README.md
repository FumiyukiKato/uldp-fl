# Acsilo

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
    $ python run_server.py --n_silos=3 --n_silo_per_round=2 --n_total_round=5
    ```

    Wake up 3 silos
    ```bash
    $ cd src
    $ python run_silo.py --n_silos=3 --n_silo_per_round=2 --silo_id=0 --n_total_round=1
    $ python run_silo.py --n_silos=3 --n_silo_per_round=2 --silo_id=1 --n_total_round=1
    $ python run_silo.py --n_silos=3 --n_silo_per_round=2 --silo_id=2 --n_total_round=1
    ```

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

--- 

The source code is based on a simplified version of FedML (https://github.com/FedML-AI/FedML).
