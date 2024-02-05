import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import os
import options
import noise_utils
from run_simulation import run_simulation
import pickle
import ast
import time

# import pprint

src_path = os.path.dirname(os.path.abspath(__file__))
path_project = os.path.dirname(src_path)

img_path = os.path.join(path_project, "exp", "img")
pickle_path = os.path.join(path_project, "exp", "pickle")
results_path = os.path.join(path_project, "exp", "results")

Q_LIST_SIZE = 30


# for saving figure with printing the file path
def save_figure(file_name, fig=None):
    file_path = os.path.join(img_path, file_name + ".png")
    if fig is None:
        plt.savefig(
            file_path,
            dpi=150,
            bbox_inches="tight",
        )
    else:
        fig.savefig(
            file_path,
            dpi=150,
            bbox_inches="tight",
        )
    print("Result image saved to:", file_path)


# for updating or creating result file
def dump_results(file_name, results_dict):
    file_path = os.path.join(pickle_path, file_name)
    lock_file_path = f"{file_path}.lock"

    try:
        # wait lock
        while os.path.exists(lock_file_path):
            time.sleep(1)
            print("wait lock")

        # get lock
        with open(lock_file_path, "w") as lock_file:
            lock_file.write("locked")

        try:
            with open(file_path, "rb") as file:
                prev_results_dict = pickle.load(file)
            results_dict = results_dict | prev_results_dict
        except FileNotFoundError:
            pass

        with open(file_path, "wb") as file:
            pickle.dump(results_dict, file)

    finally:
        os.remove(lock_file_path)

    print("Result objects saved to: ", file_name)


def check_results_file_already_exist(file_name):
    file_path = os.path.join(pickle_path, file_name)
    if os.path.exists(file_path):
        raise FileExistsError(f"File '{file_path}' already exists.")


# for depicting qC curve
def make_q_c_curve(epsilon_u, delta, sigma, n_round=100, num_points=20, min=-5):
    T = n_round
    num_points = num_points // 3 * 2
    x = (
        np.logspace(min, -1, num_points).tolist()
        + np.linspace(0.15, 1.0, int(num_points / 2)).tolist()
    )
    y = []
    for q_u in x:
        sensitivity_u, eps, _ = noise_utils.from_q_u(
            q_u=q_u, delta=delta, epsilon_u=epsilon_u, sigma=sigma, T=T
        )
        assert eps <= epsilon_u, f"eps={eps} > epsilon_u={epsilon_u}"
        # print("sensitivity_u =", sensitivity_u, "eps =", eps)
        y.append(sensitivity_u)
    return x, y


# plotting qC curve
def plot_q_c_curve(x, y, title="", img_name="", log=True):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker="o", label="sensitivity_u")
    ax.set_xlabel(r"$q_u$", fontsize=20)
    ax.set_ylabel(r"$C_u$", fontsize=20)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_title(title, fontsize=24)
    save_figure("q_c_pair-example" + img_name, fig)
    plt.grid(True, linestyle="--")
    plt.show()


def make_static_params(
    epsilon_u_dct,
    delta,
    sigma,
    n_round,
    idx_per_group,
    q_step_size,
    static_q_u_list=None,
):
    C_u_dct = {}
    q_u_dct = {}

    if static_q_u_list is not None:
        q_u_list = static_q_u_list
    else:
        # exponential
        n_of_q_u = Q_LIST_SIZE
        q_u_list = []
        init_q_u = 1.0
        for _ in range(n_of_q_u):
            q_u_list.append(init_q_u)
            init_q_u *= q_step_size

    C_and_q_per_group = {}
    for group_eps, idx in idx_per_group.items():
        q_u = q_u_list[idx]
        C_u, _eps, _ = noise_utils.from_q_u(
            q_u=q_u, delta=delta, epsilon_u=group_eps, sigma=sigma, T=n_round
        )
        assert _eps <= group_eps, f"_eps={_eps} > eps_u={group_eps}"
        C_and_q_per_group[group_eps] = (C_u, q_u)

    for user_id, eps_u in epsilon_u_dct.items():
        C_u, q_u = C_and_q_per_group[eps_u]
        C_u_dct[user_id] = C_u
        q_u_dct[user_id] = q_u

    return C_u_dct, q_u_dct


# wrapper of run_simulation without input argument
def fed_simulation(
    delta,
    sigma,
    n_users,
    C_u=None,
    q_u=None,
    q_step_size=None,
    momentum_weight=0.9,
    times=1,
    user_dist="uniform-iid",
    silo_dist="uniform",
    dataset_name="light_mnist",
    clipping_bound=1.0,
    n_round=10,
    global_learning_rate=10.0,
    local_learning_rate=0.01,
    local_epochs=50,
    agg_strategy="PULDP-AVG",
    epsilon_u=None,
    group_thresholds=None,
    validation_ratio=0.0,
    with_momentum=True,
    off_train_loss_noise=False,
    step_decay=True,
    hp_baseline=None,
    initial_q_u=None,
    gpu_id=None,
    parallelized=False,
    seed=0,
):
    args = options.build_default_args(path_project)

    if dataset_name == "heart_disease":
        from flamby_utils.heart_disease import update_args

        args = update_args(args)

    elif dataset_name == "tcga_brca":
        from flamby_utils.tcga_brca import update_args

        args = update_args(args)

    args.dataset_name = dataset_name
    args.agg_strategy = agg_strategy
    args.n_total_round = n_round
    args.n_users = n_users
    args.local_epochs = local_epochs
    args.times = times

    args.user_dist = user_dist
    args.silo_dist = silo_dist
    args.global_learning_rate = global_learning_rate
    args.local_learning_rate = local_learning_rate
    args.clipping_bound = clipping_bound
    args.with_momentum = with_momentum
    args.momentum_weight = momentum_weight
    args.off_train_loss_noise = off_train_loss_noise
    args.step_decay = step_decay
    args.hp_baseline = hp_baseline

    args.delta = delta
    args.sigma = sigma
    args.C_u = C_u
    args.q_u = q_u
    args.q_step_size = q_step_size
    args.epsilon_u = epsilon_u
    args.group_thresholds = group_thresholds
    args.dry_run = False
    args.secure_w = False
    args.initial_q_u = initial_q_u

    args.validation_ratio = validation_ratio
    args.client_optimizer = "adam"
    args.gpu_id = gpu_id
    args.parallelized = parallelized

    args.seed = seed

    results_list = []
    for i in range(args.times):
        print("======== TIME:", i, "start")
        args.seed = args.seed + i
        sim_results = run_simulation(args, path_project)
        results_list.append(sim_results)
    return results_list


def calc_metric(results, symbol="test"):
    if symbol == "train":
        acc_list = np.array([r["train"]["train_metric"] for r in results])
        loss_list = np.array([r["train"]["train_loss"] for r in results])
    else:
        acc_list = np.array([r["global"][f"global_{symbol}"][-1][1] for r in results])
        loss_list = np.array([r["global"][f"global_{symbol}"][-1][2] for r in results])
    acc_mean, acc_std, loss_mean, loss_std = (
        np.mean(acc_list),
        np.std(acc_list),
        np.mean(loss_list),
        np.std(loss_list),
    )
    return acc_mean, acc_std, loss_mean, loss_std


def make_epsilon_u(
    epsilon=1.0,
    n_users=0,
    dist="homo",
    epsilon_list=[],
    ratio_list=[],
    random_state: np.random.RandomState = None,
):
    if dist == "homo":
        epsilon_u = {user_id: epsilon for user_id in range(n_users)}
    elif dist == "hetero":
        assert len(epsilon_list) > 0 and len(ratio_list) > 0
        epsilon_u_list = random_state.choice(epsilon_list, size=n_users, p=ratio_list)
        epsilon_u = {user_id: epsilon_u_list[user_id] for user_id in range(n_users)}
    else:
        raise ValueError(f"invalid dist {dist}")
    return epsilon_u


def group_by_closest_below(epsilon_u_dct: dict, group_thresholds: list):
    minimum = min(epsilon_u_dct.values())
    group_thresholds = set(group_thresholds) | {minimum}
    grouped = {
        g: [] for g in group_thresholds
    }  # Initialize the dictionary with empty lists for each group threshold
    for key, value in epsilon_u_dct.items():
        # Find the closest group threshold that is less than or equal to the value
        closest_group = max([g for g in group_thresholds if g <= value], default=None)
        # If a suitable group is found, append the key to the corresponding list
        if closest_group is not None:
            grouped[closest_group].append(key)

    return grouped


# STATIC MANUAL OPTIMIZATION
def prepare_grid_search(epsilon_u, start_idx: int, end_idx: int):
    # set idx list for each group
    group_eps_set = set(epsilon_u.values())
    idx_per_group_list = []

    idx_list = list(range(Q_LIST_SIZE))[start_idx:end_idx]
    idx_list_list = [idx_list for _ in range(len(group_eps_set))]

    for combination in itertools.product(*idx_list_list):
        idx_per_group = {}
        for idx, group_eps in zip(combination, group_eps_set):
            idx_per_group[group_eps] = idx
        idx_per_group_list.append(idx_per_group)

    return {"name": "grid", "params": {"idx_per_group_list": idx_per_group_list}}


def prepare_random_search(
    epsilon_u,
    start_idx: int,
    end_idx: int,
    random_state: np.random.RandomState,
    n_samples: int,
):
    # set idx list for each group``
    group_eps_set = set(epsilon_u.values())
    idx_per_group_list = []

    idx_list = list(range(Q_LIST_SIZE))[start_idx:end_idx]
    idx_list_list = [idx_list for _ in range(len(group_eps_set))]
    all_candidates = itertools.product(*idx_list_list)
    samples = random_state.choice(len(all_candidates), size=n_samples, replace=False)

    for sample in samples:
        idx_per_group = {}
        for idx, group_eps in zip(all_candidates[sample], group_eps_set):
            idx_per_group[group_eps] = idx
        idx_per_group_list.append(idx_per_group)

    return {"name": "random", "params": {"idx_per_group_list": idx_per_group_list}}


def prepare_independent_search(epsilon_u, start_idx: int, end_idx: int):
    # set idx list for each group
    group_eps_set = set(epsilon_u.values())
    idx_per_group_list = []
    idx_list = list(range(Q_LIST_SIZE))[start_idx:end_idx]

    for group_eps in group_eps_set:
        for idx in idx_list:
            idx_per_group = {}
            idx_per_group[group_eps] = idx
            for group_eps in group_eps_set:
                if group_eps != group_eps:
                    idx_per_group[group_eps] = int((start_idx + end_idx) / 2)
        idx_per_group_list.append(idx_per_group)

    return {"name": "independent", "params": {"idx_per_group_list": idx_per_group_list}}


def static_optimization_result_file_name(
    sigma,
    delta,
    n_users,
    n_round,
    dataset_name,
    q_step_size,
    opt_strategy,
    validation_ratio,
    prefix_epsilon_u,
    static_q_u_list,
    user_dist,
    global_learning_rate=None,
    local_learning_rate=None,
    local_epochs=None,
):
    if global_learning_rate is not None:
        return f'static_optimization_{sigma}_{delta}_{n_users}_{n_round}_{dataset_name}_{q_step_size}_{opt_strategy["name"]}_{validation_ratio}_{prefix_epsilon_u}_{static_q_u_list}_{user_dist}_{global_learning_rate}_{local_learning_rate}_{local_epochs}.pkl'
    else:
        return f'static_optimization_{sigma}_{delta}_{n_users}_{n_round}_{dataset_name}_{q_step_size}_{opt_strategy["name"]}_{validation_ratio}_{prefix_epsilon_u}_{static_q_u_list}_{user_dist}.pkl'


# Do Static Optimization which could be the Best Utility Baseline
# Basically, we use grid search with prepare_grid_search() method above
def static_optimization(
    epsilon_u,
    sigma,
    delta,
    n_users,
    n_round,
    dataset_name,
    times,
    q_step_size,
    opt_strategy: dict,
    global_learning_rate=10.0,
    local_learning_rate=0.01,
    local_epochs=50,
    validation_ratio=0.0,
    user_dist="uniform-iid",
    silo_dist="uniform",
    static_q_u_list=None,
    gpu_id=None,
    force_update=False,
    parallelized=False,
):
    results_dict = {}

    prefix_epsilon_u = list(epsilon_u.items())[:4]
    results_file_name = static_optimization_result_file_name(
        sigma,
        delta,
        n_users,
        n_round,
        dataset_name,
        q_step_size,
        opt_strategy,
        validation_ratio,
        prefix_epsilon_u,
        static_q_u_list,
        user_dist,
        global_learning_rate,
        local_learning_rate,
        local_epochs,
    )

    if not force_update:
        check_results_file_already_exist(results_file_name)

    if opt_strategy["name"] in ["grid", "random", "independent"]:
        # grid search
        for idx_per_group in opt_strategy["params"]["idx_per_group_list"]:
            print("IDX: ", idx_per_group)
            C_u, q_u = make_static_params(
                epsilon_u,
                delta,
                sigma,
                n_round,
                idx_per_group=idx_per_group,
                q_step_size=q_step_size,
                static_q_u_list=static_q_u_list,
            )
            result = fed_simulation(
                delta,
                sigma,
                n_users,
                C_u=C_u,
                q_u=q_u,
                agg_strategy="PULDP-AVG",
                times=times,
                n_round=n_round,
                user_dist=user_dist,
                silo_dist=silo_dist,
                global_learning_rate=global_learning_rate,
                local_learning_rate=local_learning_rate,
                dataset_name=dataset_name,
                local_epochs=local_epochs,
                epsilon_u=epsilon_u,
                validation_ratio=validation_ratio,
                gpu_id=gpu_id,
                parallelized=parallelized,
            )
            results_dict[str(idx_per_group)] = (q_u, C_u, result)
    else:
        raise ValueError(f"invalid opt_strategy {opt_strategy}")

    dump_results(results_file_name, results_dict)


def show_static_optimization_result(
    epsilon_u,
    sigma,
    delta,
    n_users,
    n_round,
    dataset_name,
    q_step_size,
    opt_strategy: dict,
    n_silos,
    validation_ratio=0.0,
    train_loss=False,
    errorbar=True,
    img_name="",
    is_3d=False,
    static_q_u_list=None,
    user_dist=None,
    global_learning_rate=None,
    local_learning_rate=None,
    local_epochs=None,
):
    prefix_epsilon_u = list(epsilon_u.items())[:4]
    results_file_name = static_optimization_result_file_name(
        sigma,
        delta,
        n_users,
        n_round,
        dataset_name,
        q_step_size,
        opt_strategy,
        validation_ratio,
        prefix_epsilon_u,
        static_q_u_list,
        user_dist,
        global_learning_rate,
        local_learning_rate,
        local_epochs,
    )
    with open(os.path.join(pickle_path, results_file_name), "rb") as file:
        results_dict = pickle.load(file)

    x = list(results_dict.keys())
    q_u_list = [results_dict[i][0][0] for i in x]
    C_u_list = [results_dict[i][1][0] for i in x]
    acc_mean_acc_std_loss_mean_loss_std = [
        calc_metric(results_dict[i][2], "test") for i in x
    ]
    y = [acc_mean_acc_std_loss_mean_loss_std[i][2] for i in range(len(x))]  # loss_mean
    error = [
        acc_mean_acc_std_loss_mean_loss_std[i][3] for i in range(len(x))
    ]  # loss_std
    y_metric = [
        acc_mean_acc_std_loss_mean_loss_std[i][0] for i in range(len(x))
    ]  # acc_mean
    error_metric = [
        acc_mean_acc_std_loss_mean_loss_std[i][1] for i in range(len(x))
    ]  # acc_std

    if not is_3d:
        if len(x) > 12:
            plt.figure(figsize=(12, 5))
        else:
            plt.figure(figsize=(9, 5))
        plt.title(
            r"{}: $\epsilon_u={}$, $\sigma={}$, $|U|={}$, $|S|={}$, $T={}$".format(
                dataset_name, prefix_epsilon_u[0][1], sigma, n_users, n_silos, n_round
            ),
            fontsize=20,
        )
        plt.ylabel("Test Loss", fontsize=20)
        if errorbar:
            plt.errorbar(x, y, yerr=error, fmt="-o", label="Test Loss")
        else:
            plt.plot(x, y, "-o", label="Test Loss")
        for i in range(len(x)):
            plt.text(
                x[i],
                y[i] * 1.02,
                f"q={q_u_list[i]:.3f}\nC={C_u_list[i]:.3f}",
                fontsize=8,
            )
        plt.legend(loc="upper left", fontsize=16)
        plt.yscale("log")

        ax2 = plt.twinx()
        if errorbar:
            ax2.errorbar(
                x, y_metric, yerr=error_metric, fmt="-o", color="red", label="Accuracy"
            )
        else:
            ax2.plot(x, y_metric, "-o", color="red", label="Accuracy")
        ax2.set_ylabel("Accuracy", fontsize=20)
        ax2.legend(loc="upper right", fontsize=18)
        plt.xticks([])

        save_figure("static_optimization_result-1d-" + img_name)
        plt.show()

    if len(x) == 1:
        _, ax_loss = plt.subplots()
        result = results_dict[x[0]][2]
        loss_means = []
        loss_stds = []
        acc_means = []
        acc_stds = []

        symbol = "test" if validation_ratio <= 0.0 else "valid"
        label = ""

        for i in range(len(result[0]["global"][f"global_{symbol}"])):
            losses_at_position = [
                result[round_id]["global"][f"global_{symbol}"][i][2]
                for round_id in range(len(result))
            ]
            accs_at_position = [
                result[round_id]["global"][f"global_{symbol}"][i][1]
                for round_id in range(len(result))
            ]

            loss_means.append(np.mean(losses_at_position))
            loss_stds.append(np.std(losses_at_position))

            acc_means.append(np.mean(accs_at_position))
            acc_stds.append(np.std(accs_at_position))

        _x = range(len(loss_means))
        if errorbar:
            ax_loss.errorbar(
                _x, loss_means, yerr=loss_stds, label="Loss", alpha=0.8, color="blue"
            )
        else:
            ax_loss.plot(_x, loss_means, label="Loss", alpha=0.8, color="blue")

        ax_loss.set_ylabel("Test Loss", fontsize=20)
        ax_loss.set_xlabel("Round", fontsize=20)
        ax_loss.set_yscale("log")
        ax_loss.legend(loc="upper right", fontsize=20)

        ax2 = plt.twinx()
        if errorbar:
            ax2.errorbar(
                _x, acc_means, yerr=acc_stds, label="Accuracy", alpha=0.8, color="red"
            )
        else:
            ax2.plot(_x, acc_means, label="Accuracy", alpha=0.8, color="red")
        ax2.legend(loc="upper right", fontsize=20)

        save_figure("static_optimization_result-1d-loss-analysis-" + img_name)
        plt.show()

    if is_3d:

        def transform_string(input_str):
            i = int(input_str.strip("[]").split(",")[0])
            value = round(q_step_size**i, 2)
            return f"{value}, {value}"

        param_set_list = [
            list(ast.literal_eval(key).values()) for key in results_dict.keys()
        ]
        title_eps = [0.15, 3.0, 5.0]
        num_x = 7
        x_values = np.arange(0, num_x)
        x_labels = [round(q_step_size**i, 2) for i in x_values]

        if static_q_u_list is not None:
            num_x = len(static_q_u_list)
            x_labels = static_q_u_list

            def transform_string(input_str):  # noqa: F811
                idx_list = input_str.strip("[]").split(",")
                assert len(idx_list) == 2
                values = [static_q_u_list[int(idx)] for idx in idx_list]
                return f"{values[0]}, {values[1]}"

        for i in range(3):
            data = {"x": [], "y": [], "label": []}
            for param_set, acc in zip(param_set_list, y_metric):
                data["x"].append(param_set[i])
                data["y"].append(acc)
                data["label"].append(str(param_set[:i] + param_set[i + 1 :]))

            max_points_count = {}
            label_with_max_point = []
            for label in set(data["label"]):
                label_data = [
                    d for d in zip(data["x"], data["y"], data["label"]) if d[2] == label
                ]
                max_point = max(label_data, key=lambda item: item[1])
                label_with_max_point.append((label, max_point))

            plt.figure(figsize=(6, 4))
            label_with_max_point_top_20 = sorted(
                label_with_max_point, key=lambda item: item[1][1], reverse=True
            )[:20]
            for label, max_point in label_with_max_point_top_20:
                plt.scatter(max_point[0], max_point[1], color="red", s=60)
                max_points_count[max_point[0]] = (
                    max_points_count.get(max_point[0], 0) + 1
                )
                # plt.text(max_point[0], max_point[1], label, color='black', ha='right', va='bottom')

            filtered_x = [
                d
                for d, l in zip(data["x"], data["label"])
                if l in {label for label, _ in label_with_max_point_top_20}
            ]
            filtered_y = [
                d
                for d, l in zip(data["y"], data["label"])
                if l in {label for label, _ in label_with_max_point_top_20}
            ]
            filtered_label = [
                d
                for d, l in zip(data["label"], data["label"])
                if l in {label for label, _ in label_with_max_point_top_20}
            ]
            sns.lineplot(
                data={"x": filtered_x, "y": filtered_y, "label": filtered_label},
                x="x",
                y="y",
                hue="label",
            )

            other_epses = [e for e in title_eps if e != title_eps[i]]
            legend = plt.legend(
                title=r"$q_u$ for groups with $\epsilon={}$ and $\epsilon={}$".format(
                    other_epses[0], other_epses[1]
                ),
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),
                shadow=True,
                ncol=num_x,
                fontsize=8,
            )
            legend_texts = legend.get_texts()
            for text in legend_texts:
                original_text = text.get_text()
                new_text = transform_string(original_text)
                text.set_text(new_text)

            # show max points for each x
            for x_val, count in max_points_count.items():
                plt.text(
                    x_val,
                    max(data["y"]),
                    f"{count}",
                    color="blue",
                    ha="center",
                    va="bottom",
                    fontsize=24,
                )

            plt.title(
                r"{}: Various HP ($q_u$) on the group with $\epsilon={}$"
                "\n"
                r"($\sigma={}$, $|U|={}$, $|S|={}$, $T={}$)".format(
                    dataset_name, title_eps[i], sigma, n_users, n_silos, n_round
                ),
                fontsize=18,
                y=1.1,
            )
            plt.ylabel("Test Accuracy", fontsize=20)
            plt.xlabel(
                r"$q_u$ for the group with $\epsilon={}$".format(title_eps[i]),
                fontsize=18,
            )
            plt.xticks(ticks=x_values, labels=x_labels)

            save_figure(f"static_optimization_result-top20-group{i}-" + img_name)
            plt.show()

        # Sort params_set_list by size of corresponding y_metric
        # sorted_param_set_list = [
        #     (param_set, y)
        #     for y, param_set in sorted(
        #         zip(y_metric, param_set_list), key=lambda item: item[0], reverse=True
        #     )
        # ]
        # pprint.pprint(sorted_param_set_list)

    min_idx, min_test_loss = x[np.argmin(y)], np.min(y)

    if train_loss:
        plt.figure(figsize=(8, 3))
        x = list(results_dict.keys())
        q_u_list = [results_dict[i][0][0] for i in x]
        C_u_list = [results_dict[i][1][0] for i in x]
        acc_mean_acc_std_loss_mean_loss_std = [
            calc_metric(results_dict[i][2], "train") for i in x
        ]
        y = [
            acc_mean_acc_std_loss_mean_loss_std[i][2] for i in range(len(x))
        ]  # loss_mean
        error = [
            acc_mean_acc_std_loss_mean_loss_std[i][3] for i in range(len(x))
        ]  # loss_std
        if errorbar:
            plt.errorbar(x, y, yerr=error, fmt="-o")
        else:
            plt.plot(x, y, "-o")
        for i in range(len(x)):
            plt.text(
                x[i],
                y[i] * 1.02,
                f"q={q_u_list[i]:.3f}\nC={C_u_list[i]:.3f}",
                fontsize=8,
            )
        plt.title("Train Loss Mean with Standard Deviation over different idx")
        plt.xlabel("idx")
        plt.ylabel("Train Loss Mean")
        plt.legend(loc="upper right")

        ax2 = plt.twinx()
        y_metric = [
            acc_mean_acc_std_loss_mean_loss_std[i][0] for i in range(len(x))
        ]  # acc_mean
        error_metric = [
            acc_mean_acc_std_loss_mean_loss_std[i][1] for i in range(len(x))
        ]  # acc_std
        if errorbar:
            ax2.errorbar(
                x, y_metric, yerr=error_metric, fmt="-o", color="red", label="Accuracy"
            )
        else:
            ax2.plot(x, y_metric, "-o", color="red", label="Accuracy")

        ax2.legend(loc="lower left")
        plt.show()

    if validation_ratio > 0.0:
        plt.figure(figsize=(8, 3))
        x = list(results_dict.keys())
        q_u_list = [results_dict[i][0][0] for i in x]
        C_u_list = [results_dict[i][1][0] for i in x]
        acc_mean_acc_std_loss_mean_loss_std = [
            calc_metric(results_dict[i][2], "valid") for i in x
        ]
        y = [
            acc_mean_acc_std_loss_mean_loss_std[i][2] for i in range(len(x))
        ]  # loss_mean
        error = [
            acc_mean_acc_std_loss_mean_loss_std[i][3] for i in range(len(x))
        ]  # loss_std
        if errorbar:
            plt.errorbar(x, y, yerr=error, fmt="-o")
        else:
            plt.plot(x, y, "-o")
        for i in range(len(x)):
            plt.text(
                x[i],
                y[i] * 1.02,
                f"q={q_u_list[i]:.3f}\nC={C_u_list[i]:.3f}",
                fontsize=8,
            )
        plt.title("Valid Loss Mean with Standard Deviation over different idx")
        plt.xlabel("idx")
        plt.ylabel("Valid Loss Mean")
        plt.legend(loc="upper right")

        ax2 = plt.twinx()
        y_metric = [
            acc_mean_acc_std_loss_mean_loss_std[i][0] for i in range(len(x))
        ]  # acc_mean
        error_metric = [
            acc_mean_acc_std_loss_mean_loss_std[i][1] for i in range(len(x))
        ]  # acc_std
        if errorbar:
            ax2.errorbar(
                x, y_metric, yerr=error_metric, fmt="-o", color="red", label="Accuracy"
            )
        else:
            ax2.plot(x, y_metric, "-o", color="red", label="Accuracy")
        ax2.legend(loc="lower left")

        plt.show()

    return min_idx, min_test_loss


def specified_idx_result_file_name(
    n_users,
    sigma,
    delta,
    dataset_name,
    n_round,
    idx_per_group,
    q_step_size,
    validation_ratio,
    prefix_epsilon_u,
    static_q_u_list,
    global_learning_rate=None,
    local_learning_rate=None,
    local_epochs=None,
):
    if global_learning_rate is not None:
        return f"specified_idx_{n_users}_{sigma}_{delta}_{dataset_name}_{n_round}_{idx_per_group}_{q_step_size}_{validation_ratio}_{prefix_epsilon_u}_{static_q_u_list}_{global_learning_rate}_{local_learning_rate}_{local_epochs}.pkl".replace(
            ":", "_"
        )
    else:
        return f"specified_idx_{n_users}_{sigma}_{delta}_{dataset_name}_{n_round}_{idx_per_group}_{q_step_size}_{validation_ratio}_{prefix_epsilon_u}_{static_q_u_list}.pkl".replace(
            ":", "_"
        )


def run_with_specified_idx(
    epsilon_u,
    sigma,
    delta,
    n_users,
    n_round,
    dataset_name,
    q_step_size,
    times,
    idx_per_group,
    global_learning_rate,
    local_learning_rate,
    local_epochs,
    validation_ratio=0.0,
    user_dist="uniform-iid",
    silo_dist="uniform",
    static_q_u_list=None,
    gpu_id=None,
    force_update=False,
    parallelized=False,
    seed=0,
):
    prefix_epsilon_u = list(epsilon_u.items())[:4]
    results_file_name = specified_idx_result_file_name(
        n_users,
        sigma,
        delta,
        dataset_name,
        n_round,
        idx_per_group,
        q_step_size,
        validation_ratio,
        prefix_epsilon_u,
        static_q_u_list,
        global_learning_rate,
        local_learning_rate,
        local_epochs,
    )

    if not force_update:
        check_results_file_already_exist(results_file_name)

    C_u, q_u = make_static_params(
        epsilon_u,
        delta,
        sigma,
        n_round,
        idx_per_group=idx_per_group,
        q_step_size=q_step_size,
        static_q_u_list=static_q_u_list,
    )
    result = fed_simulation(
        delta,
        sigma,
        n_users,
        C_u=C_u,
        q_u=q_u,
        agg_strategy="PULDP-AVG",
        times=times,
        n_round=n_round,
        user_dist=user_dist,
        silo_dist=silo_dist,
        global_learning_rate=global_learning_rate,
        local_learning_rate=local_learning_rate,
        dataset_name=dataset_name,
        local_epochs=local_epochs,
        epsilon_u=epsilon_u,
        validation_ratio=validation_ratio,
        gpu_id=gpu_id,
        parallelized=parallelized,
        seed=seed,
    )

    acc_mean, acc_std, loss_mean, loss_std = calc_metric(result, "test")
    print(
        f"TEST ACC: {acc_mean:.4f} ± {acc_std:.4f}",
        f", TEST LOSS: {loss_mean:.4f} ± {loss_std:.4f}",
    )

    if validation_ratio > 0.0:
        acc_mean, acc_std, loss_mean, loss_std = calc_metric(result, "valid")
        print(
            f"VALID ACC: {acc_mean:.4f} ± {acc_std:.4f}",
            f", VALID LOSS: {loss_mean:.4f} ± {loss_std:.4f}",
        )

    with open(os.path.join(pickle_path, results_file_name), "wb") as file:
        pickle.dump(result, file)


def show_specified_idx_result(
    prefix_epsilon_u_list,
    sigma,
    delta,
    n_users,
    n_round,
    dataset_name,
    q_step_size,
    idx_per_group_list,
    label_list="",
    validation_ratio=0.0,
    errorbar=True,
    img_name="",
    static_q_u_list=None,
    global_learning_rate=None,
    local_learning_rate=None,
    local_epochs=None,
):
    # optimal q_u
    _, ax_loss = plt.subplots()
    if type(idx_per_group_list) is not list:
        prefix_epsilon_u_list = [prefix_epsilon_u_list]
        idx_per_group_list = [idx_per_group_list]
        label_list = [label_list]
    for idx_per_group, prefix_epsilon_u, label in zip(
        idx_per_group_list, prefix_epsilon_u_list, label_list
    ):
        results_file_name = specified_idx_result_file_name(
            n_users,
            sigma,
            delta,
            dataset_name,
            n_round,
            idx_per_group,
            q_step_size,
            validation_ratio,
            prefix_epsilon_u,
            static_q_u_list,
            global_learning_rate,
            local_learning_rate,
            local_epochs,
        )
        with open(os.path.join(pickle_path, results_file_name), "rb") as file:
            result = pickle.load(file)

        loss_means = []
        loss_stds = []
        acc_means = []
        acc_stds = []

        symbol = "test" if validation_ratio <= 0.0 else "valid"

        for i in range(len(result[0]["global"][f"global_{symbol}"])):
            losses_at_position = [
                result[round_id]["global"][f"global_{symbol}"][i][2]
                for round_id in range(len(result))
            ]
            accs_at_position = [
                result[round_id]["global"][f"global_{symbol}"][i][1]
                for round_id in range(len(result))
            ]

            loss_means.append(np.mean(losses_at_position))
            loss_stds.append(np.std(losses_at_position))

            acc_means.append(np.mean(accs_at_position))
            acc_stds.append(np.std(accs_at_position))

        x = range(len(loss_means))
        if errorbar:
            ax_loss.errorbar(x, loss_means, yerr=loss_stds, label=f"{label}", alpha=0.8)
        else:
            ax_loss.plot(x, loss_means, label=f"{label}", alpha=0.8)

    ax_loss.set_ylabel("Test Loss", fontsize=20)
    ax_loss.set_xlabel("Round", fontsize=20)
    ax_loss.set_yscale("log")
    ax_loss.legend(loc="upper right", fontsize=20)

    save_figure("specified_idx_result-" + img_name)
    plt.show()
    return x, acc_means, acc_stds


def online_optimization_result_file_name(
    agg_strategy,
    n_users,
    sigma,
    delta,
    dataset_name,
    n_round,
    q_step_size,
    validation_ratio,
    prefix_epsilon_u,
    with_momentum,
    off_train_loss_noise,
    step_decay,
    hp_baseline,
    initial_q_u,
    momentum_weight,
    global_learning_rate=None,
    local_learning_rate=None,
    local_epochs=None,
):
    if global_learning_rate is not None:
        return f"online_optimization_{agg_strategy}_{n_users}_{sigma}_{delta}_{dataset_name}_{n_round}_{q_step_size}_{validation_ratio}_{prefix_epsilon_u}_{with_momentum}_{off_train_loss_noise}_{step_decay}_{hp_baseline}_{initial_q_u}_{momentum_weight}_{global_learning_rate}_{local_learning_rate}_{local_epochs}.pkl"
    else:
        return f"online_optimization_{agg_strategy}_{n_users}_{sigma}_{delta}_{dataset_name}_{n_round}_{q_step_size}_{validation_ratio}_{prefix_epsilon_u}_{with_momentum}_{off_train_loss_noise}_{step_decay}_{hp_baseline}_{initial_q_u}_{momentum_weight}.pkl"


# ONLINE OPTIMIZATION
def run_online_optimization(
    epsilon_u,
    sigma,
    delta,
    n_users,
    n_round,
    dataset_name,
    q_step_size,
    group_thresholds,
    times,
    agg_strategy,
    global_learning_rate=10.0,
    local_learning_rate=0.01,
    local_epochs=50,
    validation_ratio=0.0,
    momentum_weight=0.9,
    initial_q_u=None,
    with_momentum=None,
    step_decay=True,
    hp_baseline=None,
    off_train_loss_noise=None,
    user_dist="uniform-iid",
    silo_dist="uniform",
    gpu_id=None,
    force_update=False,
    parallelized=False,
):
    prefix_epsilon_u = list(epsilon_u.items())[:4]
    results_file_name = online_optimization_result_file_name(
        agg_strategy,
        n_users,
        sigma,
        delta,
        dataset_name,
        n_round,
        q_step_size,
        validation_ratio,
        prefix_epsilon_u,
        with_momentum,
        off_train_loss_noise,
        step_decay,
        hp_baseline,
        initial_q_u,
        momentum_weight,
        global_learning_rate,
        local_learning_rate,
        local_epochs,
    )

    if not force_update:
        check_results_file_already_exist(results_file_name)

    result = fed_simulation(
        delta,
        sigma,
        n_users,
        C_u=None,
        q_u=None,
        q_step_size=q_step_size,
        agg_strategy=agg_strategy,
        times=times,
        n_round=n_round,
        user_dist=user_dist,
        silo_dist=silo_dist,
        momentum_weight=momentum_weight,
        initial_q_u=initial_q_u,
        global_learning_rate=global_learning_rate,
        local_learning_rate=local_learning_rate,
        dataset_name=dataset_name,
        local_epochs=local_epochs,
        epsilon_u=epsilon_u,
        group_thresholds=group_thresholds,
        validation_ratio=validation_ratio,
        with_momentum=with_momentum,
        step_decay=step_decay,
        off_train_loss_noise=off_train_loss_noise,
        hp_baseline=hp_baseline,
        gpu_id=gpu_id,
        parallelized=parallelized,
    )

    acc_mean, acc_std, loss_mean, loss_std = calc_metric(result)
    print(
        f"TEST ACC: {acc_mean:.4f} ± {acc_std:.4f}",
        f", TEST LOSS: {loss_mean:.4f} ± {loss_std:.4f}",
    )

    if validation_ratio > 0.0:
        acc_mean, acc_std, loss_mean, loss_std = calc_metric(result, "valid")
        print(
            f"VALID ACC: {acc_mean:.4f} ± {acc_std:.4f}",
            f", VALID LOSS: {loss_mean:.4f} ± {loss_std:.4f}",
        )

    with open(os.path.join(pickle_path, results_file_name), "wb") as file:
        pickle.dump(result, file)


def get_eps_u_color_mapping(eps_u_values):
    eps_u_colors = ["b", "g", "m", "c", "y", "k", "r"]
    eps_u_color_mapping = {
        eps_u: eps_u_colors[i % len(eps_u_colors)]
        for i, eps_u in enumerate(eps_u_values)
    }
    return eps_u_color_mapping


def show_online_optimization_result(
    epsilon_u,
    sigma,
    delta,
    n_users,
    n_round,
    dataset_name,
    q_step_size,
    agg_strategy,
    validation_ratio=0.0,
    with_momentum=None,
    hp_baseline=None,
    off_train_loss_noise=None,
    step_decay=True,
    errorbar=True,
    img_name="",
    initial_q_u=None,
    momentum_weight=0.9,
    is_show_accuracy=False,
    global_learning_rate=None,
    local_learning_rate=None,
    local_epochs=None,
):
    prefix_epsilon_u = list(epsilon_u.items())[:4]
    results_file_name = online_optimization_result_file_name(
        agg_strategy,
        n_users,
        sigma,
        delta,
        dataset_name,
        n_round,
        q_step_size,
        validation_ratio,
        prefix_epsilon_u,
        with_momentum,
        off_train_loss_noise,
        step_decay,
        hp_baseline,
        initial_q_u,
        momentum_weight,
        global_learning_rate,
        local_learning_rate,
        local_epochs,
    )
    with open(os.path.join(pickle_path, results_file_name), "rb") as file:
        result = pickle.load(file)

    # eps_uの値のリストを取得（すべての辞書から共通のキーを抽出）
    eps_u_values = set(key for dct in result for key in dct["param_history"].keys())
    eps_u_color_mapping = get_eps_u_color_mapping(eps_u_values)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    for eps_u in eps_u_values:
        # 各辞書からeps_uに対応するデータを集める
        all_data = np.array(
            [
                dct["param_history"][eps_u][:-1]
                for dct in result
                if eps_u in dct["param_history"]
            ]
        )

        # 平均値と標準偏差を計算
        means = np.mean(all_data, axis=0)
        stds = np.std(all_data, axis=0)

        # 平均値と標準偏差をプロット
        x = range(len(means))
        y1 = [item[0] for item in means]
        # y2 = [item[1] for item in means]
        error1 = [item[0] for item in stds]
        # error2 = [item[1] for item in stds]

        color = eps_u_color_mapping[eps_u]

        if errorbar:
            ax1.errorbar(
                x,
                y1,
                yerr=error1,
                label=r"$\epsilon_u={}$".format(eps_u),
                alpha=1.0,
                color=color,
            )
            # ax1.errorbar(x, y2, yerr=error2, label=r"$C_u$ \epsilon_u={}$".format(eps_u), alpha=0.5, color=color)
        else:
            ax1.plot(
                x, y1, label=r"$\epsilon_u={}$".format(eps_u), alpha=1.0, color=color
            )
            # ax1.plot(x, y2, label=r"$C_u$ \epsilon_u={}$".format(eps_u), alpha=0.5, color=color)

    loss_means = []
    loss_stds = []
    acc_means = []
    acc_stds = []

    symbol = "test" if validation_ratio <= 0.0 else "valid"

    # 各ラウンドに対して処理
    for i in range(len(result[0]["global"][f"global_{symbol}"])):
        # その位置における全ラウンドのloss値を集める
        losses_at_position = [
            result[round_id]["global"][f"global_{symbol}"][i][2]
            for round_id in range(len(result))
        ]
        accs_at_position = [
            result[round_id]["global"][f"global_{symbol}"][i][1]
            for round_id in range(len(result))
        ]

        loss_means.append(np.mean(losses_at_position))
        loss_stds.append(np.std(losses_at_position))

        acc_means.append(np.mean(accs_at_position))
        acc_stds.append(np.std(accs_at_position))

    ax2 = ax1.twinx()
    if errorbar:
        ax2.errorbar(
            x, loss_means, yerr=loss_stds, label="Test Loss", color="red", alpha=1.0
        )
    else:
        ax2.plot(x, loss_means, label="Test Loss", color="red", alpha=1.0)
    # ax1.set_yscale('log')
    ax1.set_xlabel("Round", fontsize=20)
    ax1.set_ylabel(r"HP ($q_u$)", fontsize=22)
    ax1.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Test Loss", fontsize=20)
    # ax2.set_yscale('log')
    ax1.legend(loc="lower left", fontsize=14)
    ax2.legend(loc="lower right", fontsize=14)

    plt.grid(True, linestyle="--")
    plt.title(r"HP ($q_u$) with Test Loss", fontsize=20)
    save_figure("online_optimization_result-" + img_name)
    plt.show()

    _, ax_train_loss = plt.subplots(figsize=(9, 6))
    y = []
    for eps_u in eps_u_values:
        # 各辞書からeps_uに対応するデータを集める
        all_data = np.array(
            [
                dct["loss_history"][eps_u][:-1]
                for dct in result
                if eps_u in dct["loss_history"]
            ]
        )
        means = np.mean(all_data, axis=0)
        y.extend([item[0] for item in means])

    up = np.min([np.max(y), 10]) * 1.05
    bottom = -up * 1.05

    for eps_u in eps_u_values:
        all_data = np.array(
            [
                dct["loss_history"][eps_u][:-1]
                for dct in result
                if eps_u in dct["loss_history"]
            ]
        )

        means = np.mean(all_data, axis=0)
        stds = np.std(all_data, axis=0)

        train_loss_round = range(len(means))
        y = [item[0] for item in means]
        error = [item[0] for item in stds]
        y_clamped = np.clip(y, bottom, up)

        color = eps_u_color_mapping[eps_u]
        if errorbar:
            ax_train_loss.errorbar(
                train_loss_round,
                y_clamped,
                yerr=error,
                label=r"$\epsilon_u={}$".format(eps_u),
                alpha=0.9,
                color=color,
            )
        else:
            ax_train_loss.plot(
                train_loss_round,
                y_clamped,
                label=r"$\epsilon_u={}$".format(eps_u),
                alpha=0.9,
                color=color,
                linewidth=2,
            )

        y = [item[1] for item in means]
        error = [item[1] for item in stds]

        if errorbar:
            ax_train_loss.errorbar(
                train_loss_round,
                y,
                yerr=error,
                label=r"(w/o momentum)\epsilon_u={}$".format(eps_u),
                alpha=0.4,
                color=color,
            )
        else:
            ax_train_loss.plot(
                train_loss_round,
                y,
                label=r"(w/o momentum) $\epsilon_u={}$".format(eps_u),
                alpha=0.4,
                color=color,
            )

    ax_train_loss.set_ylim(bottom, up * 1.01)
    ax_train_loss.set_xlabel("Round")
    ax_train_loss.set_ylabel("(surrogate function) \n Metric", fontsize=20)
    ax_train_loss.legend(
        loc="upper left", bbox_to_anchor=(1.02, 0.8), fontsize=16, borderaxespad=0.0
    )
    plt.subplots_adjust(right=0.75)
    plt.grid(True, linestyle="--")

    plt.title("Approximated Gradient for FDM", fontsize=20)
    save_figure("online_optimization_result-losses-" + img_name)
    plt.show()

    if is_show_accuracy:
        fig, ax = plt.subplots(figsize=(6, 4))
        if errorbar:
            ax.errorbar(
                x,
                acc_means,
                yerr=acc_stds,
                label="Test Accuracy",
                color="red",
                alpha=0.5,
            )
        else:
            ax.plot(x, acc_means, label="Test Accuracy", color="red", alpha=0.5)
        ax.set_xlabel("Round", fontsize=20)
        ax.set_ylabel("Test Accuracy", fontsize=20)
        ax.legend(loc="upper left", fontsize=12)

        save_figure("online_optimization_result-testacc-" + img_name)
        plt.show()

    return x, acc_means, acc_stds


def plot_acc_results(
    all_acc_results,
    dataset_name,
    initial_q_u_list,
    delta,
    alpha,
    N,
    T,
    sigma,
    errorbar=True,
):
    for initial_q_u in initial_q_u_list:
        plt.figure(figsize=(7, 5))
        for (agg_strategy, param), (x, acc_means, acc_stds) in all_acc_results.items():
            if type(param) is float and param == initial_q_u:
                if errorbar:
                    plt.errorbar(
                        x, acc_means, yerr=acc_stds, label=f"{agg_strategy}", marker="o"
                    )
                else:
                    plt.plot(x, acc_means, label=f"{agg_strategy}", marker="o")
            elif (
                agg_strategy == "PULDP-AVG" and type(param) is str
            ) or agg_strategy == "random":
                if errorbar:
                    plt.errorbar(
                        x, acc_means, yerr=acc_stds, label=f"{param}", marker="o"
                    )
                else:
                    plt.plot(x, acc_means, label=f"{param}", marker="o")

        # グラフの設定
        plt.xlabel("Round", fontsize=20)
        plt.ylabel("Test Accuracy", fontsize=20)
        plt.title(
            r"{}: Comparison with Baselines"
            "\n"
            r"$\epsilon_u=(0.15, 3.0, 5.0)$, $\delta={}$, $\alpha={}$, $N={}$, $T={}$, $\sigma={}$".format(
                dataset_name, delta, alpha, N, T, sigma
            ),
            fontsize=14,
        )

        # 目盛りを点線で表示
        plt.grid(True, linestyle="--")

        plt.legend(fontsize=14)
        save_figure(f"comparison-with-baselines-{dataset_name}-{initial_q_u}")
        plt.show()
