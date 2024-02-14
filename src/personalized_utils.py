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
import hashlib

# import pprint

src_path = os.path.dirname(os.path.abspath(__file__))
path_project = os.path.dirname(src_path)

img_path = os.path.join(path_project, "exp", "img")
pickle_path = os.path.join(path_project, "exp", "pickle")
results_path = os.path.join(path_project, "exp", "results")


def init_heart_disease_param(**kwargs):
    fed_sim_params = FLSimulationParameters(
        dataset_name="heart_disease",
        agg_strategy="PULDP-AVG",
        n_total_round=30,
        n_users=400,
        n_silos=4,
        times=5,
        delta=1e-5,
        sigma=1.0,
        user_dist="uniform-iid",
        silo_dist="uniform",
        local_epochs=30,
        global_learning_rate=10.0,
        local_learning_rate=0.001,
        validation_ratio=0.0,
        q_step_size=0.8,
        dynamic_global_learning_rate=False,
        with_momentum=True,
        step_decay=True,
    )

    for key, value in kwargs.items():
        if key == "eps_u":
            fed_sim_params.epsilon_list = [kwargs["eps_u"]]
            fed_sim_params.group_thresholds = [kwargs["eps_u"]]
            fed_sim_params.ratio_list = [1.0]
        else:
            setattr(fed_sim_params, key, value)

    return fed_sim_params


def init_mnist_param(**kwargs):
    fed_sim_params = FLSimulationParameters(
        dataset_name="mnist",
        agg_strategy="PULDP-AVG",
        n_total_round=50,
        n_users=1000,
        n_silos=5,
        times=5,
        delta=1e-5,
        sigma=1.0,
        user_dist="uniform-iid",
        silo_dist="uniform",
        local_epochs=50,
        global_learning_rate=5.0,
        local_learning_rate=0.001,
        validation_ratio=0.0,
        q_step_size=0.8,
        dynamic_global_learning_rate=False,
    )

    for key, value in kwargs.items():
        if key == "eps_u":
            fed_sim_params.epsilon_list = [kwargs["eps_u"]]
            fed_sim_params.group_thresholds = [kwargs["eps_u"]]
            fed_sim_params.ratio_list = [1.0]
        else:
            setattr(fed_sim_params, key, value)

    return fed_sim_params


class FLSimulationParameters:
    def __init__(
        self,
        dataset_name,
        agg_strategy,
        n_total_round,
        n_users,
        n_silos,
        times,
        delta,
        sigma,
        user_dist,
        silo_dist,
        local_epochs,
        global_learning_rate,
        local_learning_rate,
        epsilon_u=None,
        initial_q_u=1.0,
        q_step_size=None,
        C_u=None,
        q_u=None,
        step_decay=True,
        with_momentum=True,
        hp_baseline=None,
        momentum_weight=0.9,
        off_train_loss_noise=False,
        dynamic_global_learning_rate=False,
        validation_ratio=0.0,
        gpu_id=None,
        parallelized=False,
        seed=0,
        epsilon_list=None,
        ratio_list=None,
        group_thresholds=None,
        version=None,
    ):
        self.seed = seed
        self.dataset_name = dataset_name
        self.agg_strategy = agg_strategy
        self.n_total_round = n_total_round
        self.n_users = n_users
        self.n_silos = n_silos
        self.local_epochs = local_epochs
        self.times = times

        self.user_dist = user_dist
        self.silo_dist = silo_dist
        self.global_learning_rate = global_learning_rate
        self.local_learning_rate = local_learning_rate
        self.with_momentum = with_momentum
        self.momentum_weight = momentum_weight
        self.off_train_loss_noise = off_train_loss_noise
        self.step_decay = step_decay
        self.hp_baseline = hp_baseline

        self.delta = delta
        self.sigma = sigma
        self.C_u = C_u
        self.q_u = q_u
        self.q_step_size = q_step_size
        self.epsilon_u = epsilon_u
        self.initial_q_u = initial_q_u
        self.dynamic_global_learning_rate = dynamic_global_learning_rate

        self.validation_ratio = validation_ratio
        self.gpu_id = gpu_id
        self.parallelized = parallelized

        self.client_optimizer = "adam"
        self.dry_run = False
        self.secure_w = False

        self.epsilon_list = epsilon_list
        self.ratio_list = ratio_list
        self.group_thresholds = group_thresholds

        self.version = version

    def create_file_prefix(self, prefix="", middle="", suffix=""):
        exclude_keys = [
            "times",
            "seed",
            "C_u",
            "q_u",
            "epsilon_u",
            "group_thresholds",
            "parallelized",
            "gpu_id",
            "dry_run",
            "secure_w",
            "idx_per_group",
            "version",
        ]

        # 辞書から除外キーを除いたもので文字列を構築
        parts = [
            f"{key}-{self.__dict__[key]}"
            for key in sorted(self.__dict__.keys())
            if key not in exclude_keys
        ]

        if self.epsilon_u is not None:
            epsilon_u_part = f"{list(self.epsilon_u.items())[:4]}"
            parts.append(epsilon_u_part)

        if self.version is not None:
            parts.append(f"version-{self.version}")

        key_strings = middle + "x".join(parts)

        # 全体の文字列のMD5ハッシュを計算
        hash_object = hashlib.md5(key_strings.encode())
        hash_hex = hash_object.hexdigest()

        # ハッシュ値をプレフィックスとして使用
        return prefix + hash_hex + suffix

    def get_group_eps_set(self):
        if self.epsilon_list is not None:
            return set(self.epsilon_list)
        else:
            ValueError("epsilon_list is None")


def get_eps_u_color_mapping(eps_u_values):
    eps_u_colors = ["b", "g", "m", "c", "y", "k", "r"]
    eps_u_color_mapping = {
        eps_u: eps_u_colors[i % len(eps_u_colors)]
        for i, eps_u in enumerate(eps_u_values)
    }
    return eps_u_color_mapping


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
        return True
    return False


# for depicting qC curve
def make_q_c_curve(epsilon_u, delta, sigma, n_total_round=100, num_points=20, min=-5):
    T = n_total_round
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
def plot_q_c_curve(x, y, title="", img_name="", log=True, is_qC=False):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker="o", label=r"$C_u$")
    if is_qC:
        ax2 = ax.twinx()
        ax2.plot(
            x,
            np.array(x) * np.array(y),
            marker="x",
            linestyle="--",
            color="red",
            label=r"$q_u \times C_u$",
        )
        ax2.set_yscale("log")
        ax2.set_ylabel(r"$q_u \times C_u$", fontsize=20)
        ax2.legend(loc="upper right", fontsize=16)
    ax.set_xlabel(r"$q_u$", fontsize=20)
    ax.set_ylabel(r"$C_u$", fontsize=20)
    ax.legend(loc="upper left", fontsize=16)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_title(title, fontsize=24)
    plt.grid(True, linestyle="--")
    if is_qC:
        save_figure("q_c_pair-example-qc" + img_name, fig)
    else:
        save_figure("q_c_pair-example" + img_name, fig)
    plt.show()


# wrapper of run_simulation without input argument
def fed_simulation(
    fed_sim_params: FLSimulationParameters, idx_per_group=None, static_q_u_list=None
):
    args = options.build_default_args(path_project)

    if fed_sim_params.dataset_name == "heart_disease":
        from flamby_utils.heart_disease import update_args

        args = update_args(args)

    elif fed_sim_params.dataset_name == "tcga_brca":
        from flamby_utils.tcga_brca import update_args

        args = update_args(args)

    args.dataset_name = fed_sim_params.dataset_name
    args.agg_strategy = fed_sim_params.agg_strategy
    args.n_total_round = fed_sim_params.n_total_round
    args.n_users = fed_sim_params.n_users
    args.local_epochs = fed_sim_params.local_epochs
    args.times = fed_sim_params.times

    args.user_dist = fed_sim_params.user_dist
    args.silo_dist = fed_sim_params.silo_dist
    args.global_learning_rate = fed_sim_params.global_learning_rate
    args.local_learning_rate = fed_sim_params.local_learning_rate
    args.with_momentum = fed_sim_params.with_momentum
    args.momentum_weight = fed_sim_params.momentum_weight
    args.off_train_loss_noise = fed_sim_params.off_train_loss_noise
    args.step_decay = fed_sim_params.step_decay
    args.hp_baseline = fed_sim_params.hp_baseline

    args.delta = fed_sim_params.delta
    args.sigma = fed_sim_params.sigma
    args.C_u = fed_sim_params.C_u
    args.q_u = fed_sim_params.q_u
    args.q_step_size = fed_sim_params.q_step_size
    args.epsilon_u = fed_sim_params.epsilon_u
    args.group_thresholds = fed_sim_params.group_thresholds
    args.dry_run = fed_sim_params.dry_run
    args.secure_w = fed_sim_params.secure_w
    args.initial_q_u = fed_sim_params.initial_q_u
    args.dynamic_global_learning_rate = fed_sim_params.dynamic_global_learning_rate

    args.validation_ratio = fed_sim_params.validation_ratio
    args.client_optimizer = fed_sim_params.client_optimizer
    args.gpu_id = fed_sim_params.gpu_id
    args.parallelized = fed_sim_params.parallelized

    args.seed = fed_sim_params.seed

    results_list = []
    for i in range(args.times):
        print("======== TIME:", i, "start")
        args.seed = args.seed + i
        sim_results = run_simulation(
            args,
            path_project,
            epsilon_list=fed_sim_params.epsilon_list,
            ratio_list=fed_sim_params.ratio_list,
            q_step_size=fed_sim_params.q_step_size,
            idx_per_group=idx_per_group,
            static_q_u_list=static_q_u_list,
        )
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
def prepare_grid_search(group_eps_set, start_idx: int, end_idx: int):
    # set idx list for each group
    idx_per_group_list = []

    assert end_idx <= 30, "end_idx should be less than 30"
    idx_list = list(range(30))[start_idx:end_idx]
    idx_list_list = [idx_list for _ in range(len(group_eps_set))]

    for combination in itertools.product(*idx_list_list):
        idx_per_group = {}
        for idx, group_eps in zip(combination, group_eps_set):
            idx_per_group[group_eps] = idx
        idx_per_group_list.append(idx_per_group)

    return idx_per_group_list


# Do Static Optimization which could be the Best Utility Baseline
# Basically, we use grid search with prepare_grid_search() method above
def static_optimization(
    fed_sim_params: FLSimulationParameters,
    idx_per_group_list,  # HP candidates
    static_q_u_list=None,
    force_update=False,
):
    results_dict = {}

    results_file_name = fed_sim_params.create_file_prefix(
        prefix="static_optimization_",
        middle=f"{idx_per_group_list[0]}-{len(idx_per_group_list)}_"
        + f"{static_q_u_list}_",
        suffix=".pkl",
    )
    print(results_file_name)

    if not force_update and check_results_file_already_exist(results_file_name):
        print("Skip: File already exists.")
        return

    for idx_per_group in idx_per_group_list:
        print("IDX: ", idx_per_group)
        result = fed_simulation(
            fed_sim_params, idx_per_group=idx_per_group, static_q_u_list=static_q_u_list
        )
        q_u, C_u = result[0]["q_u"], result[0]["C_u"]

        results_dict[str(idx_per_group)] = (q_u, C_u, result)

    dump_results(results_file_name, results_dict)


def show_static_optimization_result(
    fed_sim_params: FLSimulationParameters,
    idx_per_group_list,
    static_q_u_list=None,
    train_loss=False,
    errorbar=False,
    img_name="",
    is_3d=False,
    outlier=None,
):
    results_file_name = fed_sim_params.create_file_prefix(
        prefix="static_optimization_",
        middle=f"{idx_per_group_list[0]}-{len(idx_per_group_list)}_"
        + f"{static_q_u_list}_",
        suffix=".pkl",
    )
    with open(os.path.join(pickle_path, results_file_name), "rb") as file:
        results_dict = pickle.load(file)

    x = list(results_dict.keys())
    if outlier is not None:
        x = x[:outlier]
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
            plt.figure(figsize=(16, 5))
        else:
            plt.figure(figsize=(10, 5))
        plt.title(
            r"{}: $\epsilon_u={}$, $\sigma={}$, $|U|={}$, $|S|={}$, $T={}$".format(
                fed_sim_params.dataset_name,
                fed_sim_params.get_group_eps_set().pop(),
                fed_sim_params.sigma,
                fed_sim_params.n_users,
                fed_sim_params.n_silos,
                fed_sim_params.n_total_round,
            ),
            fontsize=20,
        )
        plt.ylabel("Test Loss", fontsize=20)
        if errorbar:
            plt.errorbar(x, y, yerr=error, fmt="-o", label="Test Loss")
        else:
            plt.plot(x, y, "-o", label="Test Loss")
        plt.legend(loc="upper left", fontsize=16)
        # lossの最小値を持つ点を大きい丸で強調
        min_y_index = y.index(min(y))  # yの最大値のインデックスを取得
        plt.scatter(
            x[min_y_index],
            y[min_y_index],
            color="blue",
            s=200,
        )  # sはマーカーのサイズ

        ax2 = plt.twinx()
        if errorbar:
            ax2.errorbar(
                x, y_metric, yerr=error_metric, fmt="-o", color="red", label="Accuracy"
            )
        else:
            ax2.plot(x, y_metric, "-o", color="red", label="Accuracy")
        ax2.set_ylabel("Accuracy", fontsize=20)
        ax2.legend(loc="upper right", fontsize=18)

        x_ticks_labels = [
            f"q={q_u_list[i]:.2f}\nC={C_u_list[i]:.2f}" for i in range(len(x))
        ]
        plt.xticks(range(len(x)), x_ticks_labels)

        plt.grid(True, linestyle="--")
        save_figure("static_optimization_result-1d-" + img_name)
        plt.show()

    if len(x) == 1:
        _, ax_loss = plt.subplots()
        result = results_dict[x[0]][2]
        loss_means = []
        loss_stds = []
        acc_means = []
        acc_stds = []

        symbol = "test" if fed_sim_params.validation_ratio <= 0.0 else "valid"
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

        plt.grid(True, linestyle="--")
        save_figure("static_optimization_result-1d-loss-analysis-" + img_name)
        plt.show()

    if is_3d:

        def transform_string(input_str):
            i = int(input_str.strip("[]").split(",")[0])
            value = round(fed_sim_params.q_step_size**i, 2)
            return f"{value}, {value}"

        param_set_list = [
            list(ast.literal_eval(key).values()) for key in results_dict.keys()
        ]
        title_eps = [0.15, 3.0, 5.0]
        num_x = 7
        x_values = np.arange(0, num_x)
        x_labels = [round(fed_sim_params.q_step_size**i, 2) for i in x_values]

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
            TOP_N = 20
            label_with_max_point_top_20 = sorted(
                label_with_max_point, key=lambda item: item[1][1], reverse=True
            )[:TOP_N]
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
                    fed_sim_params.dataset_name,
                    title_eps[i],
                    fed_sim_params.sigma,
                    fed_sim_params.n_users,
                    fed_sim_params.n_silos,
                    fed_sim_params.n_total_round,
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

            plt.grid(True, linestyle="--")
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

    max_idx, max_metric = x[np.argmax(y_metric)], np.max(y_metric)
    print("Max Metric:", max_metric, "at", max_idx)

    min_idx, min_metric = x[np.argmin(y)], np.min(y)

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
        plt.grid(True, linestyle="--")
        plt.show()

    if fed_sim_params.validation_ratio > 0.0:
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

        plt.grid(True, linestyle="--")
        plt.show()

    return min_idx, min_metric


def run_with_specified_idx(
    fed_sim_params: FLSimulationParameters,
    idx_per_group,
    static_q_u_list=None,
    force_update=False,
):
    results_file_name = fed_sim_params.create_file_prefix(
        prefix="static_optimization_",
        middle=f"{idx_per_group}_" + f"{static_q_u_list}_",
        suffix=".pkl",
    )

    if not force_update and check_results_file_already_exist(results_file_name):
        print("Skip: File already exists.")
        return

    print("IDX: ", idx_per_group)
    result = fed_simulation(
        fed_sim_params, idx_per_group=idx_per_group, static_q_u_list=static_q_u_list
    )

    acc_mean, acc_std, loss_mean, loss_std = calc_metric(result, "test")
    print(
        f"TEST ACC: {acc_mean:.4f} ± {acc_std:.4f}",
        f", TEST LOSS: {loss_mean:.4f} ± {loss_std:.4f}",
    )

    if fed_sim_params.validation_ratio > 0.0:
        acc_mean, acc_std, loss_mean, loss_std = calc_metric(result, "valid")
        print(
            f"VALID ACC: {acc_mean:.4f} ± {acc_std:.4f}",
            f", VALID LOSS: {loss_mean:.4f} ± {loss_std:.4f}",
        )

    with open(os.path.join(pickle_path, results_file_name), "wb") as file:
        pickle.dump(result, file)


def show_specified_idx_result(
    fed_sim_params: FLSimulationParameters,
    idx_per_group,
    static_q_u_list=None,
    label="",
    errorbar=True,
    img_name="",
):
    # optimal q_u
    _, ax_loss = plt.subplots()
    results_file_name = fed_sim_params.create_file_prefix(
        prefix="static_optimization_",
        middle=f"{idx_per_group}_" + f"{static_q_u_list}_",
        suffix=".pkl",
    )
    with open(os.path.join(pickle_path, results_file_name), "rb") as file:
        result = pickle.load(file)

    loss_means = []
    loss_stds = []
    acc_means = []
    acc_stds = []

    symbol = "test" if fed_sim_params.validation_ratio <= 0.0 else "valid"

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
        ax_loss.errorbar(
            x, loss_means, yerr=loss_stds, label=f"{label}", alpha=0.8, color="red"
        )
    else:
        ax_loss.plot(x, loss_means, label=f"{label}", alpha=0.8, color="red")

    last_x = x[-1]
    last_loss_mean = loss_means[-1]
    last_loss_std = loss_stds[-1]
    last_acc_mean = acc_means[-1]
    last_acc_std = acc_stds[-1]
    ax_loss.text(
        last_x,
        last_loss_mean,
        f"{last_loss_mean:.3f}±{last_loss_std:.3f}\n (Acc: {last_acc_mean:.3f}±{last_acc_std:.3f})",
        ha="center",
        va="bottom",
        fontsize=14,
    )

    ax_loss.set_ylabel("Test Loss", fontsize=20)
    ax_loss.set_xlabel("Round", fontsize=20)
    ax_loss.set_yscale("log")
    ax_loss.legend(loc="upper right", fontsize=20)

    plt.grid(True, linestyle="--")
    save_figure("specified_idx_result-" + img_name)
    plt.show()
    return x, acc_means, acc_stds


# ONLINE OPTIMIZATION
def run_online_optimization(fed_sim_params: FLSimulationParameters, force_update=False):
    results_file_name = fed_sim_params.create_file_prefix(
        prefix="online_optimization_", suffix=".pkl"
    )

    if not force_update and check_results_file_already_exist(results_file_name):
        print("Skip: File already exists.")
        return

    result = fed_simulation(fed_sim_params)

    acc_mean, acc_std, loss_mean, loss_std = calc_metric(result)
    print(
        f"TEST ACC: {acc_mean:.4f} ± {acc_std:.4f}",
        f", TEST LOSS: {loss_mean:.4f} ± {loss_std:.4f}",
    )

    if fed_sim_params.validation_ratio > 0.0:
        acc_mean, acc_std, loss_mean, loss_std = calc_metric(result, "valid")
        print(
            f"VALID ACC: {acc_mean:.4f} ± {acc_std:.4f}",
            f", VALID LOSS: {loss_mean:.4f} ± {loss_std:.4f}",
        )

    with open(os.path.join(pickle_path, results_file_name), "wb") as file:
        pickle.dump(result, file)


def show_online_optimization_result(
    fed_sim_params: FLSimulationParameters,
    errorbar=True,
    img_name="",
    is_show_accuracy=False,
):
    results_file_name = fed_sim_params.create_file_prefix(
        prefix="online_optimization_", suffix=".pkl"
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
                dct["param_history"][eps_u]
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

    symbol = "test" if fed_sim_params.validation_ratio <= 0.0 else "valid"

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
    if len(x) == len(loss_means) + 1:
        x = x[:-1]
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

    if fed_sim_params.hp_baseline not in ["random", "random-log"]:
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

        up = np.min([np.max(y), 100]) * 1.05
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

        plt.grid(True, linestyle="--")
        save_figure("online_optimization_result-testacc-" + img_name)
        plt.show()

    return x, acc_means, acc_stds


def plot_acc_results(
    fed_sim_params: FLSimulationParameters,
    all_acc_results,
    initial_q_u_list,
    errorbar=True,
):
    N = fed_sim_params.n_users
    T = fed_sim_params.n_total_round
    sigma = fed_sim_params.sigma
    delta = fed_sim_params.delta
    dataset_name = fed_sim_params.dataset_name
    alpha = fed_sim_params.q_step_size

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
                (agg_strategy == "PULDP-AVG" and type(param) is str)
                or agg_strategy == "random"
                or agg_strategy == "random-log"
            ):
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
        save_figure(f"comparison-with-baselines-{dataset_name}-{initial_q_u}-")
        plt.show()
