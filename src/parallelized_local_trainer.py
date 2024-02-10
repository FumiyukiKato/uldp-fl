from collections import OrderedDict
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
from multiprocessing import Queue
from dataset import CREDITCARD, HEART_DISEASE, TCGA_BRCA
from method_group import (
    METHOD_DEFAULT,
    METHOD_GROUP_ULDP_AVG,
    METHOD_GROUP_ULDP_SGD,
    METHOD_GROUP_ONLINE_OPTIMIZATION,
    METHOD_GROUP_ULDP_GROUPS,
    METHOD_GROUP_WITHIN_SILO_DP_ACCOUNTING,
    METHOD_PULDP_AVG,
    METHOD_PULDP_QC_TEST,
    METHOD_PULDP_QC_TRAIN,
    METHOD_RECORD_LEVEL_DP,
    METHOD_SILO_LEVEL_DP,
    METHOD_ULDP_NAIVE,
)

from mylogger import logger
import noise_utils
import copy


def parallelized_train_worker(input_queue: Queue, output_queue: Queue):
    while True:
        task = input_queue.get()
        if task is None:
            break
        (
            silo_id,
            gpu_id,
            random_state,
            model,
            device,
            agg_strategy,
            train_loader,
            criterion,
            round_idx,
            n_silo_per_round,
            privacy_engine,
            clipping_bound,
            sigma,
            delta,
            user_level_data_loader,
            user_weights,
            dataset_name,
            client_optimizer,
            local_learning_rate,
            weight_decay,
            local_epochs,
            C_u,
            epsilon_groups,
            C_u_list,
            stepped_C_u_list_for_optimization,
            user_weights_for_optimization,
            stepped_user_weights_for_optimization,
            C_u_list_for_optimization,
            q_u_list,
            stepped_q_u_list,
            off_train_loss_noise,
            accountant_dct,
            n_total_round,
            hp_baseline,
        ) = task
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
        model.to(device)
        result = parallelized_train(
            random_state=random_state,
            model=model,
            device=device,
            agg_strategy=agg_strategy,
            train_loader=train_loader,
            criterion=criterion,
            round_idx=round_idx,
            n_silo_per_round=n_silo_per_round,
            privacy_engine=privacy_engine,
            local_clipping_bound=clipping_bound,
            local_sigma=sigma,
            local_delta=delta,
            user_level_data_loader=user_level_data_loader,
            user_weights=user_weights,
            dataset_name=dataset_name,
            client_optimizer=client_optimizer,
            local_learning_rate=local_learning_rate,
            weight_decay=weight_decay,
            local_epochs=local_epochs,
            C_u=C_u,
            epsilon_groups=epsilon_groups,
            C_u_list=C_u_list,
            stepped_C_u_list_for_optimization=stepped_C_u_list_for_optimization,
            user_weights_for_optimization=user_weights_for_optimization,
            stepped_user_weights_for_optimization=stepped_user_weights_for_optimization,
            C_u_list_for_optimization=C_u_list_for_optimization,
            q_u_list=q_u_list,
            stepped_q_u_list=stepped_q_u_list,
            off_train_loss_noise=off_train_loss_noise,
            accountant_dct=accountant_dct,
            n_total_round=n_total_round,
            hp_baseline=hp_baseline,
        )
        if gpu_id is not None:
            torch.cuda.empty_cache()
        output_queue.put((silo_id, random_state, accountant_dct, result))


def parallelized_train(
    random_state: np.random.RandomState,
    model: nn.Module,
    device: torch.device,
    agg_strategy: str,
    train_loader: DataLoader,
    criterion,
    round_idx,
    n_silo_per_round=None,
    privacy_engine=None,
    local_clipping_bound=None,
    local_sigma=None,
    local_delta=None,
    user_level_data_loader=None,
    user_weights=None,
    dataset_name=None,
    client_optimizer="sgd",
    local_learning_rate=None,
    weight_decay=None,
    local_epochs=None,
    C_u=None,
    epsilon_groups=None,
    C_u_list=None,
    stepped_C_u_list_for_optimization=None,
    user_weights_for_optimization=None,
    stepped_user_weights_for_optimization=None,
    C_u_list_for_optimization=None,
    q_u_list=None,
    stepped_q_u_list=None,
    off_train_loss_noise=None,
    accountant_dct=None,
    n_total_round=None,
    hp_baseline=None,
):
    tick = time.time()
    model.train()
    global_weights = copy.deepcopy(model.state_dict())

    torch.manual_seed(random_state.randint(9223372036854775807))

    def loss_callback(loss):
        if torch.isnan(loss):
            logger.warn("loss is nan: skipping")
            return True
        return False

    if dataset_name == HEART_DISEASE:
        from flamby_utils.heart_disease import (
            custom_optimizer,
        )

        optimizer = custom_optimizer(
            model,
            local_learning_rate,
            client_optimizer,
            weight_decay,
        )
    elif dataset_name == TCGA_BRCA:
        from flamby_utils.tcga_brca import (
            custom_optimizer,
        )

        optimizer = custom_optimizer(
            model,
            local_learning_rate,
            client_optimizer,
            weight_decay,
        )
    else:
        if client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=local_learning_rate,
            )
        elif client_optimizer == "adam":
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=local_learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError("Unknown client optimizer")

    # Optimization step like CALCULATE GRADIENTS
    if agg_strategy in METHOD_GROUP_WITHIN_SILO_DP_ACCOUNTING:
        noise_generator = torch.Generator(device=device).manual_seed(
            random_state.randint(9223372036854775807)
        )
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            max_grad_norm=local_clipping_bound,
            noise_multiplier=local_sigma,
            noise_generator=noise_generator,
        )

    if agg_strategy in METHOD_GROUP_ULDP_SGD:
        grads_list = []  # TODO: memory optimization (use online aggregation)
        for user_id, user_train_loader in user_level_data_loader:
            if (
                user_weights[user_id] <= 0.0
            ):  # for efficiency, if w is encrypted for DDP, it can't work
                continue
            user_avg_grad = OrderedDict()
            for name, param in model.named_parameters():
                user_avg_grad[name] = torch.zeros_like(param.data)

            for x, labels in user_train_loader:
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                if dataset_name == CREDITCARD:
                    labels = labels.long()
                loss = criterion(log_probs, labels)
                loss_callback(loss)
                loss.backward()
                # Don't optimize (i.e., Don't call step())

                for name, param in model.named_parameters():
                    # Due to different batch size for each user
                    user_avg_grad[name] += param.grad / len(x)

            clipped_grads = noise_utils.global_clip(
                model, user_avg_grad, local_clipping_bound
            )
            weighted_clipped_grads = noise_utils.multiple_weights(
                model, clipped_grads, user_weights[user_id]
            )
            grads_list.append(weighted_clipped_grads)

        # calculate the average gradient
        if len(grads_list) <= 0:
            avg_grads = OrderedDict()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    avg_grads[name] = torch.zeros_like(param.data)
        else:
            avg_grads = noise_utils.torch_aggregation(grads_list)
        noisy_avg_grads = noise_utils.add_global_noise(
            model,
            avg_grads,
            random_state,
            local_sigma * local_clipping_bound / np.sqrt(n_silo_per_round),
            device=device,
        )

    elif agg_strategy in METHOD_GROUP_ULDP_AVG:
        # If suddenly becomes unstable, skip the update gradient (call step()) for now.

        weights_diff_list = []  # TODO: memory optimization (use online aggregation)
        for user_id, user_train_loader in user_level_data_loader:
            if (
                user_weights[user_id] <= 0.0
            ):  # for efficiency, if w is encrypted for DDP, it can't work
                continue
            model_u = copy.deepcopy(model)
            if client_optimizer == "sgd":
                optimizer_u = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, model_u.parameters()),
                    lr=local_learning_rate,
                )
            elif client_optimizer == "adam":
                optimizer_u = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model_u.parameters()),
                    lr=local_learning_rate,
                    weight_decay=weight_decay,
                )
            else:
                raise ValueError("Unknown client optimizer")
            for epoch in range(local_epochs):
                batch_loss = []
                for x, labels in user_train_loader:
                    x, labels = x.to(device), labels.to(device)
                    optimizer_u.zero_grad()
                    log_probs = model_u(x)
                    if dataset_name == CREDITCARD:
                        labels = labels.long()
                    loss = criterion(log_probs, labels)
                    if loss_callback(loss):
                        continue
                    loss.backward()
                    optimizer_u.step()
                    batch_loss.append(loss.item())

            weights = model_u.state_dict()
            if noise_utils.check_nan_inf(model_u):
                # If it includes Nan or Inf, then
                pass
            else:
                weights_diff = noise_utils.diff_weights(global_weights, weights)
                clipped_weights_diff = noise_utils.global_clip(
                    model_u, weights_diff, local_clipping_bound
                )
                weighted_clipped_weights_diff = noise_utils.multiple_weights(
                    model_u, clipped_weights_diff, user_weights[user_id]
                )
                weights_diff_list.append(weighted_clipped_weights_diff)

        if len(weights_diff_list) <= 0:
            avg_weights_diff = OrderedDict()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    avg_weights_diff[name] = torch.zeros_like(param.data)
        else:
            avg_weights_diff = noise_utils.torch_aggregation(weights_diff_list)
        noisy_avg_weights_diff = noise_utils.add_global_noise(
            model,
            avg_weights_diff,
            random_state,
            local_sigma * local_clipping_bound / np.sqrt(n_silo_per_round),
            device=device,
        )

    elif agg_strategy == METHOD_PULDP_AVG:
        weights_diff_list = []  # TODO: memory optimization (use online aggregation)
        for user_id, user_train_loader in user_level_data_loader:
            if (
                user_weights[user_id] <= 0.0
            ):  # for efficiency, if w is encrypted for DDP, it can't work
                continue
            model_u = copy.deepcopy(model)
            if client_optimizer == "sgd":
                optimizer_u = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, model_u.parameters()),
                    lr=local_learning_rate,
                )
            elif client_optimizer == "adam":
                optimizer_u = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model_u.parameters()),
                    lr=local_learning_rate,
                    weight_decay=weight_decay,
                )
            else:
                raise ValueError("Unknown client optimizer")
            for epoch in range(local_epochs):
                batch_loss = []
                for x, labels in user_train_loader:
                    x, labels = x.to(device), labels.to(device)
                    optimizer_u.zero_grad()
                    log_probs = model_u(x)
                    if dataset_name == CREDITCARD:
                        labels = labels.long()
                    loss = criterion(log_probs, labels)
                    if loss_callback(loss):
                        continue
                    loss.backward()
                    optimizer_u.step()
                    batch_loss.append(loss.item())

            weights = model_u.state_dict()
            weights_diff = noise_utils.diff_weights(global_weights, weights)
            clipped_weights_diff = noise_utils.global_clip(
                model_u, weights_diff, C_u[user_id]
            )
            weighted_clipped_weights_diff = noise_utils.multiple_weights(
                model_u, clipped_weights_diff, user_weights[user_id]
            )
            weights_diff_list.append(weighted_clipped_weights_diff)

        if len(weights_diff_list) <= 0:
            avg_weights_diff = OrderedDict()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    avg_weights_diff[name] = torch.zeros_like(param.data)
        else:
            avg_weights_diff = noise_utils.torch_aggregation(weights_diff_list)
        noisy_avg_weights_diff = noise_utils.add_global_noise(
            model,
            avg_weights_diff,
            random_state,
            local_sigma
            / np.sqrt(
                n_silo_per_round
            ),  # this local_sigma is the standard deviation of normal dist itself
            device=device,
        )

    elif agg_strategy in METHOD_GROUP_ONLINE_OPTIMIZATION:
        weights_diff_dct_per_epsilon_group = {}
        weights_diff_dct_per_epsilon_group_for_optimization = {}
        stepped_weights_diff_dct_per_epsilon_group_for_optimization = {}
        for eps_u, user_ids_per_eps_u in epsilon_groups.items():
            user_ids_per_eps_u_set = set(user_ids_per_eps_u)
            for ith, (user_weights, sensitivities, weights_diff_dct) in enumerate(
                [
                    (
                        user_weights,
                        C_u_list,
                        weights_diff_dct_per_epsilon_group,
                    ),
                    (
                        user_weights_for_optimization,
                        C_u_list_for_optimization,
                        weights_diff_dct_per_epsilon_group_for_optimization,
                    ),
                    (
                        stepped_user_weights_for_optimization,
                        stepped_C_u_list_for_optimization,
                        stepped_weights_diff_dct_per_epsilon_group_for_optimization,
                    ),
                ]
            ):
                if ith > 0 and hp_baseline is not None:
                    break

                weights_diff_list = []
                for user_id, user_train_loader in user_level_data_loader:
                    # to compute model delta per epsilon group
                    if user_id not in user_ids_per_eps_u_set:
                        continue
                    if (  # not sampled users
                        user_weights[user_id] <= 0.0
                    ):  # for efficiency, if w is encrypted for DDP, it can't work
                        continue
                    model_u = copy.deepcopy(model)
                    if client_optimizer == "sgd":
                        optimizer_u = torch.optim.SGD(
                            filter(lambda p: p.requires_grad, model_u.parameters()),
                            lr=local_learning_rate,
                        )
                    elif client_optimizer == "adam":
                        optimizer_u = torch.optim.Adam(
                            filter(lambda p: p.requires_grad, model_u.parameters()),
                            lr=local_learning_rate,
                            weight_decay=weight_decay,
                        )
                    else:
                        raise ValueError("Unknown client optimizer")
                    for epoch in range(local_epochs):
                        batch_loss = []
                        for x, labels in user_train_loader:
                            x, labels = x.to(device), labels.to(device)
                            optimizer_u.zero_grad()
                            log_probs = model_u(x)
                            if dataset_name == CREDITCARD:
                                labels = labels.long()
                            loss = criterion(log_probs, labels)
                            if loss_callback(loss):
                                continue
                            loss.backward()
                            optimizer_u.step()
                            batch_loss.append(loss.item())

                    weights = model_u.state_dict()
                    weights_diff = noise_utils.diff_weights(global_weights, weights)
                    clipped_weights_diff = noise_utils.global_clip(
                        model_u, weights_diff, sensitivities[user_id]
                    )
                    weighted_clipped_weights_diff = noise_utils.multiple_weights(
                        model_u, clipped_weights_diff, user_weights[user_id]
                    )
                    weights_diff_list.append(weighted_clipped_weights_diff)

                # average weights diff in the same epsilon group
                if len(weights_diff_list) <= 0:
                    avg_weights_diff = OrderedDict()
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            avg_weights_diff[name] = torch.zeros_like(param.data)
                else:
                    avg_weights_diff = noise_utils.torch_aggregation(weights_diff_list)
                weights_diff_dct[eps_u] = avg_weights_diff

        # compute delta(q_u, C_u)
        default_avg_weights_diff = noise_utils.torch_aggregation(
            list(weights_diff_dct_per_epsilon_group.values())
        )
        default_noisy_avg_weights_diff = noise_utils.add_global_noise(
            model,
            default_avg_weights_diff,
            random_state,
            local_sigma
            / np.sqrt(
                n_silo_per_round
            ),  # this local_sigma is the standard deviation of normal dist itself
            device=device,
        )
        DEFAULT_NAME = "default"
        noisy_avg_weights_diff_dct = {DEFAULT_NAME: default_noisy_avg_weights_diff}

        if hp_baseline:
            return noisy_avg_weights_diff_dct, {}

        # For finite difference method, compute delta(q_u_i, C_u_i)
        for eps_u in epsilon_groups.keys():
            # Need to compute delta(q_u_i, C_u_i) for each eps_i
            # They need to be noised for DP in distributed manner
            if agg_strategy == METHOD_PULDP_QC_TEST:
                original_avg_weights_diff = noise_utils.torch_aggregation(
                    [weights_diff_dct_per_epsilon_group_for_optimization[eps_u]]
                )
                noisy_original_avg_weights_diff = noise_utils.add_global_noise(
                    model,
                    original_avg_weights_diff,
                    random_state,
                    local_sigma
                    / np.sqrt(
                        n_silo_per_round
                    ),  # this local_sigma is the standard deviation of normal dist itself
                    device=device,
                )
                stepped_avg_weights_diff = noise_utils.torch_aggregation(
                    [stepped_weights_diff_dct_per_epsilon_group_for_optimization[eps_u]]
                )
                noisy_stepped_avg_weights_diff = noise_utils.add_global_noise(
                    model,
                    stepped_avg_weights_diff,
                    random_state,
                    local_sigma
                    / np.sqrt(
                        n_silo_per_round
                    ),  # this local_sigma is the standard deviation of normal dist itself
                    device=device,
                )

                noisy_avg_weights_diff_dct[eps_u] = (
                    noisy_original_avg_weights_diff,
                    noisy_stepped_avg_weights_diff,
                )

            elif agg_strategy == METHOD_PULDP_QC_TRAIN:
                # doesn't need noise
                original_avg_weights_diff = noise_utils.torch_aggregation(
                    [weights_diff_dct_per_epsilon_group_for_optimization[eps_u]]
                )
                stepped_avg_weights_diff = noise_utils.torch_aggregation(
                    [stepped_weights_diff_dct_per_epsilon_group_for_optimization[eps_u]]
                )

                noisy_avg_weights_diff_dct[eps_u] = (
                    original_avg_weights_diff,
                    stepped_avg_weights_diff,
                )
            else:
                raise NotImplementedError("Unknown aggregation strategy")

    else:
        for epoch in range(local_epochs):
            batch_loss = []
            for x, labels in train_loader:
                if len(x) == 0:  # this is possible in poisson sampling in DP-SGD
                    continue
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                log_probs = model(x)
                if dataset_name == CREDITCARD:
                    labels = labels.long()
                loss = criterion(log_probs, labels)
                loss_callback(loss)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

        weights = model.state_dict()
        weights_diff = noise_utils.diff_weights(global_weights, weights)

    train_time = time.time() - tick
    logger.debug("Train/Time : %s", train_time)

    # Post-process step
    if agg_strategy == METHOD_RECORD_LEVEL_DP:
        model.remove_hooks()
        return weights_diff, len(train_loader)
    elif agg_strategy == METHOD_DEFAULT:
        return weights_diff, len(train_loader)
    elif agg_strategy in METHOD_GROUP_ULDP_GROUPS:
        model.remove_hooks()
        return weights_diff, len(train_loader)
    elif agg_strategy == METHOD_ULDP_NAIVE:
        clipped_weights_diff = noise_utils.global_clip(
            model, weights_diff, local_clipping_bound
        )
        noised_clipped_weights_diff = noise_utils.add_global_noise(
            model,
            clipped_weights_diff,
            random_state,
            local_sigma * local_clipping_bound * np.sqrt(n_silo_per_round),
            device=device,
        )
        return noised_clipped_weights_diff, len(train_loader)
    elif agg_strategy == METHOD_SILO_LEVEL_DP:
        clipped_weights_diff = noise_utils.global_clip(
            model, weights_diff, local_clipping_bound
        )
        return clipped_weights_diff, len(train_loader)
    elif agg_strategy in METHOD_GROUP_ULDP_SGD:
        return noisy_avg_grads, len(train_loader)
    elif agg_strategy in METHOD_GROUP_ULDP_AVG:
        return noisy_avg_weights_diff, len(train_loader)
    elif agg_strategy == METHOD_PULDP_AVG:
        return noisy_avg_weights_diff, len(train_loader)
    elif agg_strategy == METHOD_PULDP_QC_TEST:
        consume_dp_for_model_optimization(
            q_u_list=q_u_list,
            epsilon_groups=epsilon_groups,
            C_u_list=C_u_list,
            local_sigma=local_sigma,
            accountant_dct=accountant_dct,
        )
        consume_dp_for_model_optimization(
            q_u_list=q_u_list,
            epsilon_groups=epsilon_groups,
            C_u_list=C_u_list,
            local_sigma=local_sigma,
            accountant_dct=accountant_dct,
        )
        consume_dp_for_stepped_model_optimization(
            stepped_q_u_list=stepped_q_u_list,
            epsilon_groups=epsilon_groups,
            stepped_C_u_list_for_optimization=stepped_C_u_list_for_optimization,
            local_sigma=local_sigma,
            accountant_dct=accountant_dct,
        )
        return noisy_avg_weights_diff_dct
    elif agg_strategy == METHOD_PULDP_QC_TRAIN:
        consume_dp_for_model_optimization(
            q_u_list=q_u_list,
            epsilon_groups=epsilon_groups,
            C_u_list=C_u_list,
            local_sigma=local_sigma,
            accountant_dct=accountant_dct,
        )
        if hp_baseline:
            local_loss_diff_dct = {}
        else:
            local_loss_diff_dct = compute_train_loss_for_online_optimization_and_consume_dp_for_train_loss_metric(
                device,
                dataset_name,
                local_delta,
                epsilon_groups,
                accountant_dct,
                model,
                n_silo_per_round,
                user_weights_for_optimization,
                stepped_user_weights_for_optimization,
                user_level_data_loader,
                criterion,
                round_idx,
                q_u_list,
                stepped_q_u_list,
                noisy_avg_weights_diff_dct,
                n_total_round,
                random_state,
                off_train_loss_noise=off_train_loss_noise,
            )
        return noisy_avg_weights_diff_dct, local_loss_diff_dct
    else:
        raise NotImplementedError("Unknown aggregation strategy")


def compute_train_loss_for_online_optimization_and_consume_dp_for_train_loss_metric(
    device,
    dataset_name,
    local_delta,
    epsilon_groups,
    accountant_dct,
    model,
    n_silo_per_round,
    user_weights_for_optimization,
    stepped_user_weights_for_optimization,
    user_level_data_loader,
    criterion,
    round_idx: int,
    q_u_list: List,
    stepped_q_u_list: List,
    local_updated_weights_dct: Dict,
    n_total_round: int,
    random_state: np.random.RandomState,
    off_train_loss_noise: bool = False,
) -> Dict[float, float]:
    # Calculate the difference of the train loss (or approximated train loss like user level accuracy) between the original and the updated sampling rate
    # At the same time, consume privacy for differentially private training loss (or user level accuracy)
    diff_dct = {}

    for eps_u, eps_user_ids in epsilon_groups.items():
        q_u = q_u_list[eps_user_ids[0]]

        original_weights_diff, stepped_weights_diff = local_updated_weights_dct[eps_u]

        # Calculate train loss with original sampling rate for approximated HP gradients
        model = copy.deepcopy(model)
        averaged_param_diff = noise_utils.torch_aggregation(
            [original_weights_diff],
            np.ceil(len(eps_user_ids) * n_silo_per_round * q_u),
        )
        noise_utils.update_global_weights_from_diff(
            averaged_param_diff,
            model,
            learning_rate=1.0,
        )
        if off_train_loss_noise:
            original_test_loss, _ = train_loss(
                device=device,
                model=model,
                dataset_name=dataset_name,
                user_level_data_loader=user_level_data_loader,
                criterion=criterion,
                round_idx=round_idx,
            )
            logger.debug(
                "Original sampling_rate_q = {}, metric = {}".format(
                    q_u, original_test_loss
                )
            )
        else:
            (
                original_user_level_metric,
                local_noise_multiplier,
            ) = dp_train_loss(
                model=model,
                eps_u=eps_u,
                user_weights=user_weights_for_optimization,
                device=device,
                dataset_name=dataset_name,
                epsilon_groups=epsilon_groups,
                user_level_data_loader=user_level_data_loader,
                criterion=criterion,
                local_delta=local_delta,
                n_total_round=n_total_round,
                accountant_dct=accountant_dct,
                random_state=random_state,
                n_silo_per_round=n_silo_per_round,
                round_idx=round_idx,
                sampling_rate_q=q_u,
                current_round=round_idx * 3 + 1,
            )
            # (Consume privacy)
            accountant_dct[eps_u].step(
                noise_multiplier=local_noise_multiplier,
                sample_rate=q_u,
            )
            logger.debug(
                "Original sampling_rate_q = {}, metric = {}".format(
                    q_u, original_user_level_metric
                )
            )

        # Calculate train loss with updated (stepped) sampling rate
        model = copy.deepcopy(model)
        stepped_q_u = stepped_q_u_list[eps_user_ids[0]]
        averaged_param_diff = noise_utils.torch_aggregation(
            [stepped_weights_diff],
            np.ceil(len(eps_user_ids) * n_silo_per_round * stepped_q_u),
        )
        noise_utils.update_global_weights_from_diff(
            averaged_param_diff,
            model,
            learning_rate=1.0,
        )
        if off_train_loss_noise:
            stepped_test_loss, _ = train_loss(
                device=device,
                model=model,
                dataset_name=dataset_name,
                user_level_data_loader=user_level_data_loader,
                criterion=criterion,
                round_idx=round_idx,
            )
            logger.debug(
                "Stepped sampling_rate_q = {}, metric = {}".format(
                    stepped_q_u, stepped_test_loss
                )
            )
        else:
            (
                stepped_user_level_metric,
                stepped_local_noise_multiplier,
            ) = dp_train_loss(
                model=model,
                eps_u=eps_u,
                user_weights=stepped_user_weights_for_optimization,
                device=device,
                dataset_name=dataset_name,
                epsilon_groups=epsilon_groups,
                user_level_data_loader=user_level_data_loader,
                criterion=criterion,
                local_delta=local_delta,
                n_total_round=n_total_round,
                accountant_dct=accountant_dct,
                random_state=random_state,
                n_silo_per_round=n_silo_per_round,
                round_idx=round_idx,
                sampling_rate_q=stepped_q_u,
                current_round=round_idx * 3 + 2,
            )
            # (Consume privacy)
            accountant_dct[eps_u].step(
                noise_multiplier=stepped_local_noise_multiplier,
                sample_rate=stepped_q_u,
            )
            logger.debug(
                "Stepped sampling_rate_q = {}, metric = {}".format(
                    stepped_q_u, stepped_user_level_metric
                )
            )

        # diff < 0 means that the model is improved
        if off_train_loss_noise:
            diff = stepped_test_loss - original_test_loss
        else:
            diff = original_user_level_metric - stepped_user_level_metric

        diff_dct[eps_u] = diff
        logger.debug("eps_u = {}, diff = {}".format(eps_u, diff))

    return diff_dct


def train_loss(
    device,
    model: nn.Module,
    dataset_name,
    user_level_data_loader,
    criterion,
    round_idx=None,
) -> Tuple[float, float]:
    model.eval()

    user_level_total_loss = []
    user_level_metrics = []
    if dataset_name == TCGA_BRCA:
        logger.warning("TCGA_BRCA does not support train_loss")
        return 0, 0

    if dataset_name == HEART_DISEASE:
        from flamby_utils.heart_disease import (
            custom_metric,
        )

        metric = custom_metric()

    for user_id, user_train_loader in user_level_data_loader:
        if dataset_name in [HEART_DISEASE]:
            with torch.no_grad():
                y_pred_final = []
                y_true_final = []
                n_total_data = 0
                train_loss = 0
                for x, y in user_train_loader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    train_loss += loss.item()
                    y_pred_final.append(y_pred.numpy())
                    y_true_final.append(y.numpy())
                    n_total_data += len(y)

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)
            train_metric = metric(y_true_final, y_pred_final)
            logger.debug(
                f"|----- Local test result of round {round_idx}, user {user_id}"
            )
            logger.debug(
                f"\t |----- Local Train/Acc: {train_metric} ({n_total_data}), Local Train/Loss: {train_loss}"
            )

        elif dataset_name == CREDITCARD:
            from sklearn.metrics import roc_auc_score

            criterion = nn.CrossEntropyLoss().to(device)

            with torch.no_grad():
                n_total_data = 0
                train_loss = 0
                y_pred_final = []
                y_true_final = []
                for x, y in user_train_loader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    loss = criterion(y_pred, y.long())
                    train_loss += loss.item()
                    y_pred = y_pred.argmax(dim=1)
                    y_pred_final.append(y_pred.numpy())
                    y_true_final.append(y.numpy())
                    n_total_data += len(y)

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)
            train_metric = roc_auc_score(y_true_final, y_pred_final)
            logger.debug(
                f"|----- Local test result of round {round_idx}, user {user_id}"
            )
            logger.debug(
                f"\t |----- Local Test/ROC_AUC: {train_metric} ({n_total_data}), Local Test/Loss: {train_loss}"
            )
        else:
            with torch.no_grad():
                n_total_data = 0
                train_loss = 0
                test_correct = 0
                for x, labels in user_train_loader:
                    x, labels = x.to(device), labels.to(device)
                    pred = model(x)
                    loss = criterion(pred, labels)
                    train_loss += loss.item()
                    _, predicted = torch.max(pred, 1)
                    test_correct += torch.sum(torch.eq(predicted, labels)).item()
                    n_total_data += len(labels)

            train_metric = test_correct / n_total_data
            logger.debug(
                f"|----- Local test result of round {round_idx}, user {user_id}"
            )
            logger.debug(
                f"\t |----- Local Test/Acc: {train_metric} ({test_correct} / {n_total_data}), Local Test/Loss: {train_loss}"
            )

        user_level_total_loss.append(train_loss / n_total_data)
        user_level_metrics.append(train_metric)

    return sum(user_level_total_loss), sum(user_level_metrics)


def dp_train_loss(
    model: nn.Module,
    eps_u: float,
    user_weights: Dict[int, float],
    device,
    dataset_name,
    epsilon_groups,
    user_level_data_loader,
    criterion,
    local_delta,
    n_total_round,
    accountant_dct,
    random_state,
    n_silo_per_round,
    round_idx=None,
    sampling_rate_q: Optional[float] = None,
    current_round: Optional[int] = None,
) -> Tuple[Tuple[float, float], float]:
    # metric is approximated metric instead of the real train loss
    # it needs to be bounded by 1
    # privacy accounting is done in aggregator.py consume_dp_for_train_loss_metric()
    model.eval()

    if dataset_name == TCGA_BRCA:
        logger.warning(
            "TCGA_BRCA does not currently support train_loss because cox-loss is not well-compatible for user-level setting"
        )
        return 0, 0

    if dataset_name == HEART_DISEASE:
        from flamby_utils.heart_disease import (
            custom_metric,
        )

        metric = custom_metric()

    loss_list = []
    metric_list = []
    raw_metric_list = []

    user_ids_per_eps_u_set = set(epsilon_groups[eps_u])

    for user_id, user_train_loader in user_level_data_loader:
        # to compute model delta per epsilon group
        if user_id not in user_ids_per_eps_u_set:
            continue
        if (  # not sampled users
            user_weights[user_id] <= 0.0
        ):  # for efficiency, if w is encrypted for DDP, it can't work
            continue

        if dataset_name in [HEART_DISEASE]:
            with torch.no_grad():
                y_pred_final = []
                y_true_final = []
                n_total_data = 0
                train_loss = 0
                for x, y in user_train_loader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    train_loss += loss.item()
                    y_pred_final.append(y_pred.numpy())
                    y_true_final.append(y.numpy())
                    n_total_data += len(y)

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)
            train_metric = metric(y_true_final, y_pred_final)
            logger.debug(
                f"|----- Local test result of round {round_idx}, user {user_id})"
            )
            logger.debug(
                f"\t |----- Local Train/Acc: {train_metric} ({n_total_data}), Local Train/Loss: {train_loss}"
            )

        elif dataset_name == CREDITCARD:
            from sklearn.metrics import roc_auc_score

            criterion = nn.CrossEntropyLoss().to(device)

            with torch.no_grad():
                n_total_data = 0
                train_loss = 0
                y_pred_final = []
                y_true_final = []
                for x, y in user_train_loader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    loss = criterion(y_pred, y.long())
                    train_loss += loss.item()
                    y_pred = y_pred.argmax(dim=1)
                    y_pred_final.append(y_pred.numpy())
                    y_true_final.append(y.numpy())
                    n_total_data += len(y)

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)
            # ROC AUC is hard to use in case the label is extremely imbalanced
            # train_metric = roc_auc_score(y_true_final, y_pred_final)
            # Accuracy
            # train_metric = np.sum(y_true_final == y_pred_final) / len(y_true_final)
            if np.any(y_true_final == 1) and np.any(
                y_true_final == 0
            ):  # ポジティブクラスが存在する場合
                train_metric = roc_auc_score(
                    y_true_final, y_pred_final
                )  # またはrecall_score, f1_scoreなど
            else:
                train_metric = np.sum(y_true_final == y_pred_final) / len(y_true_final)
                train_metric = train_metric * 0.1  # lower the importance

            logger.debug(
                f"|----- Local test result of round {round_idx}, user {user_id}"
            )
            logger.debug(
                f"\t |----- Local Test/Accuracy: {train_metric} ({n_total_data}), Local Test/Loss: {train_loss}"
            )

        else:
            with torch.no_grad():
                n_total_data = 0
                train_loss = 0
                test_correct = 0
                for x, labels in user_train_loader:
                    x, labels = x.to(device), labels.to(device)
                    pred = model(x)
                    loss = criterion(pred, labels)
                    train_loss += loss.item()
                    _, predicted = torch.max(pred, 1)
                    test_correct += torch.sum(torch.eq(predicted, labels)).item()
                    n_total_data += len(labels)

            train_metric = test_correct / n_total_data
            logger.debug(
                f"|----- Local test result of round {round_idx}, user {user_id}"
            )
            logger.debug(
                f"\t |----- Local Test/Acc: {train_metric} ({test_correct} / {n_total_data}), Local Test/Loss: {train_loss}"
            )

        # the metric is bounded by C_u per user
        user_level_shrunk_metric = train_metric * user_weights[user_id]
        loss_list.append(train_loss / n_total_data)
        metric_list.append(user_level_shrunk_metric)
        raw_metric_list.append(train_metric)

    logger.debug("metric_list: (", len(metric_list), ")", metric_list)
    noise_multiplier, _ = noise_utils.get_noise_multiplier_with_history(
        sampling_rate_q,
        local_delta,
        epsilon_u=eps_u,
        total_round=n_total_round * 3,
        current_round=current_round,
        current_accountant=accountant_dct[eps_u],
    )
    noise = noise_utils.single_gaussian_noise(
        random_state=random_state,
        std_dev=noise_multiplier / np.sqrt(n_silo_per_round),
        # sensitivity is same for all users in the same epsilon group, and add distributed noise here
    )
    logger.debug("noise:", noise)
    final_shrunk_noisy_metric = np.sum(metric_list) + noise
    final_shrunk_noisy_metric /= sampling_rate_q

    return final_shrunk_noisy_metric, noise_multiplier


def consume_dp_for_model_optimization(
    q_u_list, epsilon_groups, C_u_list, local_sigma, accountant_dct
):
    # (Consume privacy) Update local accountants for model aggregation for each user groups
    for eps_u, eps_user_ids in epsilon_groups.items():
        q_u = q_u_list[eps_user_ids[0]]
        C_u = C_u_list[eps_user_ids[0]]
        accountant_dct[eps_u].step(
            noise_multiplier=local_sigma / C_u,
            sample_rate=q_u,
        )


def consume_dp_for_stepped_model_optimization(
    stepped_q_u_list,
    epsilon_groups,
    stepped_C_u_list_for_optimization,
    local_sigma,
    accountant_dct,
):
    # (Consume privacy) Update local accountants for model aggregation for each user groups
    for eps_u, eps_user_ids in epsilon_groups.items():
        q_u = stepped_q_u_list[eps_user_ids[0]]
        C_u = stepped_C_u_list_for_optimization[eps_user_ids[0]]
        accountant_dct[eps_u].step(
            noise_multiplier=local_sigma / C_u,
            sample_rate=q_u,
        )
