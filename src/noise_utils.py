from collections import OrderedDict
from typing import Dict, List
import numpy as np
import torch
import warnings
from opacus.accountants.analysis import rdp as analysis


def torch_aggregation(raw_grad_list: List[Dict], N: int) -> Dict:
    """
    Aggregate the local trained models from the selected silos for Pytorch model.

    Params:
        raw_grad_list (list): the list of local trained models from the selected silos.
    Return:
        averaged_params (dict): the averaged model parameters.
    """
    avg_params = raw_grad_list[0]
    w = 1.0 / N

    for k in avg_params.keys():
        for i in range(0, len(raw_grad_list)):
            local_model_params = raw_grad_list[i]
            if i == 0:
                avg_params[k] = local_model_params[k] * w
            else:
                avg_params[k] += local_model_params[k] * w
    return avg_params


def global_clip(
    model: torch.nn.Module, params: Dict, clipping_bound: float
) -> OrderedDict:
    """
    Clip the L2-norm of parameters of the local trained models for DP.
    """
    sensitive_params = [
        params[name] for name, param in model.named_parameters() if param.requires_grad
    ]
    total_norm = torch.norm(
        torch.stack([torch.norm(g, 2.0) for g in sensitive_params]), 2.0
    )
    clip_coef = clipping_bound / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    clipped_params = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            clipped_params[name] = params[name] * clip_coef_clamped
        else:
            clipped_params[name] = params[name]
    return clipped_params


def multiple_weights(model: torch.nn.Module, params: OrderedDict, weight: float):
    weighted_params = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            weighted_params[name] = weight * params[name]
        else:
            weighted_params[name] = params[name]
    return weighted_params


def add_global_noise(
    model, grad, random_state: np.random.RandomState, std_dev: float, device: str
):
    new_grad = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            new_grad[name] = _compute_new_grad(
                grad[name], random_state, std_dev, device
            )
        else:
            new_grad[name] = grad[name]
    return new_grad


def _compute_new_grad(
    grad, random_state: np.random.RandomState, std_dev: float, device: str
):
    # Gaussian noise
    noise = torch.tensor(
        random_state.normal(0, std_dev, size=grad.shape), device=torch.device(device)
    )
    return noise + grad


def get_group_privacy_spent(
    group_k: int, accountant_history: list, delta: float
) -> float:
    """
    Following Proposition 2 of RDP paper https://arxiv.org/pdf/1702.07476.pdf

    Returns:
        Pair of epsilon and optimal order alpha.
    """
    if not accountant_history:
        return 0.0

    if group_k > 0 and (group_k & (group_k - 1)) != 0:
        return 0.0, 0.0
        raise ValueError(
            "The group size must be a power of 2, but got group size = {}".format(
                group_k
            )
        )

    orders_vec = [group_k * 2 + x / 10.0 for x in range(0, 100)] + list(
        np.linspace(group_k + 10, group_k * 64, 20)
    )

    rdp_vec = sum(
        [
            analysis.compute_rdp(
                q=sample_rate,
                noise_multiplier=noise_multiplier,
                steps=num_steps,
                orders=orders_vec,
            )
            for (noise_multiplier, sample_rate, num_steps) in accountant_history
        ]
    )

    if len(orders_vec) != len(rdp_vec):
        raise ValueError(
            f"Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
        )

    c = np.log2(group_k)
    group_rdp_vec = np.array([3**c * rdp for rdp in rdp_vec])
    group_orders_vec = np.array([a / group_k for a in orders_vec])

    eps = (
        group_rdp_vec
        - (np.log(delta) + np.log(group_orders_vec)) / (group_orders_vec - 1)
        + np.log((group_orders_vec - 1) / group_orders_vec)
    )

    # special case when there is no privacy
    if np.isnan(eps).all():
        return np.inf, np.nan

    idx_opt = np.nanargmin(eps)  # Ignore NaNs
    if idx_opt == 0 or idx_opt == len(eps) - 1:
        extreme = "smallest" if idx_opt == 0 else "largest"
        warnings.warn(
            f"Optimal order is the {extreme} alpha. Please consider expanding the range of alphas to get a tighter privacy bound."
        )
    return eps[idx_opt], orders_vec[idx_opt] * group_k


def get_normal_group_privacy_spent(
    group_k: int, accountant_history: list, delta: float
) -> tuple[float, float]:
    if group_k == 1:
        eps, alpha = get_privacy_spent(delta=delta, history=accountant_history)
        return eps, delta
    if (
        group_k > 62
    ):  # using sigma=5.0 and opacus accountant, group_k > 62 cannot compute
        warnings.warn(
            "GROUP-k is larger than 62, compute for group_k=62, which means fairly underestimate"
        )
        group_k = 62

    upper = np.log(delta)
    lower = -1e20
    group_delta = lower
    accuracy = delta / 1e3
    first = True
    while np.abs(delta - group_delta) > accuracy:
        middle = (upper + lower) / 2.0
        eps, alpha = get_privacy_spent(log_delta=middle, history=accountant_history)
        group_esp, group_delta = convert_to_group_privacy(
            epsilon=eps, log_delta=middle, group_k=group_k
        )
        if first:
            if group_delta == np.inf:
                raise ValueError("Cannot compute group delta")
            first = False
        upper = middle if group_delta > delta else upper
        lower = middle if group_delta < delta else lower
    return group_esp, group_delta


def get_privacy_spent(*, history, delta: float = None, log_delta: float = None):
    DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    alphas = DEFAULT_ALPHAS
    rdp = sum(
        [
            analysis.compute_rdp(
                q=sample_rate,
                noise_multiplier=noise_multiplier,
                steps=num_steps,
                orders=alphas,
            )
            for (noise_multiplier, sample_rate, num_steps) in history
        ]
    )

    orders_vec = np.atleast_1d(alphas)
    rdp_vec = np.atleast_1d(rdp)

    if len(orders_vec) != len(rdp_vec):
        raise ValueError(
            f"Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
        )

    if log_delta is None:
        eps = (
            rdp_vec
            - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
            + np.log((orders_vec - 1) / orders_vec)
        )
    else:
        eps = (
            rdp_vec
            - (log_delta + np.log(orders_vec)) / (orders_vec - 1)
            + np.log((orders_vec - 1) / orders_vec)
        )

    # special case when there is no privacy
    if np.isnan(eps).all():
        return np.inf, np.nan

    idx_opt = np.nanargmin(eps)  # Ignore NaNs
    if idx_opt == 0 or idx_opt == len(eps) - 1:
        extreme = "smallest" if idx_opt == 0 else "largest"
        warnings.warn(
            f"Optimal order is the {extreme} alpha. Please consider expanding the range of alphas to get a tighter privacy bound."
        )
    return eps[idx_opt], orders_vec[idx_opt]


def convert_to_group_privacy(
    epsilon: float, group_k: int, log_delta: float = None, delta: float = None
):
    if log_delta is None:
        return epsilon * group_k, group_k * np.exp(
            (group_k - 1) * epsilon + np.log(delta)
        )
    else:
        return epsilon * group_k, group_k * np.exp((group_k - 1) * epsilon + log_delta)
