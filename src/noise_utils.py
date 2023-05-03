from collections import OrderedDict
import numpy as np
import torch
import warnings
from opacus.accountants.analysis import rdp as analysis


def global_clip(grad, clipping_bound):
    """
    Clip the L2-norm of parameters of the local trained models for DP.
    """
    total_norm = torch.norm(
        torch.stack([torch.norm(grad[k], 2.0) for k in grad.keys()]),
        2.0,
    )
    for k in grad.keys():
        clip_coef = clipping_bound / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for k in grad.keys():
            grad[k].mul_(clip_coef_clamped)
    return grad


# from typing import List, Tuple
# def global_clip_list(
#     raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]],
#     clipping_bound: float,
# ):
#     """
#     Clip the L2-norm of parameters of the local trained models for DP.

#     Param:
#         raw_client_model_or_grad_list (list): the list of local trained models from the selected silos.
#         clipping_bound (float): the L2 clipping bound.
#     """
#     new_grad_list = []
#     for n_sample, local_grad in raw_client_model_or_grad_list:
#         local_grad = global_clip(local_grad, clipping_bound)
#         new_grad_list.append((n_sample, local_grad))
#     return new_grad_list


def add_global_noise(grad, random_state: np.random.RandomState, std_dev: float):
    new_grad = OrderedDict()
    for k in grad.keys():
        new_grad[k] = _compute_new_grad(grad[k], random_state, std_dev)
    return new_grad


def _compute_new_grad(grad, random_state: np.random.RandomState, std_dev: float):
    # Gaussian noise
    noise = torch.Tensor(random_state.normal(0, std_dev, size=grad.shape))
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
        raise ValueError(
            "The group size must be a power of 2, but got group size = {}".format(
                group_k
            )
        )

    orders_vec = [group_k * 2 + x / 10.0 for x in range(1, 100)] + list(
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
    return eps[idx_opt], group_orders_vec[idx_opt]
