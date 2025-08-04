import numpy as np
# from opacus_new.accountants.utils import get_sample_rates as get_sample_rates_boenisch
from dp_accounting.rdp.rdp_privacy_accountant import (
    _compute_rdp_poisson_subsampled_gaussian,
    compute_epsilon,
)
import concurrent.futures
import os
from time import time
from functools import cache
from absl import logging

logging.set_verbosity(
    logging.ERROR
)  # suppress dp_accounting numerical warnings for RDP

MIN_Q = 1e-14
MAX_Q = 1
MAX_SIGMA = 1e6
DEFAULT_RDP_ORDERS = (
    [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512, 1024]
)


@cache
def _rdp_vec(sigma: float, q: float):
    return _compute_rdp_poisson_subsampled_gaussian(
        q=q, noise_multiplier=sigma, orders=DEFAULT_RDP_ORDERS
    )


def epsilon_for_delta(sigma: float, q: float, steps: int, delta: float) -> float:
    rdp = _rdp_vec(sigma, q) * steps
    epsilon, _ = compute_epsilon(DEFAULT_RDP_ORDERS, rdp, delta)
    return epsilon


def get_sample_rate_estimate_rdp(
    target_epsilon: float,
    target_delta: float,
    noise_multiplier: float,
    steps: int,
    precision: float = 0.001,
) -> float:
    q_low, q_high = MIN_Q, MAX_Q
    eps_low = epsilon_for_delta(
        sigma=noise_multiplier, q=q_low, steps=steps, delta=target_delta
    )
    if eps_low > target_epsilon:
        raise ValueError("The privacy budget is too low.")
    eps_high = epsilon_for_delta(
        sigma=noise_multiplier, q=q_high, steps=steps, delta=target_delta
    )
    while eps_high < 0:
        q_high *= 0.9
        eps_high = epsilon_for_delta(
            sigma=noise_multiplier, q=q_high, steps=steps, delta=target_delta
        )
    if eps_high < target_epsilon:
        raise ValueError(f"The given noise_multiplier {noise_multiplier} is too high.")

    while q_low / q_high < 1 - precision:
        q = (q_low + q_high) / 2
        eps = epsilon_for_delta(
            sigma=noise_multiplier, q=q, steps=steps, delta=target_delta
        )
        if eps < target_epsilon:
            q_low = q
        else:
            q_high = q

    return q_low


def _q_worker(args: tuple[int, float, float, float, int]):
    idx, eps, sigma, delta, steps = args
    q = get_sample_rate_estimate_rdp(
        target_epsilon=eps,
        target_delta=delta,
        noise_multiplier=sigma,
        steps=steps,
    )
    return idx, q
    # except ValueError:
    #     print("Error in rate compuation")
    #     return idx, None


def get_sample_rate_estimates_rdp(
    ratios: list[float],
    target_epsilons: list[float],
    target_delta: float,
    default_sample_rate: float,
    steps: int,
    precision: float = 0.001,
):
    # print("Numeric")
    ratios = np.asarray(ratios, dtype=np.float32)
    n_groups = len(target_epsilons)
    sigma_low, sigma_high = 1e-5, 1e-3
    for group, target_epsilon in enumerate(target_epsilons):
        eps_high, sigma_high_group = np.inf, 1e-3
        while eps_high > target_epsilon:
            sigma_high_group *= 2
            sigma_high = max(sigma_high, sigma_high_group)
            eps_high = epsilon_for_delta(
                sigma=sigma_high_group,
                q=default_sample_rate,
                steps=steps,
                delta=target_delta,
            )
            if sigma_high_group > MAX_SIGMA:
                raise ValueError(
                    f"The privacy budget ({target_epsilon}) of group {group} is too low."
                )

    qs = np.zeros(n_groups, dtype=np.float32)
    max_workers = min(n_groups, os.cpu_count() or 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        while sigma_low / sigma_high < 1 - precision:
            sigma = 0.5 * (sigma_low + sigma_high)
            tasks = [
                (idx, eps, sigma, target_delta, steps)
                for idx, eps in enumerate(target_epsilons)
            ]
            # for idx, task in enumerate(tasks):
            #     _, q = _q_worker(task)
            #     qs[idx] = q if q is not None else 0.0
            for idx, q in pool.map(_q_worker, tasks):
                qs[idx] = q if q is not None else 0.0
            q_mean = float(np.dot(qs, ratios))
            if q_mean > default_sample_rate:
                sigma_high = sigma
            else:
                sigma_low = sigma
    return sigma_high, qs


if __name__ == "__main__":
    epsilons = [1.0, 5.0, 10.0, 15.0, 20.0]
    ratios = [0.05, 0.10, 0.15, 0.40, 0.30]
    assert np.isclose(sum(ratios), 1.0)
    target_delta = 1e-5
    n_data = 50_000
    default_batch_size = 1024
    epochs = 100
    tick = time()
    sigma_sample_rdp, qs_rdp = get_sample_rate_estimates_rdp(
        ratios,
        epsilons,
        target_delta,
        default_batch_size / n_data,
        epochs * n_data // default_batch_size,
    )
    tock = time()
    time_rdp = tock - tick
    print(f"RDP: sigma_sample = {sigma_sample_rdp:.4f}, qs = {qs_rdp}")
    print(f"RDP took {time_rdp:.4f}s")
    # tick = time()
    # sigma_sample_boenisch, qs_boenisch = get_sample_rates_boenisch(
    #     ratios,
    #     epsilons,
    #     target_delta,
    #     default_batch_size / n_data,
    #     epochs * n_data // default_batch_size,
    # )
    # tock = time()
    # time_boenisch = tock - tick
    # print(
    #     f"Boenisch: sigma_sample = {sigma_sample_boenisch:.4f}, qs = {np.array(qs_boenisch)}"
    # )
    # print(f"Boenisch took {time_boenisch:.4f}s")
