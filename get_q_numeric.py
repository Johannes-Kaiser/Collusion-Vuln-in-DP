from opacus_new.accountants.utils import get_sample_rates as get_sample_rates_boenisch
import numpy as np
from dp_accounting.pld import privacy_loss_distribution
from typing import List, Tuple
from time import time

MAX_SIGMA = 1e6
MIN_Q = 1e-9
MAX_Q = 1
VDI = 0.1


def get_sample_rate_numeric(
    target_epsilon: float,
    target_delta: float,
    noise_multiplier: float,
    steps: int,
    precision: float = 0.001,
) -> float:
    q_low, q_high = MIN_Q, MAX_Q
    accountant = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sampling_prob=q_low,
        value_discretization_interval=VDI,
    ).self_compose(steps)
    eps_low = accountant.get_epsilon_for_delta(target_delta)
    if eps_low > target_epsilon:
        raise ValueError("The privacy budget is too low.")
    accountant = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sampling_prob=q_high,
        value_discretization_interval=VDI,
    ).self_compose(steps)
    eps_high = accountant.get_epsilon_for_delta(target_delta)
    print("Getting epsilon")
    while eps_high < 0:  # decrease q_high whenever a numerical error happens
        q_high *= 0.9
        accountant = privacy_loss_distribution.from_gaussian_mechanism(
            standard_deviation=noise_multiplier,
            sampling_prob=q_high,
            value_discretization_interval=VDI,
        ).self_compose(steps)
        eps_high = accountant.get_epsilon_for_delta(target_delta)
    if eps_high < target_epsilon:
        raise ValueError(
            f"The given noise_multiplier {noise_multiplier} is " f"too high."
        )
    print("getting q")
    while q_low / q_high < 1 - precision:
        q = (q_low + q_high) / 2
        accountant = privacy_loss_distribution.from_gaussian_mechanism(
            standard_deviation=noise_multiplier,
            sampling_prob=q,
            value_discretization_interval=VDI,
        ).self_compose(steps)
        eps = accountant.get_epsilon_for_delta(target_delta)
        if eps < target_epsilon:
            q_low = q
        else:
            q_high = q

    return q_low


def get_sample_rates_numeric(
    ratios: List[float],
    target_epsilons: List[float],
    target_delta: float,
    default_sample_rate: float,
    steps: int,
    precision: float = 0.001,
    **kwargs,
) -> Tuple[float, np.ndarray]:
    n_groups = len(ratios)
    ratios = np.asarray(ratios)
    sigma_low, sigma_high = 1e-5, 1e-3
    for group, target_epsilon in enumerate(target_epsilons):
        eps_high = float("inf")
        sigma_high_group = 1e-3
        print("getting sigma")
        while eps_high > target_epsilon:
            sigma_high_group = 2 * sigma_high_group
            if sigma_high_group > sigma_high:
                sigma_high = sigma_high_group
            accountant = privacy_loss_distribution.from_gaussian_mechanism(
                standard_deviation=sigma_high_group,
                sampling_prob=default_sample_rate,
                value_discretization_interval=VDI,
            ).self_compose(steps)
            eps_high = accountant.get_epsilon_for_delta(target_delta)
            if sigma_high_group > MAX_SIGMA:
                raise ValueError(
                    f"The privacy budget ({target_epsilon}) of"
                    f"group {group} is too low."
                )

    q_mean = MAX_Q
    qs = np.array([q_mean] * n_groups, dtype=np.float32)
    while sigma_low / sigma_high < 1 - precision:
        sigma = (sigma_high + sigma_low) / 2
        q_mean = 0
        for group, target_epsilon in enumerate(target_epsilons):
            try:
                q = get_sample_rate_numeric(
                    target_epsilon=target_epsilon,
                    target_delta=target_delta,
                    noise_multiplier=sigma,
                    steps=steps,
                    precision=precision,
                    **kwargs,
                )
                qs[group] = q
                q_mean += q * ratios[group]
                if q_mean > default_sample_rate:
                    sigma_high = sigma
                    break
            except ValueError:
                continue
        q_mean = sum(qs * ratios)
        if q_mean > default_sample_rate:
            sigma_high = sigma
        else:
            sigma_low = sigma
    return sigma_high, list(qs)


if __name__ == "__main__":
    epsilons = [1.0, 20.0]
    ratios = [0.05, 0.95]
    target_delta = 1e-5
    n_data = 10_000
    default_batch_size = 64
    epochs = 100
    tick = time()
    sigma_sample_boenisch, qs_boenisch = get_sample_rates_boenisch(
        ratios=ratios,
        target_epsilons=epsilons,
        target_delta=target_delta,
        default_sample_rate=default_batch_size / n_data,
        steps=epochs * n_data // default_batch_size,
    )
    tock = time()
    print(f"Boenisch: sigma_sample = {sigma_sample_boenisch}, qs = {qs_boenisch}")
    print(f"Boenisch took {tock-tick}s")
    tick = time()
    sigma_sample_numeric, qs_numeric = get_sample_rates_numeric(
        ratios=ratios,
        target_epsilons=epsilons,
        target_delta=target_delta,
        default_sample_rate=default_batch_size / n_data,
        steps=epochs * n_data // default_batch_size,
    )
    tock = time()
    print(f"Numeric: sigma_sample = {sigma_sample_numeric}, qs = {qs_numeric}")
    print(f"Numeric took {tock-tick}s")
