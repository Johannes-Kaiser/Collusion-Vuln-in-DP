## Load Imports

from opacus_new.accountants.utils import get_sample_rate, get_noise_multiplier
import numpy as np
import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", message="Secure RNG turned off.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

aggregator = np.max
list_of_epsilons_1 = [8, 16, 32]
list_of_epsilons_2 = [8, 16]
delta = 1e-5
max_grad_norm = 1.0
noise_1 = get_noise_multiplier(target_epsilon=aggregator(np.array(list_of_epsilons_1)), target_delta=delta, sample_rate=1, epochs=10, accountant='rdp')
noise_2 = get_noise_multiplier(target_epsilon=aggregator(np.array(list_of_epsilons_2)), target_delta=delta, sample_rate=1, epochs=10, accountant='rdp')

print(f"Noise for group 1 (epsilons {list_of_epsilons_1}): {noise_1}")
print(f"Noise for group 2 (epsilons {list_of_epsilons_2}): {noise_2}")


sample_rates_1 = {k: get_sample_rate(target_epsilon=k, target_delta=delta, noise_multiplier=noise_1, steps=10, accountant='rdp') for k in list_of_epsilons_1}
sample_rates_2 = {k: get_sample_rate(target_epsilon=k, target_delta=delta, noise_multiplier=noise_2, steps=10, accountant='rdp') for k in list_of_epsilons_2}