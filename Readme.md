# Your Privacy Depends on Others: Collusion Vulnerabilities in Individual Differential Privacy

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](link-to-arxiv)

Official implementation of **"Your Privacy Depends on Others: Collusion Vulnerabilities in Individual Differential Privacy"** accepted at SaTML 2026.

**Authors:** Johannes Kaiser, Alexander Ziller, Eleni Triantafillou, Daniel Rückert, Georgios Kaissis

## Abstract

Individual Differential Privacy (iDP) promises users control over their privacy through personalized privacy budgets. However, we reveal a critical vulnerability: **in sampling-based iDP mechanisms, your privacy risk depends not just on your own budget choice, but on everyone else's choices too.**

We demonstrate that:
- 🔍 Privacy co-dependencies create exploitable attack vectors
- ⚔️ Adversaries can manipulate budgets to increase targeted individuals' vulnerability
- 📊 **62% of targeted individuals** were successfully attacked in our evaluation
- 🛡️ We propose (εᵢ, δᵢ, Δ)-iDP to bound excess vulnerabilities

<p align="center">
  <img src="figures/figure1.png" alt="Privacy profiles showing different mechanisms" width="700"/>
  <br>
  <em>Figure 1: Privacy profiles and adversarial advantage of mechanisms calibrated to (2, 0.08)-DP show vastly different protections despite identical (ε, δ) guarantees.</em>
</p>

## Key Findings

### The Core Problem

Sampling-based iDP mechanisms adjust per-sample sampling rates to meet individual privacy budgets. However:

1. **Privacy Co-dependence**: To maintain fixed batch sizes, if some users choose strict budgets (low sampling rates), others must have higher sampling rates
2. **Incomplete Specification**: Mechanisms are calibrated only to a single (ε, δ) point, leaving the full privacy profile unconstrained
3. **Excess Vulnerability**: Different mechanisms with identical (ε, δ) can expose users to vastly different real-world risks

<p align="center">
  <img src="figures/figure3.png" alt="Theoretical adversarial advantage" width="600"/>
  <br>
  <em>Figure 3: Theoretical adversarial advantage for data points with ε₁=8 depends heavily on the proportion and budget of other groups.</em>
</p>

### Novel Attacks

We introduce two attacks that exploit this vulnerability **while operating entirely within DP's formal guarantees**:

#### 1. Budget Manipulation Attack
A central adversary (e.g., model trainer) strategically assigns privacy budgets within contractual limits to maximize a target's vulnerability.

<p align="center">
  <img src="figures/figure4.png" alt="Budget Manipulation Attack" width="500"/>
</p>

#### 2. Collusion Attack
Multiple data contributors coordinate their privacy budget choices to increase a victim's risk—no central authority needed.

<p align="center">
  <img src="figures/figure5.png" alt="Collusion Attack" width="500"/>
</p>

### Empirical Results

<p align="center">
  <img src="figures/figure6.png" alt="Attack results" width="600"/>
  <br>
  <em>Figure 6: Privacy score increases for targeted individuals in budget manipulation attacks. Each line represents one attacked sample, with 62% showing significantly increased vulnerability.</em>
</p>

**Table I** shows statistically significant increases in privacy scores (membership inference susceptibility) across multiple datasets when privacy budget distributions are manipulated:

| Dataset | ε₁ | ε₂ | Significance (Group 1) | Significance (Group 2) |
|---------|----|----|----------------------|----------------------|
| Credit Card Default | 4 | 20 | ★★★ (ES: 20.2) | ★★★ (ES: 17.5) |
| MNIST [4,1000] | 16 | 50 | ★★★ (ES: 21.2) | ★★★ (ES: 17.7) |
| CIFAR-10 | 16 | 50 | ★★★ (ES: 13.9) | ★★★ (ES: 53.5) |
| HAM10k [2000] | 8 | 32 | ★★★ (ES: 31.9) | ★★★ (ES: 37.2) |

★★★ indicates p ≤ 0.001; ES = Effect Size

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/idp-collusion-vulnerabilities.git
cd idp-collusion-vulnerabilities

# Create conda environment
conda create -n idp-vuln python=3.9
conda activate idp-vuln

# Install dependencies
pip install -r requirements.txt
```

## Repository Structure

```
.
├── src/
│   ├── mechanisms/          # iDP mechanism implementations
│   │   ├── sampling_based.py
│   │   └── sensitivity_based.py
│   ├── attacks/             # Attack implementations
│   │   ├── budget_manipulation.py
│   │   └── collusion.py
│   ├── auditing/            # LiRA and MIA implementations
│   │   └── lira.py
│   └── utils/               # Helper functions
├── experiments/             # Experiment scripts
│   ├── excess_vulnerability.py
│   ├── budget_manipulation_attack.py
│   └── collusion_attack.py
├── configs/                 # Configuration files
├── data/                    # Dataset directory
├── figures/                 # Generated figures
└── requirements.txt
```

## Running Experiments

### 1. Demonstrating Excess Vulnerability

Reproduce the results from Table I showing how privacy budget distributions affect individual vulnerability:

```bash
python experiments/excess_vulnerability.py \
    --dataset credit_card_default \
    --epsilon1 4 \
    --epsilon2 20 \
    --delta 1e-12 \
    --proportions 0.2,0.4,0.6,0.8 \
    --num_seeds 5 \
    --num_shadows 512
```

**Options:**
- `--dataset`: Choose from `credit_card_default`, `german_credit`, `mnist`, `cifar10`, `organc_mnist`, `organs_mnist`, `pneumonia`, `ham10k`
- `--epsilon1`, `--epsilon2`: Privacy budgets for the two groups
- `--delta`: Delta parameter (default: 1e-12)
- `--proportions`: Comma-separated proportions for group 1
- `--num_shadows`: Number of shadow models for LiRA (default: 512)

### 2. Budget Manipulation Attack

Evaluate the budget manipulation attack on targeted individuals:

```bash
python experiments/budget_manipulation_attack.py \
    --dataset credit_card_default \
    --target_epsilon 32 \
    --adversary_epsilons 4,8,16,32 \
    --delta 1e-12 \
    --num_targets 1000 \
    --num_shadows 64
```

**Options:**
- `--target_epsilon`: Privacy budget assigned to target individuals
- `--adversary_epsilons`: Comma-separated list of budgets the adversary can choose
- `--num_targets`: Number of random samples to target
- `--num_shadows`: Number of shadow models per target (default: 64)

### 3. Collusion Attack

Simulate collusion attacks with varying proportions of colluding parties:

```bash
python experiments/collusion_attack.py \
    --dataset mnist \
    --target_epsilon 32 \
    --collusion_epsilon 4 \
    --collusion_proportions 0.2,0.4,0.6,0.8 \
    --delta 1e-12 \
    --num_targets 100
```

**Options:**
- `--collusion_epsilon`: Budget chosen by colluding parties
- `--collusion_proportions`: Proportions of dataset controlled by colluders

### 4. Sensitivity-Based iDP Ablation

Compare sampling-based vs sensitivity-based iDP (Table II):

```bash
python experiments/sensitivity_based_ablation.py \
    --dataset mnist \
    --epsilon1 16 \
    --epsilon2 50 \
    --delta 1e-12 \
    --num_shadows 512
```

## Proposed Mitigation: (εᵢ, δᵢ, Δ)-iDP

We propose extending the iDP contract to include a Δ-divergence bound:

```python
from src.mechanisms.delta_idp import DeltaIDP

# Create mechanism with excess vulnerability bounds
mechanism = DeltaIDP(
    epsilon_i=[4, 8, 16, 32],  # Individual budgets
    delta=1e-12,
    max_delta_divergence=0.05  # Maximum excess vulnerability
)

# Mechanism will reject parameter choices that exceed Δ bound
params = mechanism.calibrate(dataset)
```

<p align="center">
  <img src="figures/figure7.png" alt="Delta-iDP allowable region" width="500"/>
  <br>
  <em>Figure 7: Valid region for (8, 10⁻⁵, 0.05)-DP bounded by Δ-approximately Blackwell-dominant mechanisms.</em>
</p>

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{kaiser2026privacy,
  title={Your Privacy Depends on Others: Collusion Vulnerabilities in Individual Differential Privacy},
  author={Kaiser, Johannes and Ziller, Alexander and Triantafillou, Eleni and Rückert, Daniel and Kaissis, Georgios},
  booktitle={IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)},
  year={2026}
}
```

## Key Takeaways

⚠️ **For Practitioners:**
- iDP systems require auditing beyond single (ε, δ) points
- Privacy budget distributions must be monitored and controlled
- Consider (εᵢ, δᵢ, Δ)-iDP contracts to bound excess risk

⚠️ **For Data Contributors:**
- Your privacy risk in sampling-based iDP depends on others' choices
- Demand transparency about privacy budget distributions
- Request Δ-divergence bounds in privacy contracts

⚠️ **For Researchers:**
- Single-point (ε, δ) calibration is insufficient
- Full privacy profiles must be considered
- Federated learning scenarios are particularly vulnerable to collusion

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open a GitHub issue or contact:
- Johannes Kaiser: johannes.kaiser@tum.de

## Acknowledgments

This work was supported by the European Union under Grant Agreement 101100633 (EUCAIM) and the German Ministry of Education and Research through the Medical Informatics Initiative (PrivateAIM Project, grant no. 01ZZ2316C).