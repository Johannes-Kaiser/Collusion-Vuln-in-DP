# Evaluating Individual Differential Privacy (iDP)

This document presents an evaluation of individual differential privacy (iDP) in the context of clipping and sampling settings.

Todos:
* Still heavily RDP based (However very fast now)
* Empirically evaluate how this behaves with increasing number of iterations
* Find setting where they diverge even for small delta
* Define MIA for private model training


## Experimental Results

The following image illustrates results from our experiments:

![Experiment Results](./experiments/experiment_b0bc6513edce414c4065c148efe05e0a/combined_sampling.png)
![Experiment Results](./experiments/experiment_b0bc6513edce414c4065c148efe05e0a/combined_clipping.png)

## Membership Inference

Membership Inference is working, however has not been tried on private (including iDP) models

![MNIST Binary Results - FPR vs TPR (Target)](./.images/fprtpr_target.png)