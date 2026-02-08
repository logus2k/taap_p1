# Further improvements (to be considered)

## Other Loss Functions:

1. Relativistic GAN (RaGAN):
 - D outputs how much more real the real image is compared to fake
 - Often improves stability and quality

2. Non-Saturating GAN (NSGAN):
 - G maximizes log(D(G(z))) instead of minimizing log(1-D(G(z)))
 - Stronger gradients early in training

3. Softplus Loss:
 - Uses softplus activation instead of sigmoid
 - More numerically stable

4. Focal Loss for GANs:
 - Down-weights easy examples
 - Helps with class imbalance

## Other Recommendations:

1. Learning Rate Scheduling:
 - Linear decay after 50% of training
 - Helps fine-tune in later stages

2. Exponential Moving Average (EMA) of Generator:
 - Keep a smoothed version of G weights
 - Often produces better final samples

3. Progressive Growing:
 - Start with low resolution, gradually increase
 - For higher resolution tasks

4. Self-Attention:
 - Add attention layers to capture global dependencies
 - Significant improvement for complex datasets

5. Multiple Discriminators:
 - Ensemble of D networks
 - Can improve robustness



| Strategy | FID (v8) | FID (v9) | Class-Consistency (v8) | Class-Consistency (v9) |
|----------|----------|----------|------------------------|------------------------|
| BCE | 7.74 | 7.91 | 97.64% | 97.66% |
| LSGAN | 6.30 | 7.66 | 97.38% | 97.02% |
| Hinge | 10.44 | 10.22 | 97.18% | 97.26% |
| **WGAN-GP** | **68.03** | **4.93** | **71.02%** | **99.18%** |

