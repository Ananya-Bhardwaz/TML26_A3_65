# Adversarial Robustness

This project focuses on training an image classifier that is robust against adversarial attacks.
The goal is to maximize a combined score that equally weights clean accuracy and robustness accuracy
on a test set.

We implement an **Adversarially Robust Image Classifier** for a **9-class CIFAR-style dataset**
using the following techniques:

- PGD adversarial training
- OneCycleLR scheduling with cosine annealing
- Label smoothing regularization
- Data augmentation
- Gradient clipping

## Methods Used

- **PGD Adversarial Training** — During each training step, a PGD attack generates worst-case
  adversarial examples within a ball of radius ε=8/255. The model is then trained on these
  adversarial examples, allowing it to correctly classify perturbed inputs.

- **OneCycleLR Scheduler** — Learning rate follows a one-cycle cosine annealing policy with a
  short 5% warmup phase. This speeds up convergence and improves final accuracy compared to
  step-decay schedules.

- **Label Smoothing** — Cross-entropy loss uses a label smoothing factor of 0.1 to prevent
  overconfident predictions and improve generalization.

- **Gradient Clipping** — Gradients are clipped to a max norm of 1.0 during training for
  numerical stability.

- **Data Augmentation** — RandomCrop (32×32 with padding=4) and RandomHorizontalFlip are
  applied during training only.

## Final PGD Attack Parameters

The training attack uses the following configuration:

```
epsilon  = 8 / 255      # perturbation budget (l-inf)
alpha    = 2 / 255      # step size per PGD iteration
steps    = 14           # PGD iterations during training
```

Validation robustness is evaluated using a stronger PGD-20 attack to avoid overfitting to
the exact training attack strength.

## Final Rank Fusion Weights (Training Hyperparameters)

```
arch            = resnet50
epochs          = 90
batch_size      = 128
lr_max          = 0.1
weight_decay    = 5e-4
label_smoothing = 0.1
momentum        = 0.9 (Nesterov SGD)
seed            = 42
```

ResNet-50 was chosen because its greater capacity allows it to retain more discriminative
features under the adversarial training constraint, outperforming ResNet-18 and ResNet-34.

## Setup

- Dataset: 50,000 images of shape 3×32×32 across 9 classes (provided as `train.npz`)
- Model: ResNet-50 with final layer replaced to output 9 classes
- Validation split: 10% (5,000 images), used only for checkpoint selection
- All training done in adversarial mode — no clean-only training epochs

## Results

The evaluation metric is **Score = 0.5 × clean accuracy + 0.5 × robust accuracy**.

Our leaderboard results:

- **Clean training baseline:** clean=0.83, robust=0.00, score≈0.415
- **FGSM adversarial training:** clean≈0.72, robust≈0.30, score≈0.510
- **ResNet-18 PGD fine-tune:** clean=0.7228, robust=0.3352, score=0.529
- **Final ResNet-50 PGD (90 epochs):** clean=0.7008, robust=0.4818, score=**0.5913**

## Final Observations

- Clean training gives zero robustness — adversarial training is important.
- FGSM training is fast but it produces weaker robustness than PGD.
- ResNet-50 outperforms smaller architectures in the same adversarial training setup.
- OneCycleLR + label smoothing improved both clean and robust accuracy over standard SGD.
- Training just on adversarial examples (not mixing with clean) gave the best combined score.



## Takeaway

Adversarial training is currently one of the most practical and effective defence mechanisms, but it has its drawbacks. Lack of Robustness checkup and accuracy along with high computational cost being some. This just shows that Robustness should be treated as one of the main objectives at the beginning of model creation rather than a later add-on. It also highlights the importance of active research into Robustness methods to ensure safe and secure deployment of ML systems into real world environment domains. 

## Authors

Aryan Aryan — arar00002@stud.uni-saarland.de  
Ananya Bhardwaz — anbh00002@stud.uni-saarland.de
