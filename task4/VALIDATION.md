# Local verification

Nine automated tests passed on CPU using Python 3.13, PyTorch 2.8.0 and torchvision 0.23.0:

- Closed-form score values, threshold interpolation, AUROC orientation and unknown acceptance convention.
- Mahalanobis pooled within-class covariance and distances.
- PROSER classifier losses, true-class masking and strongest-dummy gradients.
- Different-class mixup and all-same-class minibatch handling.
- Actual CIFAR ResNet-18 forward/backward with five dummies, layer2 mixup shape and finite nonzero backbone/dummy gradients.
- Stratified disjoint splits and augmentation random-state preservation.
- Epoch-level resume matching uninterrupted parameters and histories exactly.
- End-to-end synthetic training, checkpoint selection, calibration, final evaluation, figures, PROSER initialization, checkpoint tamper rejection, and unknown-data access only after calibration.

The complete synthetic workflow uses a small network to keep tests fast; the architecture/gradient test uses the actual ResNet-18. No real CIFAR training or GPU execution has been performed locally. Full experiment metrics will be produced by the Colab run.

The persistence test checks preserved files on reopening the results folder, source archiving, and rejection of changed code without overwriting the original source. Real Google Drive authentication and mounted-drive failure behavior require Colab and were not tested locally.
