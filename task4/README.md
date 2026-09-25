# Task 4 — Open-set recognition

Implements the required Vanilla, GCSC and PROSER experiments, with no optional RPL extension. All comments start lowercase. Task 4 is independent of Tasks 1–3 and requires no previous checkpoint.

## Colab

Use a new GPU notebook. Upload `task4_colab.zip` to `/content`, then run:

```python
import zipfile
import runpy

# extract the standalone task 4 code
with zipfile.ZipFile('/content/task4_colab.zip') as archive:
    archive.extractall('/content/ATML_task4')

# train all required methods, evaluate, and prepare report archives
_ = runpy.run_path(
    '/content/ATML_task4/task4/scripts/colab_task4.py',
    run_name='__main__',
)
```

Authorize Google Drive when prompted. Authentication must succeed before training starts. The runner downloads CIFAR directly through torchvision. It trains Vanilla for 100 epochs, GCSC for 100 epochs, and PROSER for 50 epochs; it then extracts outputs and produces all required comparisons. Training time depends on the assigned GPU. No additional cell is needed to start the experiments. The equivalent notebook is `colab_task4.ipynb`.

The report ZIP download starts automatically after successful completion. Also run:

```python
from google.colab import files

# preserve checkpoints, cached outputs, and the exact code for reproducibility
files.download('/content/task4_full_backup.zip')
```

Checkpoints and results are written after every completed epoch to `MyDrive/ATML_PA1/task4_results/`; cached outputs are in `MyDrive/ATML_PA1/task4_cache/`. The dataset stays local for fast training. `task4_source.zip` preserves the exact source in the same Drive folder, and the final report ZIP is also copied there. If authentication fails, training does not start. Drive writes must finish successfully to preserve an epoch. If automatic downloads are blocked, use Colab's Files panel.

### Resume

Rerun the launch cell with the **same source ZIP and Google Drive account**, including in a new GPU runtime. Completed methods are skipped; unfinished methods resume from `last.pt` after their last successfully saved epoch. An interrupted partial epoch is repeated. The dataset is downloaded again if missing, but checkpoints and cached outputs remain on Drive.

Use `MyDrive/ATML_PA1/task4_source.zip` if you need the original source archive again (upload it as `task4_colab.zip`). Code/configuration mismatches stop the run rather than combining different experiments. This updated launcher does not automatically import an older local-only run; keep that run's files and original source if one is already in progress.

Epoch-boundary resume restores model, optimizer, schedule and history; seeds and augmentations are reconstructed. Different library/GPU versions can change numerical results. The original environment and package versions are recorded. Do not run two notebooks against the same results folder simultaneously.

## Fixed protocol

- CIFAR-10 official training partition: stratified 90/10 split, seed 6304; 45,000 training and 5,000 validation images. The full 10,000-image test set is reserved for evaluation.
- CIFAR ResNet-18: no pretrained weights, 3×3 stride-1 input convolution, no initial max-pool, original 32×32 images.
- Vanilla/GCSC: identical initial parameter hashes, SGD, learning rate 0.1, momentum 0.9, weight decay 0.0005, cosine schedule, batch 128, 100 epochs. GCSC adds only RandAugment(2, 9) after crop/flip.
- Standard normalization: mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616). Crop/flip randomness is matched per image/epoch; RandAugment uses a separate random stream.
- Selection: highest known validation accuracy, earliest epoch on ties; no early stopping.
- PROSER: starts from selected Vanilla, adds five random linear dummy classifiers, fine-tunes the entire network for 50 epochs with learning rate 0.001 and the specified SGD/cosine settings.
- Every minibatch is split equally. Classifier loss is CE on ten known responses plus the maximum dummy response, plus β=1 times CE toward that dummy after masking the true known response. Data loss mixes different-class representations after layer2 with a Beta(2,2) coefficient, then applies CE toward the maximum dummy. Total loss is classifier loss + γ=0.1 × data loss. Random-permutation pairs with matching classes are discarded; an empty valid set contributes zero. The actual 45,000-image training set has an even final batch of 72.

The maximum over five dummies follows the multiple-placeholder construction in the [PROSER paper](https://arxiv.org/abs/2103.15086). The [authors' implementation](https://github.com/LAMDA-CL/CVPR21-Proser/blob/main/proser_unknown_detection.py) supplies the additional detection score: collapse the strongest dummy, softmax at temperature 1024 with dummy bias 0, then dummy probability minus maximum known probability (`CONF_DeltaP`). The assignment's 95th-percentile validation rule is applied to that score. No unknown-dependent bias search or AUROC direction flipping is used. This code is an independent implementation; the assignment's architecture, five dummies and loss weights take precedence over reference defaults.

## Scores and evaluation

Vanilla MSP = 1 − max softmax; MLS = −max logit; Energy = −logsumexp at temperature 1; Mahalanobis = minimum class distance using a shared within-class diagonal covariance. Means and covariance use only unaugmented training features; covariance divides by N and adds 1e-6 per dimension.

Thresholds are the linear-interpolated 95th percentile of each model/score's known validation unknownness. Accept when score ≤ threshold. AUROC treats unknowns as positives. `*_fpr_at_val95tpr` is unknown acceptance at that fixed validation threshold, not a threshold selected from the test ROC. CSA uses the ten known logits even for PROSER. All reported rates are fractions in [0,1].

Unknowns are the fixed 800 near and 800 far CIFAR-100 **test** images. CIFAR-100 training data is never instantiated. All three checkpoints, the code, split, score definitions and calibration are hashed and fixed before unknown images are loaded. Near: bus, pickup_truck, motorcycle, tractor, wolf, fox, leopard, camel. Far: bottle, bowl, chair, clock, keyboard, mushroom, sunflower, wardrobe.

## Saved evidence

`task4_report_bundle.zip` contains files relative to the results directory, ready to extract into `task4/results/`:

- `posthoc_comparison.csv`: four scores on the same frozen Vanilla outputs.
- `trained_model_comparison.csv`: Vanilla/GCSC/PROSER MLS plus PROSER placeholder score.
- `all_results.csv/json`: CSA, near/far/all AUROC, threshold, validation/test acceptance and rejection/FPR.
- `vanilla_roc.png`: MSP, MLS and Mahalanobis, with near/far ROC curves.
- `failure_examples.png`, `selected_failures.csv`, `all_false_acceptances.csv`: actual unknowns accepted by Vanilla MLS, including class, prediction, score and threshold. Selection prioritizes confident errors across different unknown classes. If fewer than three failures exist in a group, counts explicitly report the shortfall; no examples are invented.
- `unknown_per_class.csv`, `training_curves.png`, each method's history/config/selection records, exact split, package versions, console log and protocol hashes.

For the report, inspect the six selected images and fill `semantic_interpretation` with whether each confusion is plausible or surprising and why. Compare the measured CSA and near/far rejection changes from Vanilla to GCSC and PROSER. Discuss augmentation versus classifier/data placeholders using these results; the code does not invent experimental conclusions.

`task4_full_backup.zip` additionally contains `task4_results/` with best and resumable checkpoints, fitted Mahalanobis parameters and score arrays; `task4_cache/` with features, logits, labels and original image indices; and `source/task4/` with the exact implementation. Keep this ZIP locally, not in Git. No datasets are bundled.

## Local execution and checks

Requires Python 3.11 or later. From the repository root:

```bash
python -m pip install -r task4/requirements.txt
python -m task4.scripts.run_task4 --data-root data
python -m unittest discover -s task4/tests -v
```

The structure follows the manual: configs, methods, data, models, scores, cache, evaluation, train.py, extract_outputs.py, evaluate_osr.py, results and README. Additional `scripts/`, `tests/` and `utils.py` provide orchestration and verification. The smoke tests use synthetic data; they do not replace the full Colab experiment or establish model accuracy.
