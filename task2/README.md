# Task 2: Unsupervised Domain Adaptation

Implements all six required Task 2 steps: Source-only, DAN, DANN, CDAN, final
alignment/class analysis, and a controlled DAN study with MMD weights 0.1, 1, 10.
There are **six training runs**: the four main methods plus two extra DAN runs.
The weight-1 DAN checkpoint is reused in the study. Task 1 is unchanged.

## Google Colab

1. Push `task2/` **and** `shared/` to your repository.
2. Open `task2/colab_task2.ipynb` in Colab and select a GPU runtime.
3. Run the cell and authorize Google Drive mounting. Alternatively, paste the
   complete contents of `task2/scripts/colab_task2.py` into one Colab cell.

The script clones `https://github.com/Sa1f600/ATML_27100279_PA1`, installs the
Task 2 dependencies, downloads PACS, runs all six experiments, and then evaluates
all fixed checkpoints. It writes directly to `MyDrive/ATML_PA1/task2_results`.
No target metrics appear during training. No manual hyperparameter search is needed.

PACS downloads from the archive linked in the
[DomainBed download script](https://github.com/facebookresearch/DomainBed/blob/main/domainbed/scripts/download.py).
If the public Drive link fails, the downloader uses the pinned
[Azeez577/PACS mirror](https://huggingface.co/datasets/Azeez577/PACS/tree/fb1ee820df83b4e9957251d9f6973eb6d424c494).
Its archive listing was checked for the four domains and 9,991 images; image bytes
were not compared with the original archive. Full GPU training has not been
verified by a local end-to-end PACS run.

Completed training runs are skipped on rerun. An interrupted training run restarts
that run from the same seed; it does not resume an optimizer mid-epoch. A fixed
experiment refuses changed code, configurations, or split indices. After all
checkpoints are locked, rerunning performs analysis only and verifies their hashes.
Keep the same repository version when resuming. Existing Colab checkouts are not
silently updated. Start from a fresh checkout if your initial clone predates Task 2.

## Local commands

Run from the repository root:

```bash
python -m pip install -r task2/requirements.txt
python -m task2.scripts.download_pacs --destination data/PACS
python -m task2.scripts.run_task2 --data-root data/PACS --output task2/results
```

To repeat only the final analysis after checkpoint locking:

```bash
python -m task2.evaluate_final --data-root data/PACS --output task2/results
```

## Directory layout

```text
shared/
  pacs.py
  pacs_protocol.py
  splits/pacs_sketch_seed6304.json  # generated from actual PACS on the first run
task2/
  configs/base.yaml source_only.yaml dan.yaml dann.yaml cdan.yaml
  models/backbone.py classifier_head.py domain_discriminator.py
  methods/source_only.py dan.py dann.py cdan.py
  evaluation/metrics.py domain_separability.py class_analysis.py
  train.py
  evaluate_final.py
  utils.py
  scripts/run_task2.py download_pacs.py colab_task2.py
  tests/test_task2.py
  results/
  colab_task2.ipynb
  requirements.txt
  README.md
```

## Fixed protocol and implementation choices

- Sources: Photo, Art Painting, Cartoon. Target: all Sketch images, unlabeled
  during adaptation. Each source uses an 80/20 stratified split with seed 6304.
- ImageNet-V1 ResNet-18, 512-dimensional features, linear seven-class head;
  full fine-tuning. All BatchNorm running statistics remain at pretrained values;
  BatchNorm scale/bias parameters receive gradients.
- Resize to 256 x 256, random 224 crop and flip for training, center crop for
  evaluation; ImageNet normalization. Source-only never builds a target loader.
- AdamW, learning rate 1e-4, weight decay 1e-4, at most 30 epochs, patience 5.
  Strictly highest mean macro-F1 over the three source validation domains wins.
- Each update uses 8 examples per source and, for adaptation, 24 target examples.
  One source epoch is `ceil(total source training images / 24)` updates. Shuffled
  loaders cycle independently, dropping partial batches to retain exact balance.
  Smaller source domains are oversampled. Sampling/augmentation seeds and initial
  backbone/head weights are identical across runs. The adversarial discriminators
  introduce their own dropout; source augmentation RNG is isolated from it.
- DAN uses the biased MMD V-statistic (including kernel diagonals) and the **sum**
  of three RBF kernels. For squared distance `d2`, `k = exp(-d2/(2*b))`, where
  `b` is 0.5, 1, or 2 times the detached median off-diagonal squared distance.
  A 1e-8 floor handles coincident features. No feature normalization is added.
- DANN uses 512 -> 256 -> 2, ReLU and dropout 0.5; CDAN uses 3584 -> 256 -> 2.
  CDAN conditions on the full feature/probability outer product without detaching
  either factor. Both use unit domain-loss weight and the specified GRL schedule.
  Progress is measured against the maximum 30-epoch update budget, even if stopped
  early. No learning-rate scheduler or entropy conditioning is added.
- The final logistic probe uses equal source-validation and target counts,
  a stratified 70/30 split, C=1, balanced weights, and seed 6304. Feature scaling
  is fitted on probe-training data only. Sampling and split indices are saved.
- The strength-study hypothesis is locked in `experiment_plan.json` before
  training. Target scores are for final analysis, never checkpoint selection.
  Record your own expectation and interpretation in your report.

## Saved evidence for your report

All metric scores are fractions in [0, 1]; fields ending in `_pp` are percentage
point differences. Source means give equal weight to each of the three domains.

| File | Contents |
| --- | --- |
| `experiment_plan.json` | all settings, hypothesis, code/split hashes and selection rule |
| `splits.json` | exact source split paths and complete target paths |
| `environment.json`, `pip_freeze.txt` | runtime, device, Git commit, installed package versions |
| `<run>/config.json`, `history.csv`, `complete.json` | resolved settings, epoch losses/source scores, runtime |
| `<run>/best.pt` | selected backbone/head, discriminator if used, source scores and epoch |
| `checkpoints_locked.json` | hashes of every checkpoint fixed before target evaluation |
| `final/main_comparison.csv` | all source domains, source means, target scores, changes, domain separability |
| `final/alignment_study.csv`, `.png` | controlled DAN strength comparison |
| `final/loss_curves.png` | classification and raw MMD/domain losses across epochs |
| `final/per_class.csv`, `*_classes.png` | per-class changes and confusion matrices |
| `final/*_failure_cases.json`, `*_examples.png` | examples gained/lost and incorrect predictions with paths |
| `final/*_outputs.npz` | features, logits, labels, and image paths for every evaluated split |
| `final/*_domain_probe.json` | probe settings, sampled indices, split indices, iterations and held-out score |
| `final/summary.json` | machine-readable aggregate results |

The Colab script also creates `MyDrive/ATML_PA1/task2_report_bundle.zip` with
CSV/JSON evidence, figures, histories, and environment details. Large checkpoints
and feature arrays remain separately on Drive. Copy the small evidence files into
`task2/results/` for GitHub; `.gitignore` excludes large checkpoints/features.
Copy `splits.json` to `shared/splits/pacs_sketch_seed6304.json` and commit it too.
Do not infer experimental results from the code: these files appear after training.

## Reuse in Task 3

Reuse `source_only/best.pt` unchanged as Task 3 ERM, and reuse the exact source
split. Load only source entries during Task 3 training and diagnostics. Fix Task 3
settings independently of Task 2 target results. This implementation covers Task 2;
it does not implement or launch Tasks 3 or 4.

## Checks and attribution

```bash
python -m unittest task2.tests.test_task2 -v
```

Tests cover MMD, reversed gradients, conditional classifier gradients, frozen
BatchNorm statistics, target-label masking, deterministic splits, and a synthetic
six-run training/evaluation test. Synthetic tests are not PACS performance results.

Code was written with OpenAI Codex coding assistance. Model construction and
pretrained weights use torchvision; metrics and the diagnostic probe use
scikit-learn. No external training implementation was copied. The PACS archive URL
comes from DomainBed. Method references: [DAN](https://proceedings.mlr.press/v37/long15.html),
[DANN](https://jmlr.org/papers/v17/15-239.html), and
[CDAN](https://proceedings.neurips.cc/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html).
