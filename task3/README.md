# Task 3: Domain Generalization

Implements all required Task 3 steps using Photo, Art Painting, and Cartoon only
until final evaluation. The Task 2 Source-only checkpoint is reused unchanged as
ERM. New runs train DAN-DG and SAM, plus the prescribed SAM radius study.

## Run in Google Colab

1. Enable a GPU runtime and upload `task3_colab.zip` into Colab's Files panel.
2. Paste `task3/scripts/colab_task3.py` into a cell, or open
   `task3/colab_task3.ipynb` and run its cell.
3. Allow Google Drive access. Keep the full Task 2 outputs at
   `MyDrive/ATML_PA1/task2_results`, including `source_only/best.pt`.

The upload ZIP includes Task 3 and its existing Task 2/shared code dependencies.
It excludes datasets and previous result files. No GitHub push is required to run
this upload-based workflow. The script reuses `/content/data/PACS` when available;
otherwise it invokes the existing downloader with its mirror fallback.

There are **four new training runs**: DAN-DG (weight 1), SAM (radius 0.05),
SAM (radius 0.01), and SAM (radius 0.1). Each has at most 30 source epochs with
patience 5. ERM is copied and verified, not retrained. SAM requires two passes per
batch. No target scores are printed until training and source diagnostics finish.

Results are saved directly to `MyDrive/ATML_PA1/task3_results`. The compact bundle
is `MyDrive/ATML_PA1/task3_report_bundle.zip`. Completed runs are skipped on retry;
an interrupted run restarts from its fixed seed, not from a partial optimizer
state. Keep the same code when resuming: code, split, and checkpoint hashes are
checked. Exact code is also preserved in `task3_results/code_snapshot.zip`.

## Repository structure

```text
task3/
  configs/erm.yaml dan_dg.yaml sam.yaml
  models/backbone.py classifier_head.py
  methods/erm.py dan_dg.py sam.py
  selection/source_validation.py
  evaluation/domain_metrics.py source_domain_separability.py sharpness.py
  evaluation/source_diagnostics.py
  train.py
  evaluate_sketch.py
  utils.py
  scripts/run_task3.py colab_task3.py
  tests/test_task3.py
  results/
  colab_task3.ipynb
  requirements.txt
  README.md
```

The model wrappers, classification loss, MMD, transforms, and loaders reuse Task 2
code instead of maintaining different implementations of the same protocol.
`shared/splits/pacs_sketch_seed6304.json` must match the actual Task 2 split bytes.
The generated `erm/config.json` retains the original method name `source_only`;
`erm/reuse.json` records its Task 3 identity and checkpoint hash.

## Fixed protocol

- Same ImageNet-V1 ResNet-18 and seven-class linear head, initialization seed 6304,
  image transforms, source split, and AdamW settings as Task 2. AdamW uses learning
  rate 1e-4 and weight decay 1e-4, with no learning-rate schedule.
- Eight examples from each source per batch. One epoch is
  `ceil(total source training images / 24)` balanced updates. Independently
  shuffled loaders cycle and drop incomplete batches, exactly as in Task 2.
- BatchNorm running means/variances stay frozen in every training pass, while
  scale/bias remain trainable. CPU augmentation seeding does not reset GPU RNG.
- Checkpoints maximize mean macro-F1 over the three source validation domains;
  ties retain the earliest checkpoint. Five non-improving epochs stop training.
- DAN-DG adds weight 1 times the average of the three pairwise MMD values. Each
  pair uses the unchanged Task 2 biased MMD estimator and sum of three RBF kernels:
  `exp(-squared_distance / (2 * bandwidth))`, where bandwidth is 0.5, 1, or 2 times
  that pair's detached median off-diagonal squared distance. No feature
  normalization, target features, or class-conditional matching is added.
- SAM is non-adaptive. First compute source cross-entropy gradients, perturb all
  trainable backbone/head parameters along the global gradient direction with
  the chosen L2 radius, then compute a second loss/gradient on the same augmented
  batch. Restore exact original parameters before applying AdamW using those
  second-pass gradients. No clipping or extra loss is added.
- The main SAM radius remains 0.05. The controlled study fixes radii 0.01, 0.05,
  0.1 before any training. It reuses the main SAM run, and is not selected using
  Task 2 target scores. The expected effects are recorded in the experiment plan.

## Diagnostics and final evaluation

`source_diagnostics.py` accepts source entries only and runs after all checkpoints
are locked. It reports each domain's accuracy/macro-F1, the unweighted source mean,
and the minimum domain score for each metric separately.

The source-domain probe samples equal numbers from each source-validation domain,
uses a seeded stratified 70/30 split, and fits a three-class multinomial logistic
regression with C=1 and balanced weights. Standardization is fitted only on probe
training data. Sampling indices, split indices, domain order, and convergence
iterations are saved. Chance accuracy is one third.

The sharpness diagnostic uses the same fixed 32 validation images per source for
every model (96 total), evaluation preprocessing and evaluation mode, and one
normalized gradient-ascent step with radius 0.05. It saves original loss,
perturbed loss, and their difference. This diagnostic radius stays 0.05 even
when the SAM *training* radius changes. Model parameters, gradients, and modes
are restored afterward; diagnostics cannot alter the final model.

`evaluate_sketch.py` is the only Task 3 stage that loads Sketch images. It checks
all checkpoint hashes and completed source diagnostics first. Only then does it
compute target metrics, per-class changes relative to ERM, failure examples, and
the comparison with target-aware DAN from Task 2. These target results are final
analysis only. The plots' “source-only” baseline label refers to the reused ERM.

## Outputs to keep

| Location | Contents |
| --- | --- |
| `experiment_plan.json`, `splits.json` | fixed settings, hypothesis, split and code hashes |
| `environment.json`, `pip_freeze.txt`, `code_snapshot.zip` | runtime versions and original code |
| `<run>/best.pt`, `config.json`, `history.csv`, `complete.json` | checkpoint, settings, source-only training history, runtime |
| `erm/reuse.json` | evidence that ERM was copied unchanged from Task 2 |
| `checkpoints_locked.json` | checkpoint hashes fixed before source diagnostics and target evaluation |
| `source_diagnostics/source_results.csv` | per-source/mean/worst scores, domain probe, sharpness |
| `source_diagnostics/sharpness_batch.json` | exact 96-image diagnostic batch |
| `source_diagnostics/*_probe.json`, `*_sharpness.json` | diagnostic settings, split indices, losses |
| `final/main_comparison.csv` | ERM, DAN-DG, and main SAM results |
| `final/sam_study.csv`, `.png` | fixed radius study |
| `final/training_curves.png` | classification/MMD curves and source validation macro-F1 |
| `final/per_class.csv`, `*_classes.png`, `*_failure_cases.json`, `*_examples.png` | target class changes, confusions, selected examples |
| `final/task2_comparison.csv`, `task2_per_class_comparison.csv` | matching ERM/source-only and DAN-DG/DAN comparisons |
| `**/*.npz` | features, logits, labels and image identifiers |

Scores are fractions, while `_pp` columns are percentage-point differences. Source
mean and worst scores are computed across domains, not over a pooled source set.
The bundle excludes checkpoints, feature arrays, and the code ZIP; preserve these
separately on Drive. Add the small bundle contents to `task3/results/` for GitHub.

## Commands and checks

Run from the repository root with Task 2 and shared code present:

```bash
python -m pip install -r task3/requirements.txt
python -m task3.scripts.run_task3 --data-root data/PACS --task2-results /path/to/full/task2_results --output task3/results
python -m unittest task3.tests.test_task3 -v
```

For final evaluation only, after source diagnostics already exist:

```bash
python -m task3.evaluate_sketch --data-root data/PACS --task2-results /path/to/full/task2_results --output task3/results
```

Tests check pairwise MMD, SAM against an independent AdamW calculation, ascent
radius/restoration including exceptions, frozen BatchNorm in both SAM passes,
sharpness against its formula, exact ERM reuse, and a complete synthetic workflow.
The workflow test rejects any Sketch image read during training/source diagnostics
and verifies checkpoint-tampering detection. Synthetic tests are not PACS results;
full training still needs to run on your Colab GPU.

Code was written with OpenAI Codex coding assistance, using the assignment's
DAN-DG and SAM definitions. Existing project code is reused where noted;
torchvision supplies ResNet-18 and scikit-learn supplies metrics and probes.
No external SAM training implementation was copied. Write your own report
interpretation from the saved experimental evidence.

## Run without Google Drive mounting

Upload the updated `task3_colab.zip`, `task2_report_bundle.zip`, and the original
Task 2 `source_only/best.pt` renamed to `task2_source_only_best.pt`. Run
`task3/scripts/colab_task3_local.py` instead of the Drive runner. It verifies the
checkpoint hash, uses local Colab storage, and produces both
`/content/task3_report_bundle.zip` and `/content/task3_full_backup.zip`. The compact
bundle downloads automatically; download the full backup from the Files panel too.
Local files and unfinished runs can be lost when the runtime is recycled.
