# ATML Programming Assignment 1

This repository contains four tasks on visual representations, domain adaptation, domain generalization, and open-set recognition. The sections below explain how to run the code in Google Colab and where to find the outputs.

The main random seed is **6304**. Saved results are in each task's `results/` folder, and [ATML_PA1_results_and_findings.ipynb](ATML_PA1_results_and_findings.ipynb) collects the results across tasks. Large checkpoints, datasets, and feature caches are not included in Git.

## Running in Google Colab

1. Open a Colab notebook and select **Runtime → Change runtime type → GPU**.
2. Upload the required ZIP through the Files panel. Uploaded files should appear under `/content/`.
3. Run the task's launch cell below, or use its provided notebook.
4. Allow Google Drive access when requested. Keep the checkpoints as well as the smaller report bundles, especially because Task 3 needs Task 2's Source-only checkpoint.

Tasks 3 and 4 have ZIP-based launchers. Task 2's existing Colab script clones the repository directly, so it does not need an uploaded ZIP. Task 1 runs through separate commands rather than a dedicated Colab launcher.

### Preparing the ZIP files

ZIP files are excluded from Git. If the prepared archives are unavailable, create them from the repository folders. The folders must be at the ZIP root, without an extra enclosing directory:

| Archive | Required contents at ZIP root |
| --- | --- |
| `task1_colab.zip` | `task1/` and the root `requirements.txt` |
| `task3_colab.zip` | `task3/`, `task2/`, and `shared/` |
| `task4_colab.zip` | `task4/` |

Include the code, requirements, YAML settings, and saved splits. Task 3's `shared/splits/pacs_sketch_seed6304.json` must match the split from the Task 2 run being reused. Exclude raw datasets, caches, environments, and large checkpoints from the code ZIPs; the Task 3 baseline checkpoint is supplied separately.

## Task 1: Inductive biases and feature representations

### How to run

Upload `task1_colab.zip`, then run:

```python
import zipfile

with zipfile.ZipFile('/content/task1_colab.zip') as archive:
    archive.extractall('/content/ATML_task1')
```

In the next Colab cell:

```python
%cd /content/ATML_task1
!pip install -r requirements.txt
!python -m task1.data.make_subset
!python -m task1.scripts.run_task1
!python -m task1.analysis.evaluate_bias
!python -m task1.analysis.translation
!python -m task1.analysis.patch_structure
```

These commands download STL-10, create the splits, train the classifier heads, and evaluate clean images, colour changes, translation, and patch shuffling. Pretrained backbone weights download when the models are first loaded.

Cue conflicts also require the external [pytorch-AdaIN code and weights](https://github.com/naoto0804/pytorch-AdaIN):

```python
!mkdir -p task1/external
!git clone https://github.com/naoto0804/pytorch-AdaIN.git task1/external/pytorch-AdaIN
```

Download the weights using that repository's instructions and place `vgg_normalised.pth` and `decoder.pth` inside `task1/external/pytorch-AdaIN/models/`. Then run:

```python
!python -m task1.data.make_cue_conflicts
```

The generator skips work if `task1/results/cue_conflicts/manifest.csv` already exists. To regenerate the reported images, first back up that reviewed manifest and temporarily move it out of the folder. Generate the images, then restore the original manifest before evaluation. This preserves the recorded 212 accepted and 38 rejected candidates. For a new experiment, review the generated images and fill in the manifest before examining model predictions.

Finally, run:

```python
!python -m task1.analysis.evaluate_bias --cue-conflicts
!python -m task1.analysis.feature_similarity
!python -m task1.analysis.representation
```

Feature similarity and t-SNE require the clean and transformed feature caches produced by the earlier commands. Task 1 writes to local Colab storage, so download its results and preserve any needed checkpoints/caches before disconnecting. Existing caches may be reused; use a fresh extracted folder when changing settings.

### How the code is organised

- `data/make_subset.py`: creates the 4,000 training, 1,000 validation, and balanced 500-image test split.
- `models/backbones.py`: loads frozen ResNet-50, ViT-B/16, and OpenAI CLIP ViT-B/32 through torchvision/OpenCLIP.
- `scripts/run_task1.py`: extracts features, trains linear heads, and evaluates clean images and zero-shot CLIP.
- `data/make_cue_conflicts.py`: uses AdaIN to create shape–texture conflicts.
- `analysis/evaluate_bias.py`: evaluates colour changes or cue conflicts with `--cue-conflicts`.
- `analysis/translation.py` and `patch_structure.py`: evaluate translations and patch shuffling.
- `analysis/feature_similarity.py` and `representation.py`: compute feature similarities and t-SNE visualisations.
- `results/`: saved splits, metrics, reviewed cue-conflict manifest, and figures.

Task 1 settings are recorded in `configs/task1.yaml`, but the current scripts use constants in the Python files rather than loading this YAML.

## Task 2: Unsupervised domain adaptation

### How to run

Open [task2/colab_task2.ipynb](task2/colab_task2.ipynb), select a GPU, and run its cell. Alternatively, paste the complete contents of [task2/scripts/colab_task2.py](task2/scripts/colab_task2.py) into a Colab cell.

The launcher clones this public repository, installs dependencies, downloads PACS, mounts Drive, and runs all experiments. It trains Source-only, DAN, DANN, and CDAN, plus DAN strengths 0.1 and 10. The main DAN run supplies strength 1.

Outputs are saved to:

```text
MyDrive/ATML_PA1/task2_results/
MyDrive/ATML_PA1/task2_report_bundle.zip
```

Keep the full `task2_results/` folder on Drive for Task 3, including `source_only/best.pt`. The report bundle excludes large checkpoints and feature arrays. Before preparing the Task 3 code ZIP, ensure its shared split matches this run's `task2_results/splits.json` exactly.

Completed runs are skipped on rerun. An interrupted unfinished run restarts from its fixed seed. Use the same code and settings when continuing an existing run; the launcher does not automatically update an existing checkout.

### How the code is organised

- `shared/pacs.py` and `shared/pacs_protocol.py`: load PACS and define the shared source/target protocol and splits.
- `configs/`: base settings and method-specific configurations.
- `models/`: ResNet-18 backbone, class head, and domain discriminator.
- `methods/`: Source-only classification, DAN's MMD alignment, DANN's gradient reversal, and CDAN's class-conditioned alignment.
- `train.py`: trains models and selects checkpoints by mean source-validation macro-F1.
- `evaluate_final.py` and `evaluation/`: evaluate fixed checkpoints, domain separability, class changes, and failures.
- `scripts/run_task2.py`: runs the complete comparison and strength study.
- `results/final/`: comparison tables, loss curves, confusion matrices, and failure examples.

Photo, Art Painting, and Cartoon provide labeled training data. Adaptation methods also see unlabeled Sketch images. Target labels are used only after checkpoint selection for final evaluation.

## Task 3: Domain generalization

### How to run with Drive

Upload `task3_colab.zip`. Keep the complete Task 2 outputs at `MyDrive/ATML_PA1/task2_results/`. Paste [task3/scripts/colab_task3.py](task3/scripts/colab_task3.py) into a Colab cell and run it, or use [task3/colab_task3.ipynb](task3/colab_task3.ipynb).

The launcher extracts the ZIP, checks the Task 2 checkpoint and split, installs dependencies, prepares PACS, and starts training. It reuses Source-only unchanged as ERM, trains DAN-DG, and trains SAM at radii 0.01, 0.05, and 0.1. Radius 0.05 is the main SAM run.

Outputs are saved to:

```text
MyDrive/ATML_PA1/task3_results/
MyDrive/ATML_PA1/task3_report_bundle.zip
```

### How to run without Drive

Upload these three files to `/content/`:

```text
task3_colab.zip
task2_report_bundle.zip
task2_source_only_best.pt
```

The last file must be the original Task 2 `source_only/best.pt`, renamed as shown. Paste [task3/scripts/colab_task3_local.py](task3/scripts/colab_task3_local.py) into a cell and run it. Download both `/content/task3_report_bundle.zip` and `/content/task3_full_backup.zip` before disconnecting, because local Colab files are temporary.

For either launcher, completed runs are reused. Interrupted unfinished runs restart from their fixed seeds rather than resuming a partial epoch.

### How the code is organised

- `methods/erm.py`: uses ordinary source classification; the runner reuses the exact Task 2 baseline checkpoint.
- `methods/dan_dg.py`: aligns the three source-domain pairs using their average MMD.
- `methods/sam.py`: applies SAM's parameter perturbation and second gradient calculation.
- `selection/source_validation.py`: handles source-only selection inputs.
- `evaluation/source_diagnostics.py`: measures source performance, source-domain separability, and a common sharpness proxy.
- `evaluate_sketch.py`: performs final Sketch evaluation after checkpoint locking and source diagnostics.
- `scripts/run_task3.py`: checks baseline reuse and runs the main methods and radius study.
- `results/source_diagnostics/` and `results/final/`: diagnostics, comparison tables, training curves, and class-level failures.

Task 3 reuses Task 2/shared model and data utilities. Sketch is unavailable during training, selection, and source diagnostics. Settings must remain independent of Task 2 target results.

## Task 4: Open-set recognition

### How to run

Upload `task4_colab.zip` and run this cell:

```python
import zipfile
import runpy

with zipfile.ZipFile('/content/task4_colab.zip') as archive:
    archive.extractall('/content/ATML_task4')

runpy.run_path(
    '/content/ATML_task4/task4/scripts/colab_task4.py',
    run_name='__main__',
)
```

Authorize Drive access. The launcher installs dependencies, downloads CIFAR, trains Vanilla and GCSC for 100 epochs each, and fine-tunes PROSER for 50 epochs. It then extracts model outputs, calibrates thresholds, and evaluates unknown rejection. No earlier task checkpoint is needed.

Persistent outputs are saved to:

```text
MyDrive/ATML_PA1/task4_results/
MyDrive/ATML_PA1/task4_cache/
MyDrive/ATML_PA1/task4_report_bundle.zip
MyDrive/ATML_PA1/task4_source.zip
```

The report bundle download starts automatically after successful completion. Download the full backup separately:

```python
from google.colab import files
files.download('/content/task4_full_backup.zip')
```

Rerun with the same ZIP and Drive account to continue an interrupted run. Task 4 resumes from the last saved epoch; completed methods are skipped. Do not run two notebooks against the same output folders.

### How the code is organised

- `data/`: downloads CIFAR-10, creates known train/validation splits, and selects the fixed near/far CIFAR-100 test classes.
- `models/resnet_cifar.py`: defines the CIFAR-sized ResNet-18.
- `methods/vanilla.py` and `gcsc.py`: standard training and stronger RandAugment training.
- `methods/proser.py` and `manifold_mixup.py`: placeholder learning and mixing features from different known classes.
- `scores/`: MSP, MLS, Energy, Mahalanobis, and PROSER's placeholder score.
- `train.py`: trains models and selects checkpoints using known validation accuracy.
- `extract_outputs.py`: saves features and logits.
- `evaluation/calibration.py`: sets thresholds using known validation data only.
- `evaluate_osr.py` and `evaluation/`: calculate AUROC, rejection rates, and failure examples.
- `results/`: score comparisons, model comparisons, histories, thresholds, ROC curves, and selected failures.

CIFAR-10 supplies known classes. CIFAR-100 supplies 800 near and 800 far unknown test images, used only for final evaluation. The rejection threshold retains 95% of known validation images. Optional RPL is not implemented.

## Reading the outputs

Each task preserves small numerical result files and figures under `results/`. Tasks 2–4 also save resolved settings, training histories, environment details, and checkpoint hashes. The report bundles contain this smaller evidence; full backups additionally preserve checkpoints and caches needed for later evaluation.

Most saved rates are fractions, while fields ending in `_pp` are percentage-point changes. The results notebook presents comparisons across tasks. Exact numerical reproduction can vary with GPU and package versions, so retain the recorded environment and original source when resuming runs.

## Code reuse

Task 1 directly uses the external AdaIN implementation and released weights. Torchvision supplies models and dataset loaders; OpenCLIP supplies CLIP; scikit-learn supplies metrics and feature diagnostics. The PACS download URL follows DomainBed, with a pinned mirror fallback documented in the Task 2 README. PROSER's additional detection score follows the authors' `CONF_DeltaP` construction, as documented in the Task 4 README. Code and documentation were developed with OpenAI Codex assistance.
