# ATML Programming Assignment 1

Repository for Advanced Topics in Machine Learning Programming Assignment 1.

## Tasks

- `task1/`: inductive biases and feature representations
- `task2/`: unsupervised domain adaptation
- `task3/`: domain generalization
- `task4/`: open-set recognition

Task 1 contains reproducible code, configurations, saved split indices,
machine-readable metrics, and generated figures. Raw datasets and large
feature caches are excluded from version control.

Task 2 now includes all four UDA methods and a controlled DAN strength study.
See [the Task 2 README](task2/README.md) and open
[the Colab notebook](task2/colab_task2.ipynb) to run all Task 2 experiments.
The `shared/` folder contains the reusable PACS protocol for Tasks 2 and 3.

Task 3 reuses the Task 2 ERM checkpoint and trains source-only DAN-DG and SAM.
See [the Task 3 README](task3/README.md) and
[the Task 3 Colab notebook](task3/colab_task3.ipynb).

Task 4 implements Vanilla, GCSC and PROSER with validation-calibrated open-set evaluation.
See [the Task 4 README](task4/README.md) and
[the Task 4 Colab notebook](task4/colab_task4.ipynb). Its runner saves resumable checkpoints to Google Drive after each epoch
and creates report and full-backup archives.
