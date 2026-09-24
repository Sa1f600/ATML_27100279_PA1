# Task 1: Inductive Biases and Feature Representations

This directory contains the reproducible Task 1 pipeline for STL-10 using frozen
ResNet-50, ViT-B/16, and OpenCLIP ViT-B/32 backbones.

## Planned workflow

1. Create deterministic train/validation splits and the balanced test subset.
2. Extract and cache frozen-backbone features.
3. Train the three linear classifier heads and evaluate zero-shot CLIP.
4. Generate shared color, translation, patch-shuffle, and cue-conflict inputs.
5. Evaluate prediction consistency, shape bias, coverage, and feature stability.
6. Produce translation curves and joint representation visualizations.

All random operations use seed `6304`. Generated data and machine-readable
results belong under `results/`; raw datasets should not be committed.
