# Imported Task 2 experiment results

These files were imported from the completed Colab report bundle. Original
metrics, plots, histories, configurations, and experiment hashes are preserved.

- `final/main_comparison.csv`: four-method comparison.
- `final/alignment_study.csv`: fixed DAN strength study.
- `final/per_class.csv`: per-class target results.
- `final/*_classes.png`, `final/*_examples.png`: confusions and selected examples.
- `final/loss_curves.png` and each run's `history.csv`: training diagnostics.
- `import_verification.json`: archive and consistency checks, plus their limits.
- `splits.json`: identical to `shared/splits/pacs_sketch_seed6304.json`.

Large checkpoints and feature arrays were excluded from the report bundle and
remain in `MyDrive/ATML_PA1/task2_results` on Google Drive. Preserve
`source_only/best.pt` unchanged for Task 3 ERM. The local results are an evidence
archive, not a complete training-resume directory.

The local downloader was updated with a PACS mirror fallback after the code was
uploaded to Colab. The saved training code hash therefore differs from the current
local hash; it has not been rewritten. Resuming the original experiment requires
its original code and checkpoint files.

DANN and CDAN histories contain large loss spikes. Consult their original
histories before interpreting target scores; archive consistency checks do not
establish optimization stability or verify missing checkpoint bytes.
