# Shared PACS split

`pacs_sketch_seed6304.json` is generated from the actual dataset on the first run
and copied into the results directory. It is not fabricated before PACS is present.
Commit the generated JSON after the Colab run, and reuse it unchanged for Task 3.

Each source domain uses a stratified 80/20 split with seed 6304. Paths are relative
to the PACS root. The target list contains paths only, without numeric labels.
Task 3 must load only the source entries during training and diagnostics; it must
not construct a Sketch loader or rerun target-aware Task 2 diagnostics.
