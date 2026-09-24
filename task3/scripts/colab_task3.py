"""paste into one Colab cell with a GPU and task3_colab.zip uploaded."""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from google.colab import drive

# use a separate checkout so the previous task 2 code and results stay intact
REPO = Path('/content/ATML_task3')
UPLOAD = Path('/content/task3_colab.zip')
OUTPUT = Path('/content/drive/MyDrive/ATML_PA1/task3_results')
PREVIOUS = Path('/content/drive/MyDrive/ATML_PA1/task2_results')
DATA = Path('/content/data/PACS')

if UPLOAD.exists():
    REPO.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(UPLOAD) as archive:
        for name in archive.namelist():
            if not (REPO / name).resolve().is_relative_to(REPO.resolve()):
                raise ValueError(f'unsafe archive member: {name}')
        archive.extractall(REPO)
elif not (REPO / 'task3/scripts/run_task3.py').exists():
    raise FileNotFoundError('upload task3_colab.zip to the Colab Files panel first')
os.chdir(REPO)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'task3/requirements.txt'], check=True)
import torch
if not torch.cuda.is_available():
    raise RuntimeError('enable a GPU runtime, then rerun this cell')
drive.mount('/content/drive')

# require the actual saved baseline; do not replace it with a newly trained model
for name in ['source_only/best.pt', 'source_only/config.json', 'source_only/history.csv',
             'source_only/complete.json', 'splits.json', 'final/all_results.csv', 'final/per_class.csv']:
    if not (PREVIOUS / name).exists():
        raise FileNotFoundError(f'missing Task 2 output on Drive: {PREVIOUS / name}')

# use the exact split from the completed task 2 run
split_file = REPO / 'shared/splits/pacs_sketch_seed6304.json'
if split_file.read_bytes() != (PREVIOUS / 'splits.json').read_bytes():
    raise ValueError('uploaded split does not match your completed task 2 run')
download = subprocess.run([sys.executable, '-u', '-m', 'task2.scripts.download_pacs',
                           '--destination', str(DATA)], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
print(download.stdout)
download.check_returncode()
OUTPUT.mkdir(parents=True, exist_ok=True)

# preserve the exact code used for this run alongside the full drive results
snapshot = OUTPUT / 'code_snapshot.zip'
if not snapshot.exists():
    with zipfile.ZipFile(snapshot, 'w', zipfile.ZIP_DEFLATED) as archive:
        for folder in ('task2', 'task3', 'shared'):
            for path in (REPO / folder).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts and 'results' not in path.parts:
                    archive.write(path, path.relative_to(REPO))

command = [sys.executable, '-u', '-m', 'task3.scripts.run_task3', '--data-root', str(DATA),
           '--task2-results', str(PREVIOUS), '--split-file', str(split_file), '--output', str(OUTPUT)]
with (OUTPUT / 'console.log').open('a') as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='')
        log.write(line)
        log.flush()
    return_code = process.wait()
if return_code:
    raise RuntimeError(f'run failed with code {return_code}; the full error is in console.log on Drive')

# package report evidence while keeping large models and feature arrays on drive
bundle = Path('/content/task3_report_bundle')
if bundle.exists():
    shutil.rmtree(bundle)
for path in OUTPUT.rglob('*'):
    if path.is_file() and path.suffix not in {'.pt', '.npz', '.tmp', '.zip'}:
        target = bundle / path.relative_to(OUTPUT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
archive = shutil.make_archive('/content/task3_report_bundle', 'zip', bundle)
shutil.copy2(archive, OUTPUT.parent / 'task3_report_bundle.zip')
print(f'All outputs: {OUTPUT}')
print(f'Report bundle: {OUTPUT.parent / "task3_report_bundle.zip"}')
