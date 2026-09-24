"""paste this complete script into one Google Colab cell after enabling a GPU."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive

# mount drive so completed runs and checkpoints survive a disconnected session
drive.mount('/content/drive')
REPOSITORY = 'https://github.com/Sa1f600/ATML_27100279_PA1.git'
REPO_DIR = Path('/content/ATML_27100279_PA1')
OUTPUT_DIR = Path('/content/drive/MyDrive/ATML_PA1/task2_results')
DATA_DIR = Path('/content/data/PACS')

# push task2 and shared to github before running this cell
if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', REPOSITORY, str(REPO_DIR)], check=True)
os.chdir(REPO_DIR)
if not Path('task2/scripts/run_task2.py').exists():
    raise RuntimeError('task2 is missing; push it to GitHub, then clone the updated repository')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                'task2/requirements.txt'], check=True)
import torch
if not torch.cuda.is_available():
    raise RuntimeError('choose Runtime > Change runtime type > GPU, then rerun this cell')

# download locally for faster image loading; only results are written to drive
subprocess.run([sys.executable, '-m', 'task2.scripts.download_pacs',
                '--destination', str(DATA_DIR)], check=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# tee the complete console log to drive and keep partial results if execution fails
command = [sys.executable, '-u', '-m', 'task2.scripts.run_task2',
           '--data-root', str(DATA_DIR), '--output', str(OUTPUT_DIR)]
with (OUTPUT_DIR / 'console.log').open('a') as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='')
        log.write(line)
        log.flush()
    return_code = process.wait()
if return_code:
    raise RuntimeError(f'run failed with exit code {return_code}; see console.log on Drive')

# make a compact report bundle; large checkpoints and feature arrays stay on drive
bundle = Path('/content/task2_report_bundle')
if bundle.exists():
    shutil.rmtree(bundle)
for path in OUTPUT_DIR.rglob('*'):
    if path.is_file() and path.suffix not in {'.pt', '.npz', '.tmp'}:
        target = bundle / path.relative_to(OUTPUT_DIR)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
archive = shutil.make_archive('/content/task2_report_bundle', 'zip', bundle)
shutil.copy2(archive, OUTPUT_DIR.parent / 'task2_report_bundle.zip')
print(f'All outputs: {OUTPUT_DIR}')
print(f'Report bundle: {OUTPUT_DIR.parent / "task2_report_bundle.zip"}')
print('Copy splits.json to shared/splits/pacs_sketch_seed6304.json before committing.')
