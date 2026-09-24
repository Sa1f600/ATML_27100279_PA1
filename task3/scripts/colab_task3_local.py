"""run task 3 from uploaded files without mounting Google Drive."""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from google.colab import files

REPO = Path('/content/ATML_task3')
PREVIOUS = Path('/content/task2_local_results')
OUTPUT = Path('/content/task3_results')
DATA = Path('/content/data/PACS')


def extract(archive_path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not (destination / name).resolve().is_relative_to(destination.resolve()):
                raise ValueError(f'unsafe archive path: {name}')
        archive.extractall(destination)


def run(command):
    # show subprocess errors directly in the notebook instead of hiding the cause
    with (OUTPUT / 'console.log').open('a') as log:
        process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError('command failed; inspect the output above before retrying')


# these are uploads to the colab files panel, not files on mounted drive
required = ['task3_colab.zip', 'task2_report_bundle.zip', 'task2_source_only_best.pt']
for name in required:
    if not (Path('/content') / name).exists():
        raise FileNotFoundError(f'upload {name} to Colab first')
extract('/content/task3_colab.zip', REPO)
extract('/content/task2_report_bundle.zip', PREVIOUS)
shutil.copy2('/content/task2_source_only_best.pt', PREVIOUS / 'source_only/best.pt')
OUTPUT.mkdir(parents=True, exist_ok=True)
run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'task3/requirements.txt'])
import torch
if not torch.cuda.is_available():
    raise RuntimeError('enable a GPU runtime before training')

# check the baseline fingerprint before downloading data or starting training
import hashlib
import json
checkpoint = PREVIOUS / 'source_only/best.pt'
expected = json.loads((PREVIOUS / 'source_only/complete.json').read_text())['checkpoint_sha256']
with checkpoint.open('rb') as stream:
    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
if actual != expected:
    raise ValueError('wrong checkpoint: upload task2_results/source_only/best.pt')

# preserve the exact code bundle with the local output
snapshot = OUTPUT / 'code_snapshot.zip'
if not snapshot.exists():
    shutil.copy2('/content/task3_colab.zip', snapshot)
run([sys.executable, '-u', '-m', 'task2.scripts.download_pacs', '--destination', str(DATA)])
run([sys.executable, '-u', '-m', 'task3.scripts.run_task3', '--data-root', str(DATA),
     '--task2-results', str(PREVIOUS), '--output', str(OUTPUT)])

# save both compact report evidence and a full backup including checkpoints
bundle = Path('/content/task3_report_bundle')
if bundle.exists():
    shutil.rmtree(bundle)
for path in OUTPUT.rglob('*'):
    if path.is_file() and path.suffix not in {'.pt', '.npz', '.tmp', '.zip'}:
        target = bundle / path.relative_to(OUTPUT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
report = shutil.make_archive('/content/task3_report_bundle', 'zip', bundle)
backup = shutil.make_archive('/content/task3_full_backup', 'zip', OUTPUT)
print(f'Report bundle: {report}')
print(f'Full backup: {backup}')
print('Download both ZIPs before disconnecting; local Colab files are temporary.')
files.download(report)
