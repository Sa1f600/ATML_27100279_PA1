"""Colab launcher: persistent Drive storage, visible logs, and downloadable archives."""

import os
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUT = Path('/content/task4_results')
CACHE = Path('/content/task4_cache')


def prepare_storage(drive_root):
    global OUTPUT, CACHE
    from task4.utils import code_hash
    folder = Path(drive_root) / 'ATML_PA1'
    OUTPUT, CACHE = folder / 'task4_results', folder / 'task4_cache'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    plan = OUTPUT / 'experiment_plan.json'
    if plan.exists() and json.loads(plan.read_text())['code_sha256'] != code_hash():
        raise ValueError('Drive contains a run from different code. Restore its original source ZIP before resuming.')
    # verify write access before expensive training and preserve the exact source
    probe = folder / '.task4_write_test'
    probe.write_text('ok')
    probe.unlink()
    temporary = folder / 'task4_source.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((REPO / 'task4').rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.py', '.yaml', '.md', '.txt', '.ipynb'}:
                archive.write(path, Path('task4') / path.relative_to(REPO / 'task4'))
    temporary.replace(folder / 'task4_source.zip')
    print(f'Persistent checkpoints: {OUTPUT}', flush=True)
    return folder


def bundle():
    # keep report evidence small; the full backup also preserves weights and cached outputs
    report = Path('/content/task4_report_bundle.zip')
    backup = Path('/content/task4_full_backup.zip')
    with zipfile.ZipFile(report, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT.rglob('*')):
            if path.is_file() and path.suffix not in {'.pt', '.npz', '.tmp', '.zip'}:
                archive.write(path, path.relative_to(OUTPUT))
    with zipfile.ZipFile(backup, 'w', zipfile.ZIP_DEFLATED) as archive:
        for folder in (OUTPUT, CACHE):
            for path in sorted(folder.rglob('*')):
                if path.is_file() and path.suffix != '.tmp':
                    archive.write(path, Path(folder.name) / path.relative_to(folder))
        for path in sorted((REPO / 'task4').rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.py', '.yaml', '.md', '.txt', '.ipynb'}:
                archive.write(path, Path('source/task4') / path.relative_to(REPO / 'task4'))
    print(f'\nReport bundle: {report}\nFull backup: {backup}', flush=True)
    return report, backup


def main():
    os.chdir(REPO)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                    str(REPO / 'task4/requirements.txt')], check=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('select Runtime > Change runtime type > GPU, then rerun this cell')
    # authentication failure stops here; never silently train without persistence
    from google.colab import drive
    drive.mount('/content/drive')
    if not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('Google Drive did not mount; training has not started')
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    folder = prepare_storage('/content/drive/MyDrive')
    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    print('Vanilla: 100 epochs; GCSC: 100 epochs; PROSER: 50 epochs, then evaluation.', flush=True)
    print('Checkpoints save to Drive after each epoch. Reconnect the same Drive account to resume.', flush=True)
    command = [sys.executable, '-u', '-m', 'task4.scripts.run_task4',
               '--data-root', '/content/data', '--output', str(OUTPUT), '--cache', str(CACHE)]
    environment = dict(os.environ, MPLCONFIGDIR='/content/task4_mpl', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    # stream child output explicitly so Colab shows progress and useful errors
    process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        with (OUTPUT / 'console.log').open('a') as log:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
        status = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        print('Stopped. The last completed epoch is saved; rerun to resume.', flush=True)
        raise
    finally:
        process.stdout.close()
    report, backup = bundle()
    shutil.copy2(report, folder / report.name)
    if status:
        raise RuntimeError(f'Task 4 failed (exit {status}); see the error above and task4_results/console.log')
    from google.colab import files
    files.download(str(report))
    print('The report download has started. Download the full backup with the next cell.', flush=True)


if __name__ == '__main__':
    main()
