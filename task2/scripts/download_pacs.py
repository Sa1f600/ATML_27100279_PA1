"""download PACS, with a pinned mirror if the public Drive link fails."""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from shared.pacs_protocol import SOURCES, TARGET

MIRROR = ('https://huggingface.co/datasets/Azeez577/PACS/resolve/'
          'fb1ee820df83b4e9957251d9f6973eb6d424c494/PACS.zip')


def download(destination):
    destination = Path(destination).resolve()
    if all((destination / domain).is_dir() for domain in (*SOURCES, TARGET)):
        print(f'using existing PACS data: {destination}')
        return
    import gdown
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / 'PACS.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.zip.part')
        try:
            result = gdown.download(id='1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd',
                                    output=str(temporary), quiet=False)
            if result is None:
                raise RuntimeError('public Drive download failed')
        except Exception as error:
            print(f'public Drive unavailable ({type(error).__name__}); trying PACS mirror', flush=True)
            urllib.request.urlretrieve(MIRROR, temporary)
        if not zipfile.is_zipfile(temporary):
            raise ValueError('download did not return a valid ZIP archive')
        temporary.replace(archive)
    staging = destination.parent / 'pacs_extracted'
    staging.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        # reject archive paths that escape the extraction directory
        for entry in zipped.infolist():
            if not (staging / entry.filename).resolve().is_relative_to(staging.resolve()):
                raise ValueError(f'unsafe archive member: {entry.filename}')
        zipped.extractall(staging)
    roots = [p.parent for p in staging.rglob('photo') if p.is_dir()
             and all((p.parent / domain).is_dir() for domain in (*SOURCES, TARGET))]
    if len(roots) != 1:
        raise ValueError(f'expected one PACS root; found {roots}')
    if destination.exists():
        raise FileExistsError(f'incomplete destination exists: {destination}; inspect it before retrying')
    shutil.move(str(roots[0]), destination)
    print(f'PACS ready: {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', default='data/PACS')
    args = parser.parse_args()
    download(args.destination)
