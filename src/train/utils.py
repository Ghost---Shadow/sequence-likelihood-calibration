from pathlib import Path
import shutil


def rmrf_then_mkdir(path):
    try:
        shutil.rmtree(path)
    except Exception:
        ...

    Path(path).mkdir(parents=True)
