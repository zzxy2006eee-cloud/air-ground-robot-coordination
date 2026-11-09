"""Run the archived processing pipeline in an isolated working directory.

This wrapper only prepares paths and invokes the original All.py. It does not
connect to hardware, modify algorithms, or certify the resulting motion path.
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import runpy
import shutil
import sys


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True, help='Input image (JPEG or PNG)')
    parser.add_argument('--output', type=Path, help='New, empty run directory')
    args = parser.parse_args()
    source = args.image.resolve()
    if not source.is_file():
        parser.error(f'Input image does not exist: {source}')
    output = (args.output or repo / 'runtime' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be empty; existing results will not be overwritten.')
    # Check dependencies before creating output files. Do not import hardware code.
    try:
        import cv2
        import numpy as np
        from sklearn.cluster import DBSCAN  # noqa: F401
    except (ImportError, ValueError) as exc:
        parser.exit(1, f'Processing dependency unavailable: {exc}\nUse a separate environment with requirements.txt.\n')
    image = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        parser.error('Input cannot be decoded as an image.')
    (output / 'input').mkdir(parents=True, exist_ok=True)
    (output / 'out_put').mkdir()
    # Preserve input bytes when possible; imread detects format by content.
    shutil.copy2(source, output / 'input' / 'fly.jpg')
    original_cwd = Path.cwd()
    original_path = sys.path[:]
    try:
        os.chdir(output)
        sys.path.insert(0, str(repo / 'final_code' / 'processing'))
        runpy.run_path(str(repo / 'final_code' / 'processing' / 'All.py'), run_name='__main__')
    except Exception as exc:
        print(f'Original pipeline stopped: {type(exc).__name__}: {exc}', file=sys.stderr)
        print(f'Intermediate files: {output}\nSee docs/KNOWN_LIMITATIONS.md.', file=sys.stderr)
        return 1
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_path
    print(f'Processing completed. Inspect outputs before hardware use: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
