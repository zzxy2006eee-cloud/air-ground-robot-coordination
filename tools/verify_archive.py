"""Check source syntax and provenance without importing or running robot code."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'docs/source_manifest.json').read_text(encoding='utf-8'))
    errors = []
    for entry in manifest['files']:
        file = root / entry['destination']
        if not file.is_file():
            errors.append(f'Missing file: {entry["destination"]}')
        elif hashlib.sha256(file.read_bytes()).hexdigest() != entry['sha256']:
            errors.append(f'Changed archived file: {entry["destination"]}')
    sources = [p for folder in ['final_code', 'experiments', 'tools'] for p in (root / folder).rglob('*.py')]
    for file in sources:
        try:
            ast.parse(file.read_text(encoding='utf-8-sig'), filename=str(file))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f'{file.relative_to(root)}: {exc}')
    if errors:
        print('\n'.join(errors))
        return 1
    print(f'PASS: {len(manifest["files"])} original files match SHA-256; {len(sources)} Python files parse.')
    print('No camera, flight, chassis or actuator code was executed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
