"""Freeze source-only, independently reproducible column-generation ablations."""

from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/private/tmp/colgen-todos-baseline-20260910')
PREFIX = Path('freespace_sim/planner/colgen')


def function(source, name):
    """Read a top-level function including decorators only when callers need them."""
    begin = source.index('def ' + name + '(')
    end = source.find('\ndef ', begin + 1)
    return source[begin:end if end != -1 else len(source)]


def main():
    """Compose each ablation from the frozen baseline, then record content hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots-dir', type=Path, default=Path('/private/tmp'))
    parser.add_argument('--arms', nargs='+', choices=['corridor', 'shifts', 'bootstrap', 'combined'],
                        default=['corridor', 'shifts', 'bootstrap', 'combined'])
    args = parser.parse_args()
    old_network = (BASE / PREFIX / 'network.py').read_text()
    new_network = (ROOT / PREFIX / 'network.py').read_text()
    corridor = new_network[:new_network.index('def _column_endpoint_steps(')]
    corridor = corridor.replace(
        'tuple[Any, ...], tuple[int, frozenset[RowKey], tuple[tuple[int, int], ...]]',
        'tuple[Any, ...], frozenset[RowKey]')
    corridor += old_network[old_network.index('def column_claims('):]
    old_solver = (BASE / PREFIX / 'solver.py').read_text()
    new_solver = (ROOT / PREFIX / 'solver.py').read_text()
    shifts_solver = old_solver.replace(function(old_solver, '_add_departure_ladder'),
                                       function(new_solver, '_add_departure_ladder'))
    arms = {
        'corridor': {'network.py': corridor},
        'shifts': {'network.py': new_network, 'solver.py': shifts_solver},
        'bootstrap': {'network.py': corridor,
                      **{name: (ROOT / PREFIX / name).read_text()
                         for name in ('pricing.py', 'params.py')}},
        'combined': {p.name: p.read_text() for p in (ROOT / PREFIX).glob('*.py')},
    }
    for name, files in arms.items():
        if name not in args.arms:
            continue
        target = args.snapshots_dir / f'colgen-todos-{name}-20260910'
        shutil.copytree(BASE, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for relative, source in files.items():
            (target / PREFIX / relative).write_text(source)
        manifest = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(target.rglob('*'))
                    if p.is_file() and p.name != 'source-manifest.json'}
        (target / 'source-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
        print(name, target, len(manifest))


if __name__ == '__main__':
    main()
