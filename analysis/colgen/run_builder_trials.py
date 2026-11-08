"""Reproduce per-builder and direct-solver comparisons, leaving lock windows between jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

here = Path(__file__).resolve().parents[2]
out = here / 'analysis/colgen/builder_data'
python = sys.executable
base = here / '.agent/builder-base'
benchmark = str(here / 'analysis/colgen/builder_benchmark.py')
locked = str(here / 'analysis/colgen/builder_locked.py')
solve = str(here / 'analysis/colgen/builder_solve.py')
capture = '/Users/victor/Documents/Research/congestion-utm-freespace/.agent/runs/colgen-metal-20261105-220722/pricing_captures/round_0001.pkl'
jobs = []
for repeat in (1, 2):
    for stage in (('corridor', 'geometry', 'roles', 'endpoint') if repeat == 1 else
                  ('endpoint', 'roles', 'geometry', 'corridor')):
        name = f'{stage}-prep-{repeat}'
        jobs.append((name, [python, benchmark, '--out', str(out / (name+'.json')),
                           '--stages', stage, '--early-only'], here))
for name, source, controls in [('base-prep-2', base, False), ('all-prep-2', here, False),
                               ('base-controls', base, True), ('all-controls', here, True)]:
    jobs.append((name, [python, benchmark, '--out', str(out / (name+'.json')),
                       '--source', str(source), *(['--controls'] if controls else [])], here))
for repeat in (1, 2):
    for workers in (0, 12):
        for label, source in ([('base', base), ('all', here)] if repeat == 1 else
                              [('all', here), ('base', base)]):
            name = f'{label}-solve-w{workers}-{repeat}'
            jobs.append((name, [python, locked, str(out / (name+'.process.json')), python, solve,
                               '--direct', '--capture', capture, '--out', str(out/name),
                               '--prepare-workers', str(workers)], source))
for name, command, source in jobs:
    process = out / (name + '.launcher.json')
    if process.exists():
        if json.loads(process.read_text())['exit_code']:
            raise SystemExit(f'Prior failure: {name}')
        continue
    print('Start', name, flush=True)
    started = time.perf_counter()
    with (out / (name+'.log')).open('w') as log:
        result = subprocess.run(command, cwd=here, env=dict(os.environ, PYTHONPATH=str(source),
                                PYTHONHASHSEED='0'), stdout=log, stderr=subprocess.STDOUT)
    process.write_text(json.dumps(dict(command=command, source=str(source),
                       process_wall_including_lock_wait_s=time.perf_counter()-started,
                       exit_code=result.returncode), indent=2)+'\n')
    print('Done', name, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    time.sleep(2)
