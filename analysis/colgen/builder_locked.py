"""Run one command under the shared CPU/Metal lock with an exact process manifest."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

out = Path(sys.argv[1])
command = sys.argv[2:]
lock_path = Path('/Users/victor/Documents/Research/congestion-utm-freespace/.agent/gpu-followups-20261106/metal-benchmark.lock')
with lock_path.open('a') as lock:
    print('Waiting for shared lock', flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    print('Acquired shared lock', flush=True)
    start = time.perf_counter()
    result = subprocess.run(command)
    out.write_text(json.dumps(dict(command=command, process_wall_s=time.perf_counter()-start,
                                   exit_code=result.returncode), indent=2)+'\n')
raise SystemExit(result.returncode)
