"""Offline release evidence for the research prototype (not production certification)."""
from __future__ import annotations
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from training.artifacts import atomic_write_json

def run_gate(out: Path, *, timeout_seconds=180):
    out.mkdir(parents=True,exist_ok=True)
    run=out/'run'
    steps=[
        ('compile',[sys.executable,'-m','compileall','-q','engine','environments',
                    'observations','actions','rewards','algorithms',
                    'credit_assignment','league','evaluation','training']),
        ('tests',[sys.executable,'-m','pytest','-q']),
        ('train',[sys.executable,'-m','training.cli','train','--episodes','2',
                  '--steps','8','--seed','1001','--out',str(run)]),
        ('infer',[sys.executable,'-m','training.cli','infer',
                  '--checkpoint',str(run/'policy.pt')]),
        ('evaluate_paired',[sys.executable,'-m','training.cli','evaluate',
                            '--checkpoint',str(run/'policy.pt'),'--episodes','3',
                            '--seed','9001','--side-swaps']),
        ('credit',[sys.executable,'-m','training.cli','credit','--seed','9001']),
        ('benchmark',[sys.executable,'-m','evaluation.benchmark',
                      '--episodes','1','--steps','12','--players-per-team','5'])]
    evidence={'schema_version':1,'python_version':platform.python_version(),
              'platform':platform.platform(),
              'purpose':'local research prototype release gate',
              'passed':False,'steps':[]}
    for label,command in steps:
        start=time.monotonic()
        try:
            proc=subprocess.run(command,cwd=Path(__file__).resolve().parents[1],
                                capture_output=True,text=True,timeout=timeout_seconds,
                                check=False)
            result={'name':label,'command':command,'exit_code':proc.returncode,
                    'elapsed_s':round(time.monotonic()-start,3),
                    'stdout_tail':proc.stdout[-4000:],
                    'stderr_tail':proc.stderr[-4000:]}
        except subprocess.TimeoutExpired as exc:
            result={'name':label,'command':command,'exit_code':None,
                    'elapsed_s':round(time.monotonic()-start,3),
                    'error':f'process timed out after {exc.timeout}s'}
        evidence['steps'].append(result)
        atomic_write_json(out/'release_report.json',evidence)
        print(f"[{label}] {'PASS' if result['exit_code']==0 else 'FAIL'} "
              f"{result['elapsed_s']:.2f}s",flush=True)
        if result['exit_code']!=0:
            return evidence
    evidence['passed']=True
    atomic_write_json(out/'release_report.json',evidence)
    return evidence

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/release_gate'))
    parser.add_argument('--timeout',type=int,default=180)
    args=parser.parse_args(argv)
    if args.timeout<1:
        parser.error('--timeout must be positive')
    if (args.out/'run').exists():
        parser.error('release gate output already contains a run; choose a fresh --out')
    result=run_gate(args.out,timeout_seconds=args.timeout)
    print(json.dumps({'passed':result['passed'],
                      'report':str(args.out/'release_report.json')},indent=2))
    return 0 if result['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
